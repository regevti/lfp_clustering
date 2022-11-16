import stumpy
import time
import inspect
import gzip
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
from numba import cuda
from typing import Tuple
from functools import lru_cache
from scipy.signal import decimate, find_peaks, resample
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import utils
from readers import Reader, NeuralynxReader, OpenEphysReader
from utils import consecutive
import matplotlib.colors as mcolors

COLORS = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
CAHCE_FILE_NAME = 'cache_fields.gz'


class MotifFinder:
    CACHED_FIELDS = ['motives', 'windows', 'decimate_q', 'lowpass', 'filter_order', 'search_fields',
                     'rd.root_dir', 'rd.channel', 'rd.__class__']

    def __init__(self, rd: Reader = None, decimate_q=15, lowpass=150, filter_order=3, cache_dir=None):
        if cache_dir is not None:
            self.load_cache(cache_dir)
        else:
            assert rd is not None, 'must pass Reader'
            self.rd = rd
            self.decimate_q = decimate_q
            self.lowpass = lowpass
            self.filter_order = filter_order
            self.motives = []
            self.windows = []
            self.search_fields = {k: None for k in inspect.signature(self.search).parameters.keys()}

        self.fs = self.rd.fs / (self.decimate_q or 1)
        self.all_gpu_devices = [device.id for device in cuda.list_devices()]  # Get a list of all available GPU devices
        self.masses = {}
        self.cache_dir = None

    def search(self, t_start=None, t_stop=None, v=None, durations=(2, 0.4), durations2remove=None, max_xcorr=0.8,
               is_cache=False, is_avg_motives=False, is_remove_corr=True, external_motives=(), bf=None):
        """
        Search for motives in a given signal
        @param t_start: start time in seconds
        @param t_stop: end time in seconds
        @param v: signal for motif search. If provided t_start and t_stop are ignored.
        @param durations: list of durations (in seconds) for motif search
        @param durations2remove: list of durations (in seconds) whose match examples should be removed from signal
        @param max_xcorr: max value of cross-correlation peak. If this value is reached, then the least frequent motif
                            of the two would be removed.
        @param is_cache: save found motives to cache directory
        @param is_avg_motives: Use averaging of all found motif instances for the saved motif
        @param is_remove_corr: In case of high corr (>max_xcorr) between 2 motives, remove the least frequent one.
        @param external_motives: list of external motives to be removed from signal before searching.
        @return: path of cache directory
        """
        if v is None:
            v, _ = self.read_signal(t_start, t_stop)
        # remove from the signal all the external motif matches
        for m in external_motives:
            motif_indices, _ = self.mass_motif_search(m, v, max_dist=30)
            v = self.remove_motives_from_signal(v, len(m), motif_indices)
            self.motives.append(m), self.windows.append(len(m))
        # motif search
        for duration in durations:
            motif_indices, window = self.find_duration_motif(v, duration)
            motives = [v[motif_indices[i, 0]:motif_indices[i, 0]+window] for i in range(motif_indices.shape[0])]
            good_motives_idx = np.arange(len(motives))
            if is_remove_corr:
                good_motives_idx = self.merge_with_cross_correlation(motives, motif_indices, max_xcorr)
            for motif_id in good_motives_idx:
                if is_avg_motives:  # append the avg signal of all found motives
                    m = np.vstack([v[j:j+window] for j in motif_indices[motif_id, :] if j > 0]).mean(axis=0)
                else:  # append the closest motif to self.motives
                    m = v[motif_indices[motif_id, 0]:motif_indices[motif_id, 0]+window]
                self.motives.append(m)
                self.windows.append(window)
            if durations2remove and duration in durations2remove:
                v = self.remove_motives_from_signal(v, window, motif_indices)
        self.search_fields.update({k: locals().get(k) for k in self.search_fields.keys()})
        if is_cache:
            self.cache_dir = self.get_cache_dir()
            self.save_cache()

        self.plot_search_summary(t_start, t_stop, is_save=is_cache, bf=bf)
        return self.cache_dir.as_posix() if is_cache else None

    def find_duration_motif(self, v, window_duration, max_motifs=20) -> Tuple[np.ndarray, int]:
        window = int(self.fs * window_duration)
        t0 = time.time()
        mps = stumpy.gpu_stump(v, m=window, device_id=self.all_gpu_devices)
        print(f'Finish calculating matrix profile with window of {window_duration} seconds in {(time.time() - t0) / 60:.1f} minutes.')
        motif_distances, motif_indices = stumpy.motifs(v, mps[:, 0], max_motifs=max_motifs, max_matches=1000, max_distance=30.0)
        return motif_indices, window

    def mass_motif_search(self, motif_id, v, max_dist=15, peaks_distance_duration=0.3, is_cache=True):
        """
        Find all motives in a given signal
        @param motif_id: The index of the motif or the motif itself as numpy array
        @param v: The source signal for searching motives
        @param max_dist: Maximum minkowsky distance
        @param peaks_distance_duration: The minimum duration of a peak (used to filter out adjacent motives)
        @param is_cache: use saved cache if exist, and save to cache the search results
        @return: (motif indices, distances)
        """
        t0 = time.time()
        motif = self.motives[motif_id] if isinstance(motif_id, int) else motif_id
        is_cache = isinstance(motif_id, int) and is_cache
        if is_cache and motif_id in self.masses:
            dists = self.masses[motif_id]
        else:
            dists = stumpy.mass(motif, v)
            if is_cache:
                self.masses[motif_id] = dists
            dists = self.filter_adjacent_motives(dists, distance_duration=peaks_distance_duration)

        idx = np.where(dists <= max_dist)[0]
        if not (is_cache and motif_id in self.masses):
            motif_id_label = motif_id if isinstance(motif_id, int) else 'external'
            # print(f'Motif ID: {motif_id_label}; Finish mass motif search in {(time.time() - t0) / 60:.1f} minutes.'
            #       f' # of motives found: {len(idx)}')
        return idx, dists

    def filter_adjacent_motives(self, dists, distance_duration=0.3):
        peaks, _ = find_peaks(-dists, distance=self.fs * distance_duration)
        mask = np.ones(dists.shape[0], dtype=bool)
        mask[peaks] = False
        dists[mask] = np.inf
        return dists

    def eliminate_overlap_motives(self, window: int, max_overlap=0.8, motif_indices=None,
                                  t_start=None, t_stop=None, max_dist=15):
        """
        Calculate the overlap ratio between motives in an arbitrary signal (defined by t_start and t_stop) or by using
        the motif search output motif_indices. Then, in cases of overlap greater than max_overlap, eliminate the least
        frequent motif.
        @param window: The length of the motives window. Notice this analysis works only with a single window.
        @param max_overlap: The maximum overlap ratio between motives.
        @param motif_indices: Matrix of motives indices shape: (motives, observations)
        @param t_start: The start time (seconds) of an arbitrary signal. None is recording start.
        @param t_stop: The stop time (seconds) of an arbitrary signal. None is recording end.
        @param max_dist:
        @return: (remaining_motives, overlap_matrix)
        """
        if motif_indices is None:  # arbitrary signal
            motives = [i for i, w in enumerate(self.windows) if w == window]
            v, t = self.read_signal(t_start, t_stop)
            l = []
            for motif_id in motives:
                idx, _ = self.mass_motif_search(motif_id, v, max_dist=max_dist)
                l.append(idx)
            max_length = max([len(m_idx) for m_idx in l])
            motif_indices = np.full((len(motives), max_length), -1)
            for i, m_idx in enumerate(l):
                motif_indices[i, :len(m_idx)] = m_idx
        else:
            motives = list(range(motif_indices.shape[0]))
        motives_count = []
        M = self.calc_overlap_matrix(motif_indices, window)

        # Eliminate overlapping motives
        for i in motives:
            for j in np.arange(i+1, M.shape[0], 1):
                if M[i, j] >= max_overlap and M[j, i] >= max_overlap:
                    motif2remove = j if motives_count[i] > motives_count[j] else i
                    if motif2remove in motives:
                        motives.remove(motif2remove)
        return motives, M

    @staticmethod
    def merge_with_cross_correlation(motives, motif_indices, max_xcorr=0.8):
        remained_motives_ids = list(range(len(motives)))
        for i in range(len(motives)):
            for j in range(i+1, len(motives)):
                if not all(k in remained_motives_ids for k in [i, j]):
                    continue
                c, _ = utils.cross_correlate(motives[i], motives[j])
                if max(c) > max_xcorr:
                    mi1, mi2 = motif_indices[i, :], motif_indices[j, :]
                    if len(mi1[mi1 > -1]) >= len(mi2[mi2 > -1]):
                        remained_motives_ids.remove(j)
                    else:
                        remained_motives_ids.remove(i)
        return remained_motives_ids

    @staticmethod
    def calc_overlap_matrix(motif_indices: np.ndarray, window: int):
        """
        Calculate the overlap of motives (ratio of shared bins)
        :param motif_indices: Matrix with rows as # motives and columns as # observations. Empty cells are stored
                              with -1.
        :param window: indicates the window bins length
        """
        n_motives = motif_indices.shape[0]
        M = np.zeros((n_motives, n_motives))
        for motif_id1 in range(n_motives):
            idx1 = [i for i in motif_indices[motif_id1, :] if i != -1]
            full_idx1 = [j for i in idx1 for j in np.arange(i, i + window)]
            for motif_id2 in range(n_motives):
                if motif_id1 == motif_id2:
                    M[motif_id1, motif_id2] = 1
                    continue
                idx2 = [i for i in motif_indices[motif_id2, :] if i != -1]
                full_idx2 = [j for i in idx2 for j in np.arange(i, i + window)]
                M[motif_id1, motif_id2] = len(set(full_idx1) & set(full_idx2)) / len(full_idx1)
        return M

    def plot_search_summary(self, t_start, t_stop, max_dist=35, cols=5, is_save=False, bf=None):
        if bf is None:
            probs, counts = self.sleep_cycles_stats(max_dist=max_dist, t_start=t_start, t_stop=t_stop, is_plot=False)
        else:
            probs, counts = self.strikes_stats(bf, max_dist=max_dist, t_start=t_start, t_stop=t_stop, is_plot=False)

        signal_types = ['sws', 'rem'] if bf is None else ['before', 'after']
        probs2 = [{k: p[k] for k in signal_types} for p in probs]
        for stype in signal_types:
            motives_ids = [i for i, p in enumerate(probs2) if max(p, key=p.get)==stype]
            n_motives = len(motives_ids)
            if n_motives == 0:
                print(f'No {stype} motives were found')
                continue
            motives = [(i, m, counts[i]) for i, m in enumerate(self.motives) if i in motives_ids]
            motives = sorted(motives, key=lambda x: x[2], reverse=True)
            rows = int(np.ceil(n_motives/cols))
            fig, axes = plt.subplots(rows, cols, figsize=(25, 4*rows))
            axes = axes.flatten()
            for ax_id, (i, m, c) in enumerate(motives):
                t = np.linspace(0, len(m)/self.fs, len(m))
                axes[ax_id].plot(t, m)
                axes[ax_id].set_title(f'Motif ID:{i} (window={self.windows[i]/self.fs:.1f}sec)', fontsize=16, weight='bold')
                ymin, _ = axes[ax_id].get_ylim()
                if bf is None:
                    axes[ax_id].text(0, ymin+1, f'P(m|REM)={probs[i]["rem"]:.2f}\nP(m|SWS)={probs[i]["sws"]:.2f}\nCount={c}',
                                 color='red', fontsize=16, weight='bold')
                else:
                    axes[ax_id].text(0, ymin + 1,
                                     f'P(m|AFTER)={probs[i]["after"]:.3f}\nP(m|BEFORE)={probs[i]["before"]:.3f}\nCount={c}',
                                     color='red', fontsize=16, weight='bold')
            fig.suptitle(f'{stype.upper()} Motives', fontsize=20)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            if is_save:
                fig.savefig(f'{self.cache_dir}/found_motives_{stype}.png')

    def plot_motif_rate_vs_time(self, motives_ids, max_dist, rate_window=60, overlap=0.5, t_start=None, t_stop=None,
                                is_normalize=True, colors=None):
        assert isinstance(motives_ids, (list, tuple)), 'motives_ids must be iterable'
        v, t = self.read_signal(t_start, t_stop)
        t_rate, rate, motives_counts = dict(), dict(), dict()
        for i, motif_id in enumerate(motives_ids):
            max_dist_ = max_dist[i] if isinstance(max_dist, list) else max_dist
            _, dists = self.mass_motif_search(motif_id, v, is_cache=(not t_start and not t_stop), max_dist=max_dist_)
            idx = []
            for idx_group in consecutive(np.where(dists <= max_dist_)[0]):
                if len(idx_group) == 1:
                    idx.append(idx_group[0])
                else:
                    idx.append(np.argmin(dists[idx_group]) + idx_group[0])
            motives_counts[motif_id] = len(idx)
            toverlap = rate_window * overlap
            motif_times = t[idx]
            rate[motif_id] = []
            t_rate[motif_id] = []
            for t_start in np.arange(t[0], t[-1] - rate_window, rate_window - toverlap):
                rate[motif_id].append(len(motif_times[(motif_times >= t_start) & (motif_times < t_start + rate_window)]))
                t_rate[motif_id].append(t_start / 3600)

        plt.figure(figsize=(25, 6))
        colors = colors or COLORS
        for motif_id in motives_ids:
            r = np.array(rate[motif_id])
            if is_normalize:
                r = (r - r.mean()) / r.std()
            plt.plot(t_rate[motif_id], r, label=f'Motif ID:{motif_id} (#motives={motives_counts[motif_id]}',
                     color=colors[motif_id])
        plt.ylabel('Rate [1/sec]')
        plt.xlabel('Time [hours]')
        plt.legend()
        plt.title(f'Motives Rate using sliding window of {rate_window} seconds and {overlap*100:g}% overlap')

    def plot_examples(self, motif_id, n_examples=30, cols=6, seg_duration=None, max_dist=40, is_best=False,
                      peaks_distance_duration=0.3):
        v, _ = self.read_signal(3600*4, 3600*5)
        _, dists = self.mass_motif_search(motif_id, v, is_cache=False, peaks_distance_duration=peaks_distance_duration)
        if is_best:
            idx = dists.argsort()
            examples = idx[:n_examples]
        else:
            idx = np.where(dists <= max_dist)[0]
            examples = np.random.choice(idx, n_examples)
        rows = int(np.ceil(n_examples / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(25, 3*rows))
        axes = axes.flatten()
        for i, m_id in enumerate(examples):
            if seg_duration is not None:
                motif_mid = (m_id + m_id + self.windows[motif_id]) // 2
                half_window = int((seg_duration/2)*self.fs)
                axes[i].plot(v[motif_mid-half_window:motif_mid+half_window])
                axes[i].axvline(m_id - (motif_mid - half_window), linestyle='--', color='k')
                axes[i].axvline(m_id + self.windows[motif_id] - (motif_mid - half_window), linestyle='--', color='k')
            else:
                axes[i].plot(v[m_id:m_id + self.windows[motif_id]])
            axes[i].set_title(f'dist={dists[m_id]:.1f}')
        fig.tight_layout()

    def plot_labelled_cycle(self, cycle_id, motives_ids=None, max_dist=20, is_save=False,
                             colors=None, is_separate=True):
        colors = colors or COLORS
        sc = self.rd.load_slow_cycles()
        v, t = self.read_signal(sc.loc[cycle_id, 'on'], sc.loc[cycle_id, 'off'])
        if motives_ids is None:
            motives_ids = list(range(len(self.motives)))
        rows = len(motives_ids) if is_separate else 1
        fig = plt.figure(figsize=(30, 5*rows))
        outer = fig.add_gridspec(1, 2, wspace=0.1, width_ratios=[1, 9])
        examples_grid = outer[0].subgridspec(len(motives_ids), 1, wspace=0, hspace=0)
        signal_grid = outer[1].subgridspec(rows, 1, wspace=0, hspace=0)
        signal_axes = signal_grid.subplots()
        examples_axes = examples_grid.subplots()
        if not is_separate or len(motives_ids) == 1:
            signal_axes = [signal_axes]
        if len(motives_ids) == 1:
            examples_axes = [examples_axes]

        for j, ax in enumerate(signal_axes):
            if not is_separate:
                t = t - t[0]
            ax.plot(t, v, color='black')
            ax.set_xlabel('Time [sec]')
            ax.set_ylabel('Voltage [mV]')

        for j, motif_id in enumerate(motives_ids):
            ax = examples_axes[j]
            motif_ = self.motives[motif_id] if isinstance(motif_id, int) else motif_id
            ax.plot(motif_, color=colors[j])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            if j == 0:
                ax.set_title('Motives')
            sig_ax = signal_axes[j if is_separate else 0]
            max_dist_ = max_dist[j] if isinstance(max_dist, list) else max_dist
            idx, dists = self.mass_motif_search(motif_id, v, max_dist=max_dist_, is_cache=False)
            for i, m_id in enumerate(idx):
                t_, v_ = t[m_id:m_id+len(motif_)], v[m_id:m_id+len(motif_)]
                motif_label = motif_id if isinstance(motif_id, int) else "external"
                sig_ax.plot(t_, v_, color=colors[j],
                            label=f'Motif ID: {motif_label}' if i == 0 else None)
                if is_separate:
                    sig_ax.text(t_[0], max(v_)+2, f'{dists[m_id]:.0f}\n{np.std(v_):.1f}')

        for ax in signal_axes:
            ax.legend()
            ax.axvline(sc.loc[cycle_id, 'mid'], color='red')
        if is_save:
            fig.savefig(f'labelled_cycle_{cycle_id}.png')
            plt.close(fig)

    def plot_labelled_strikes(self, bf, motives_ids=None, max_dist=20, is_save=False,
                             colors=None, is_separate=True):
        colors = colors or COLORS
        v, t = self.read_signal()
        if motives_ids is None:
            motives_ids = list(range(len(self.motives)))
        rows = len(motives_ids) if is_separate else 1
        fig = plt.figure(figsize=(30, 5*rows))
        outer = fig.add_gridspec(1, 2, wspace=0.1, width_ratios=[1, 9])
        examples_grid = outer[0].subgridspec(len(motives_ids), 1, wspace=0, hspace=0)
        signal_grid = outer[1].subgridspec(rows, 1, wspace=0, hspace=0)
        signal_axes = signal_grid.subplots()
        examples_axes = examples_grid.subplots()
        if not is_separate or len(motives_ids) == 1:
            signal_axes = [signal_axes]
        if len(motives_ids) == 1:
            examples_axes = [examples_axes]

        for j, ax in enumerate(signal_axes):
            if not is_separate:
                t = t - t[0]
            ax.plot(t, v, color='black')
            ax.set_xlabel('Time [sec]')
            ax.set_ylabel('Voltage [mV]')

        for j, motif_id in enumerate(motives_ids):
            ax = examples_axes[j]
            motif_ = self.motives[motif_id] if isinstance(motif_id, int) else motif_id
            ax.plot(motif_, color=colors[j])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            if j == 0:
                ax.set_title('Motives')
            sig_ax = signal_axes[j if is_separate else 0]
            max_dist_ = max_dist[j] if isinstance(max_dist, list) else max_dist
            idx, dists = self.mass_motif_search(motif_id, v, max_dist=max_dist_, is_cache=False)
            for i, m_id in enumerate(idx):
                t_, v_ = t[m_id:m_id+len(motif_)], v[m_id:m_id+len(motif_)]
                motif_label = motif_id if isinstance(motif_id, int) else "external"
                sig_ax.plot(t_, v_, color=colors[j],
                            label=f'Motif ID: {motif_label}' if i == 0 else None)
                if is_separate:
                    sig_ax.text(t_[0], max(v_)+2, f'{dists[m_id]:.0f}\n{np.std(v_):.1f}')

        for ax in signal_axes:
            for i, row in bf.iterrows():
                ax.legend()
                ax.axvline(row.oe_time, color='red')
        if is_save:
            fig.savefig(f'labelled_strikes.png')
            plt.close(fig)

    def scan_for_stats(self, motives_ids, t_start, t_stop, max_dist):
        if motives_ids is None:
            motives_ids = list(range(len(self.motives)))
        v, t = self.read_signal(t_start, t_stop)
        time_vectors = []
        for i, motif_id in enumerate(motives_ids):
            max_dist_ = max_dist[i] if isinstance(max_dist, list) else max_dist
            idx, _ = self.mass_motif_search(motif_id, v, is_cache=(not t_start and not t_stop), max_dist=max_dist_)
            time_vectors.append(t[idx])
        return time_vectors, t

    def strikes_stats(self, bf, window=4, motives_ids=None, max_dist=30, t_start=None, t_stop=None, is_plot=True):
        time_vectors, t = self.scan_for_stats(motives_ids, t_start, t_stop, max_dist)
        durations = {'before': 0, 'after': 0}
        motives_durations = [{'before': 0, 'after': 0} for _ in time_vectors]
        for _, row in bf.iterrows():
            if (t_stop and row.oe_time + window >= t_stop) or (t_start and row.oe_time - window <= t_start):
                continue
            durations['before'] += (row.oe_time - window)
            durations['after'] += (row.oe_time + window)
            for i, t_ in enumerate(time_vectors):
                d = self.windows[i]/self.fs
                motives_durations[i]['before'] += sum([(d if ts+d<row.oe_time else row.oe_time - ts)
                                                      for ts in t_[(row.oe_time-window <= t_) & (t_ < row.oe_time)]])
                motives_durations[i]['after'] += sum([(d if ts+d<row.oe_time+window else row.oe_time+window - ts)
                                                      for ts in t_[(row.oe_time <= t_) & (t_ < row.oe_time+window)]])
        total_duration = t[-1] - t[0]
        out_duration = total_duration - (durations['before'] + durations['after'])
        if is_plot:
            fig, axes = plt.subplots(len(motives_ids), 2, figsize=(10, len(motives_ids)*3))
        motives_probs, motives_counts = [], []
        for i, md in enumerate(motives_durations):
            p_sws, p_rem = md['before']/durations['before'], md['after']/durations['after']
            p_out = (len(time_vectors[i]) - (md['before'] + md['after'])) / out_duration if out_duration > 1 else 0
            motives_probs.append({'before': p_sws, 'after': p_rem, 'control': p_out})
            motives_counts.append(len(time_vectors[i]))
            if is_plot:
                axes[i, 0].plot(self.motives[motives_ids[i]])
                axes[i, 1].bar(['Before Strike', 'After Strike', 'Control'], [p_sws, p_rem, p_out])
        return motives_probs, motives_counts

    def sleep_cycles_stats(self, motives_ids=None, max_dist=30, t_start=None, t_stop=None, is_plot=True):
        time_vectors, t = self.scan_for_stats(motives_ids, t_start, t_stop, max_dist)
        sc = self.rd.load_slow_cycles()
        durations = {'rem': 0, 'sws': 0}
        motives_durations = [{'rem': 0, 'sws': 0} for _ in time_vectors]
        for _, row in sc.iterrows():
            if (t_stop and row.on >= t_stop) or (t_start and row.off <= t_start):
                continue
            durations['sws'] += (row.mid - row.on)
            durations['rem'] += (row.off - row.mid)
            for i, t_ in enumerate(time_vectors):
                d = self.windows[i]/self.fs
                motives_durations[i]['sws'] += sum([(d if ts+d<row.mid else row.mid - ts)
                                                      for ts in t_[(row.on <= t_) & (t_ < row.mid)]])
                motives_durations[i]['rem'] += sum([(d if ts+d<row.off else row.off - ts)
                                                      for ts in t_[(row.mid <= t_) & (t_ < row.off)]])
        total_duration = t[-1] - t[0]
        out_duration = total_duration - (durations['rem'] + durations['sws'])
        if is_plot:
            fig, axes = plt.subplots(len(motives_ids), 2, figsize=(10, len(motives_ids)*3))
        motives_probs, motives_counts = [], []
        for i, md in enumerate(motives_durations):
            p_sws, p_rem = md['sws']/durations['sws'], md['rem']/durations['rem']
            p_out = (len(time_vectors[i]) - (md['sws'] + md['rem'])) / out_duration if out_duration > 1 else 0
            motives_probs.append({'sws': p_sws, 'rem': p_rem, 'off-cycle': p_out})
            motives_counts.append(len(time_vectors[i]))
            if is_plot:
                axes[i, 0].plot(self.motives[motives_ids[i]])
                axes[i, 1].bar(['SWS', 'REM', 'off-cycle'], [p_sws, p_rem, p_out])
        return motives_probs, motives_counts

    def plot_correlations(self, motives=None):
        motives = motives or self.motives
        rows = np.arange(len(motives)).sum()
        fig, axes = plt.subplots(rows, 3, figsize=(20, 3*rows))
        row = 0
        for i in range(len(motives)):
            for j in range(i + 1, len(motives)):
                corr, lags = utils.cross_correlate(motives[i], motives[j])
                axes[row, 0].plot(motives[i]), axes[row, 0].set_title(str(i))
                axes[row, 1].plot(motives[j]), axes[row, 1].set_title(str(j))
                axes[row, 2].plot(lags, corr), axes[row, 2].set_title(f'max={max(corr):.1f}')
                row += 1
        fig.tight_layout()

    def read_signal(self, t_start=None, t_stop=None):
        i_start = int(t_start * self.rd.fs) if t_start else None
        i_stop = int(t_stop * self.rd.fs) if t_stop else None
        v, t = self.rd.read(i_start=i_start, i_stop=i_stop, lowpass=self.lowpass, filter_order=self.filter_order)
        if self.decimate_q:
            # v, t = decimate(v, self.decimate_q), decimate(t, self.decimate_q)
            v, t = resample(v, len(v)//self.decimate_q, t)
        return v, t

    @staticmethod
    def remove_motives_from_signal(v, window, motif_indices):
        """remove the motives from the signal and insert a single np.nan in their places"""
        v_new = []
        i_start = 0
        for i in np.sort(motif_indices.flatten()):
            if i > i_start:
                v_new.append(np.append(v[i_start:i], np.nan))
            i_start = i + window
        return np.hstack(v_new)

    def save_cache(self):
        cache = {}
        for attr in self.CACHED_FIELDS:
            attr_ = attr.split('.')
            value = getattr(self, attr_[0])
            if len(attr_) > 1:
                value = getattr(value, attr_[1])
            cache[attr] = value
        with gzip.open(self.cache_dir / CAHCE_FILE_NAME, 'wb') as f:
            pickle.dump(cache, f)

    def load_cache(self, cache_dir):
        with gzip.open(Path(cache_dir) / CAHCE_FILE_NAME, 'rb') as f:
            cache = pickle.load(f)
        rd_kwargs = {}
        rd_cls = NeuralynxReader
        for attr, value in cache.items():
            if attr.startswith('rd.'):
                attr = attr.split('.')[1]
                if attr == '__class__':
                    rd_cls = value
                    continue
                rd_kwargs[attr] = value
            else:
                setattr(self, attr, value)

        self.rd = rd_cls(**rd_kwargs)

    @staticmethod
    def get_cache_dir():
        now = datetime.now()
        dir_path = utils.ROOT_DIR / 'output' / 'motives' / now.strftime('%Y%m%d') / now.strftime('%H%M%S')
        dir_path.mkdir(exist_ok=True, parents=True)
        return dir_path
