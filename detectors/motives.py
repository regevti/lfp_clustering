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
from scipy.signal import decimate, find_peaks
import matplotlib.pyplot as plt

import utils
from readers import Reader, NeuralynxReader
from utils import consecutive
import matplotlib.colors as mcolors

COLORS = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
CAHCE_FILE_NAME = 'cache_fields.gz'


class MotifFinder:
    CACHED_FIELDS = ['motives', 'windows', 'decimate_q', 'lowpass', 'filter_order', 'rd.root_dir', 'rd.channel',
                     'search_fields']

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

    def search(self, t_start, t_stop, durations=(2, 0.4), durations2remove=None, max_overlap=0.8, max_xcorr=0.8,
               is_cache=False):
        v, t = self.read_signal(t_start, t_stop)
        for duration in durations:
            motif_indices, window = self.find_duration_motif(v, duration)
            motives = [v[motif_indices[i, 0]:motif_indices[i, 0]+window] for i in range(motif_indices.shape[0])]
            good_motives_idx = self.eliminate_cross_correlation(motives, motif_indices, max_xcorr)
            # good_motives_idx, _ = self.eliminate_overlap_motives(window, max_overlap, motif_indices)
            for i in good_motives_idx:
                # append the closest motif to self.motives
                self.motives.append(v[motif_indices[i, 0]:motif_indices[i, 0]+window])
                self.windows.append(window)
            if durations2remove and duration in durations2remove:
                v, t = self.remove_motives_from_signal(v, t, window, motif_indices)
        self.search_fields.update({k: locals().get(k) for k in self.search_fields.keys()})
        if is_cache:
            self.cache_dir = self.get_cache_dir()
            self.save_cache()
            self.plot_found_motives(is_save=True)

        return self.cache_dir.as_posix() if is_cache else None

    def find_duration_motif(self, v, window_duration, max_motifs=8) -> Tuple[np.ndarray, int]:
        window = int(self.fs * window_duration)
        t0 = time.time()
        mps = stumpy.gpu_stump(v, m=window, device_id=self.all_gpu_devices)
        print(f'Finish calculating matrix profile with window of {window_duration} seconds in {(time.time() - t0) / 60:.1f} minutes.')
        motif_distances, motif_indices = stumpy.motifs(v, mps[:, 0], max_motifs=max_motifs, max_matches=1000)
        return motif_indices, window

    def mass_motif_search(self, motif_id, v, max_dist=15, peaks_distance_duration=0.3, is_cache=True):
        t0 = time.time()
        if is_cache and motif_id in self.masses:
            dists = self.masses[motif_id]
        else:
            dists = stumpy.mass(self.motives[motif_id], v)
            if is_cache:
                self.masses[motif_id] = dists
            dists = self.filter_adjacent_motives(dists, distance_duration=peaks_distance_duration)
        idx = np.where(dists <= max_dist)[0]
        if not (is_cache and motif_id in self.masses):
            print(f'Motif ID: {motif_id}; Finish mass motif search in {(time.time() - t0) / 60:.1f} minutes.'
                  f' # of motives found: {len(idx)}')
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
    def eliminate_cross_correlation(motives, motif_indices, max_xcorr=0.8):
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

    def plot_found_motives(self, cols=5, is_save=False):
        n_motives = len(self.motives)
        rows = int(np.ceil(n_motives/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(25, 4*rows))
        axes = axes.flatten()
        for i, m in enumerate(self.motives):
            t = np.arange(0, len(m)/self.fs, 1/self.fs)
            axes[i].plot(t, m)
            axes[i].set_title(f'Motif ID:{i} (window={self.windows[i]/self.fs:.1f}sec)')
        fig.tight_layout()
        if is_save:
            fig.savefig(f'{self.cache_dir}/found_motives.png')
            plt.close(fig)

    def plot_motif_rate_vs_time(self, motives_ids, max_dist, rate_window=60, overlap=0.5, t_start=None, t_stop=None,
                                is_normalize=True):
        assert isinstance(motives_ids, (list, tuple)), 'motives_ids must be iterable'
        v, t = self.read_signal(t_start, t_stop)
        t_rate, rate, motives_counts = dict(), dict(), dict()
        for motif_id in motives_ids:
            _, dists = self.mass_motif_search(motif_id, v, is_cache=(not t_start and not t_stop))
            idx = []
            for idx_group in consecutive(np.where(dists < max_dist)[0]):
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
        for motif_id in motives_ids:
            r = np.array(rate[motif_id])
            if is_normalize:
                r = (r - r.mean()) / r.std()
            plt.plot(t_rate[motif_id], r, label=f'Motif ID:{motif_id} (#motives={motives_counts[motif_id]}')
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

    def plot_labelled_signal(self, t_start, t_stop, motives_ids=None, max_dist=20, is_save=False):
        v, t = self.read_signal(t_start, t_stop)
        if motives_ids is None:
            motives_ids = list(range(len(self.motives)))
        rows = len(motives_ids)
        fig, axes = plt.subplots(rows, 2, figsize=(25, 5*rows), gridspec_kw={"width_ratios": [1, 8]})
        for ax in axes[:, 1]:
            ax.plot(t, v, color='black')
        for j, motif_id in enumerate(motives_ids):
            axes[j, 0].plot(self.motives[motif_id])
            max_dist_ = max_dist[j] if isinstance(max_dist, list) else max_dist
            idx, dists = self.mass_motif_search(motif_id, v, max_dist=max_dist_, is_cache=False)
            for i, m_id in enumerate(idx):
                t_, v_ = t[m_id:m_id+self.windows[motif_id]], v[m_id:m_id+self.windows[motif_id]]
                axes[j, 1].plot(t_, v_, color=COLORS[motif_id],
                        label=f'Motif ID: {motif_id}' if i == 0 else None)
                axes[j, 1].text(t_[0], max(v_)+2, f'{dists[m_id]:.0f}')
        for ax in axes[:, 1]:
            ax.legend()
        plt.tight_layout()
        if is_save:
            fig.savefig('labelled_signal.png')
            plt.close(fig)

    def read_signal(self, t_start=None, t_stop=None):
        i_start = int(t_start * self.rd.fs) if t_start else None
        i_stop = int(t_stop * self.rd.fs) if t_stop else None
        v, t = self.rd.read(i_start=i_start, i_stop=i_stop, lowpass=self.lowpass, filter_order=self.filter_order)
        if self.decimate_q:
            v, t = decimate(v, self.decimate_q), decimate(t, self.decimate_q)
        return v, t

    @staticmethod
    def remove_motives_from_signal(v, t, window, motif_indices):
        idx = np.full(v.shape, True)
        for i in motif_indices.flatten():
            idx[i:i + window] = False
        return v[idx], t[idx]

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
        for attr, value in cache.items():
            if attr.startswith('rd.'):
                attr = attr.split('.')[1]
                rd_kwargs[attr] = value
            else:
                setattr(self, attr, value)
        self.rd = NeuralynxReader(**rd_kwargs)

    @staticmethod
    def get_cache_dir():
        now = datetime.now()
        dir_path = utils.ROOT_DIR / 'output' / 'motives' / now.strftime('%Y%m%d') / now.strftime('%H%M%S')
        dir_path.mkdir(exist_ok=True, parents=True)
        return dir_path
