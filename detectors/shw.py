import pathlib
import multiprocessing as mp
import traceback
import time
from statistics import mode
from multiprocessing.managers import SyncManager
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mne.time_frequency import morlet
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, filtfilt, resample, lfilter, convolve, correlate
from scipy.io import loadmat, savemat

import utils
from readers.mat_files import MatRecordingsParser
from utils import half_max_x, apply_parallel_dask, run_parallel, cpu_count


class SharpWavesFinder:
    def __init__(self, reader: (str, MatRecordingsParser), is_debug=True, shw_duration=1.2, use_cache=True,
                 max_width=0.3, use_matlab_algo=False):
        self.reader = reader
        self.use_cache = use_cache
        self.use_matlab_algo = use_matlab_algo
        self.max_width = max_width  # max width allowed for found ShW
        self.fs = reader.fs
        self.shw_duration_sec = shw_duration
        self.shw_w = int(self.fs * shw_duration)
        self.norm_conv = None
        self.thresh = None
        self.is_debug = is_debug
        self.sc = reader.load_slow_cycles()
        self.shw_df = pd.DataFrame()
        self.shw_records = {}
        self.processes = []

    def train(self, thresh=0.15, mfilt=None, i_start=None, i_stop=None, n_jobs=None, only_sleep=False):
        self.thresh = thresh
        if self.use_matlab_algo:
            self.load_matlab_algo()
            return
        if only_sleep:
            sc = self.reader.load_slow_cycles()
            t_start, t_stop = sc.iloc[0]['on'], sc.iloc[-1]['off']
            i_start = np.argmin(np.abs(self.reader.time_vector - t_start))
            i_stop = np.argmin(np.abs(self.reader.time_vector - t_stop))
        is_loaded = self.load_cache(i_start, i_stop, thresh)
        if not is_loaded:
            self._train(n_jobs, i_start, i_stop, mfilt, thresh)
        # if only_sleep:
        #     self.filter_sleep_cycles()

    def _train(self, n_jobs, i_start, i_stop, mfilt, thresh):
        t0 = time.time()
        with mp.Manager() as manager:
            n_jobs = n_jobs or (cpu_count() // 2)
            i_start, i_stop = i_start or 0, i_stop or len(self.reader.time_vector)
            self.print(f'start multiprocessing execution with {n_jobs} processes on record with fs={self.reader.fs:.0f}'
                       f'Hz, and duration of {(i_stop-i_start)/self.reader.recording_fs/3600:.1f} hours')

            iterable = np.array_split(np.arange(i_start, i_stop).astype(int), n_jobs)
            shw_recs, shw_dfs = manager.list([]), manager.list([])
            lock = manager.Lock()
            for proc_id, ix in enumerate(iterable):
                v, t = self.reader.read(ix[0], ix[-1])
                if proc_id == 0:
                    self.print(f'calc_fs={1/np.diff(t).mean():.1f}')
                p = SharpWavesProcess(proc_id, v, t, self.reader.fs, ix[0], shw_recs, shw_dfs, lock,
                                      mfilt, self.shw_duration_sec, thresh, self.max_width)
                p.start()
                self.processes.append(p)
            del v, t
            for p in self.processes:
                p.join(timeout=30 * 60)
            if shw_dfs:
                self.shw_df = pd.concat(shw_dfs).sort_values(by='start')
            self.shw_records = {str(rec[0]): rec[1:] for rec in shw_recs}
            self.save_cache(i_start, i_stop, thresh)
        self.print(f'finished shw training found: {len(self.shw_df)} ShWs; time taken: {(time.time()-t0)/60:.1f} minutes')

    def print(self, s):
        if self.is_debug:
            print(f'{self.reader}: {s}')

    def is_cache_exists(self, i_start, i_stop, thresh):
        i_start, i_stop = i_start or 0, i_stop or len(self.reader.time_vector)
        return self.get_cache_mat_path(i_start, i_stop, thresh).exists() and self.get_cache_df_path(i_start, i_stop, thresh)

    def get_cache_folder_path(self, i_start, i_stop, thresh) -> pathlib.Path:
        return self.reader.cache_dir_path / f'shw_{i_start}_{i_stop}_channel{self.reader.channel}_thresh{thresh}_dur{self.shw_duration_sec}'

    def get_cache_df_path(self, i_start, i_stop, thresh, is_matlab_algo=False) -> pathlib.Path:
        suffix = '_matlab_algo' if is_matlab_algo else ''
        return self.get_cache_folder_path(i_start, i_stop, thresh) / f'shw_df{suffix}.csv'

    def get_cache_mat_path(self, i_start, i_stop, thresh, is_matlab_algo=False) -> pathlib.Path:
        suffix = '_matlab_algo' if is_matlab_algo else ''
        return self.get_cache_folder_path(i_start, i_stop, thresh) / f'shw{suffix}.mat'

    def load_matlab_algo(self, dur_shw=1.5):
        df_path = self.get_cache_df_path(None, None, None, True)
        mat_path = self.get_cache_mat_path(None, None, None, True)
        if df_path.exists() and mat_path.exists():
            self.print(f'loading matlab_argo cache from {df_path}')
            self.shw_df = pd.read_csv(df_path)
            self.shw_records = {k: v for k, v in loadmat(mat_path.as_posix()).items() if not k.startswith('_')}
            return

        file_path = self.reader.analysis_folder / f'sharpWaves_ch{self.reader.channel}.mat'
        if not file_path.exists():
            return

        self.print(f'loading ShW from: {file_path}')
        t_shw = loadmat(file_path)['tSW'].flatten() / 1000
        v, t = self.reader.read()
        self.shw_df, self.shw_records = [], {}
        for ts in (tqdm(t_shw) if self.is_debug else t_shw):
            i = np.argmin(np.abs(t - ts - (dur_shw / 2)))
            i_start, i_end = i - self.shw_w//2 , i + self.shw_w//2
            self.shw_df.append({'start': i_start, 'end': i_end,
                                't_start': ts - self.shw_duration_sec/2, 't_end': ts + self.shw_duration_sec/2})
            self.shw_records[str(i_start)] = (v[i_start:i_end], t[i_start:i_end])
        del v, t
        self.shw_df = pd.DataFrame(self.shw_df)

        df_path.parent.mkdir(exist_ok=True, parents=True)
        self.shw_df.to_csv(df_path)
        savemat(mat_path.as_posix(), self.shw_records)

        return True

    def load_cache(self, i_start, i_stop, thresh) -> bool:
        if not self.use_cache:
            return False
        i_start, i_stop = i_start or 0, i_stop or len(self.reader.time_vector)
        df_path = self.get_cache_df_path(i_start, i_stop, thresh)
        mat_path = self.get_cache_mat_path(i_start, i_stop, thresh)
        if df_path.exists() and mat_path.exists():
            self.shw_df = pd.read_csv(df_path)
            self.shw_records = {k: v for k, v in loadmat(mat_path.as_posix()).items() if not k.startswith('_')}
            return True
        return False

    def save_cache(self, i_start, i_stop, thresh):
        df_path = self.get_cache_df_path(i_start, i_stop, thresh)
        mat_path = self.get_cache_mat_path(i_start, i_stop, thresh)
        df_path.parent.mkdir(exist_ok=True, parents=True)
        self.shw_df.to_csv(df_path)
        savemat(mat_path.as_posix(), self.shw_records)
        self.print(f'saved cache to {df_path.parent}')

    def plot_shw_records(self, n=40, cols=6):
        n = min([n, len(self.shw_df)])
        if not n:
            return
        rows = int(np.ceil(n/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(30, rows * 3))
        axes = axes.flatten()
        keys = np.random.choice(list(self.shw_records.keys()), n, replace=False)
        for i, key in enumerate(keys):
            v, t = self.shw_records[key]
            axes[i].plot(t, v)
            p2v = v.max() - v.min()
            row = self.shw_df.query(f'start=={int(key)}')
            if not row.empty:
                row = row.iloc[0]
                if not self.use_matlab_algo:
                    axes[i].set_title(f'({key}) power={row["power"]:.2f},width={row["width"]:.2f},p2v={p2v:.2f}\n'
                                      f't_start={t[0]}')
            axes[i].ticklabel_format(useOffset=False)
        fig.suptitle(f'ShW Records {n}/{len(self.shw_records)}')
        fig.tight_layout()

    def get_avg_shw(self, is_plot=True, shw_indices=None):
        m, t, count = None, None, 0
        shw_length = mode([len(v) for v, _ in self.shw_records.values()])
        for k, (v, t_) in self.shw_records.items():
            if shw_indices is not None and int(k) not in shw_indices:
                continue
            if m is None:
                if len(v) == shw_length:
                    m, t = v, t_
                    count += 1
                continue
            elif len(v) > len(m):
                v = v[:len(m)]
            elif len(v) < len(m):
                continue
            m += v
            count += 1
        t, m = t - t[0], m / count
        if is_plot:
            plt.plot(t, m)
            plt.title(f'Average ShW from {count} ShWs')
        else:
            return t, m

    def plot_cycle_with_sharp_waves(self, v, t, cycle_id, split=1):
        v, t, start_id = self.reader.get_sleep_cycle(cycle_id, t=t, v=v)
        assert isinstance(split, int)
        fig, axes = plt.subplots(2*split, 1, figsize=(25, 8*split))
        T = np.array_split(t, split)
        V = np.array_split(v, split)
        IDX = np.array_split(np.arange(start_id, start_id+len(t)), split)
        n_shw = 0
        for i, (id_, t_, v_) in enumerate(zip(IDX, T, V)):
            axes[2*i].plot(t_, v_)
            sf = self.shw_df.query(f'signal=={cycle_id} and group==group and start in {id_.tolist()}')
            n_shw += len(sf)
            idx = sf.start - id_[0] + (self.shw_w / 2)
            idx = idx.to_numpy().astype(int)
            axes[2*i].plot(t_[idx], v_[idx], 'o', markersize=8)
            axes[2*i+1].plot(t_, self.norm_conv[id_])
        self.print(f'# of slow waves: {n_shw}')
        return V

    def plot_sharp_waves_detection(self, rp, mfilt, t_start, t_end, ax, ax2):
        v, t = rp.read(t_start=t_start, t_stop=t_end)
        v = utils.smooth_signal(v, poly_order=7)
        ax.plot(t, v, 'k')
        ax.axis('off')
        ylim = ax.get_ylim()
        self.print(f'Ylim: {ylim[1] - ylim[0]}')

        power = -filtfilt(mfilt, 1, v)
        conv = power / power.max()
        peaks_, _ = find_peaks(conv, height=self.thresh, distance=self.fs * 0.5, width=self.fs * 0.1)
        peaks_ = [p for p in peaks_ if p > 0 and v[p] < -20]
        mw = self.shw_w // 2
        for peak in peaks_:
            ax.axvspan(t[max([peak-mw, 0])], t[min([len(t)-1, peak+mw])], facecolor='g', alpha=0.4)

        ax2.plot(t, conv, 'k')
        ax2.scatter(t[peaks_], conv[peaks_], c='g', alpha=0.4)
        ax2.axis('off')
        ax2.axhline(0, c='k')
        ax2.axhline(self.thresh)

    def plot_sw_cycle_rate(self, wt=5, overlap=0.75, interp_step=1, group_length=50, rf=None, ax=None):
        r, t, rf = self.calc_cycle_sw_rate(wt, overlap, interp_step, group_length, rf)
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(14, 6))
        ax.plot(t, r)
        ax.axvline(t[group_length-1])
        ax.set_title(f'Average Cycle Rate of Slow Waves')
        ax.set_xlabel('Cycle Time [sec]')
        ax.set_ylabel('Rate')
        return r, t

    def plot_phase_shw_cycle(self, wt=20, overlap=0.75, ax=None, nbins=25):
        rf = self.calc_sw_rate(wt=wt, overlap=overlap)  # .query('group==group')
        l = []
        # t_vec = np.arange(-np.pi, np.pi+0.1, 0.1)
        t_vec = np.linspace(-np.pi, np.pi, nbins)
        for sig_id in rf.signal.unique():
            if np.isnan(sig_id):
                continue
            t_start = self.sc.loc[sig_id, 'on']
            t_end = self.sc.loc[sig_id, 'off']
            t_mid = self.sc.loc[sig_id, 'mid']
            duration = t_end - t_start
            sf = rf.query(f'time>={t_mid-duration/2} and time<={t_mid+duration/2}')
            if len(sf) < 2:
                continue
            f = interp1d(sf.time, sf.rate, fill_value="extrapolate", kind='nearest')
            t_new = np.linspace(t_mid-duration/2, t_mid+duration/2, len(t_vec))
            l.append(f(t_new))
        r = np.vstack(l).mean(axis=0)
        r = (r - r.min())/(r.max()-r.min())

        if ax is None:
            fig, ax = plt.subplots(1, 1, dpi=100)

        ax.plot(t_vec, r, linewidth=2, label='ShW')
        ax.set_xlabel('Phase')
        ax.set_ylabel('norm. value')
        ax.margins(0, 0)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

        return t_vec, r

    def calc_sw_rate(self, wt, overlap=0.75, label=None, lights_off_only=False) -> pd.DataFrame:
        """Calculate the rate of slow-waves across all recording"""
        noverlap = int(wt * overlap)
        rf = []
        sig_df = self.shw_df.query(f'label=={label}') if label is not None else self.shw_df.copy()
        sws_times = sig_df.t_start.values
        startT = 0
        endT = self.reader.time_vector[-1]

        if lights_off_only:
            startT = self.reader.t_lights_off - 2 * 60 * 60
            if startT < 0:
                startT = 0
            endT = self.reader.t_lights_on + 2 * 60 * 60

        for t_start in np.arange(startT, endT - wt, wt - noverlap):
            r = len(sws_times[(sws_times >= t_start) & (sws_times < t_start + wt)])
            sc_id = np.where((self.sc['on'] <= t_start) &
                             (self.sc['off'] >= t_start + wt))[0]
            sc_id = int(sc_id[0]) if len(sc_id) == 1 else np.nan
            if isinstance(sc_id, int):
                group = 1 if t_start < self.sc.loc[sc_id, 'mid'] else 2
                group_time = t_start - self.sc.loc[sc_id, 'on'] if group == 1 else t_start - self.sc.loc[
                    sc_id, 'mid']
            else:
                group = np.nan
                group_time = np.nan
            if self.reader.start_timestamp is not None:
                dt = self.reader.start_timestamp + pd.to_timedelta(t_start, unit='seconds')
            else:
                dt = None
            rf.append({'rate': r, 'time': t_start + (wt/2), 'group_time': group_time, 'group': group,
                       'signal': sc_id, 'datetime': dt})
        rf = pd.DataFrame(rf)
        return rf

    def calc_cycle_sw_rate(self, wt=5, overlap=0.75, interp_step=1, group_length=50, rf=None):
        """Calculate the average slow-wave rate per sleep cycle around the transition point"""
        if rf is None:
            rf = self.calc_sw_rate(wt=wt, overlap=overlap) #.query('group==group')
        l = []
        for sig_id in rf.signal.unique():
            if np.isnan(sig_id):
                continue
            t_mid = self.sc.loc[sig_id, 'mid']
            sf = rf.query(f'time>{t_mid-group_length} and time<{t_mid+group_length}')
            if len(sf) < 2:
                continue
            f = interp1d(sf.time, sf.rate, fill_value="extrapolate", kind='nearest')
            t_new = np.arange(t_mid - group_length, t_mid + group_length, interp_step)
            l.append(f(t_new))
        r = np.vstack(l).mean(axis=0)
        t = np.arange(0, group_length*2, interp_step)
        return r, t, rf

    def filter_sleep_cycles(self):
        sc = self.reader.load_slow_cycles()
        t_start, t_end = sc.iloc[0]['on'], sc.iloc[-1]['off']
        self.shw_df = self.shw_df.query(f't_start>={t_start} and t_end<={t_end}')
        self.shw_records = {k: v for k, v in self.shw_records.items() if int(k) in self.shw_df.start.values}


class SharpWavesProcess(mp.Process):
    def __init__(self, proc_id, v, t, fs, i_start, shw_recs, shw_dfs, lock,
                 mfilt=None, shw_duration_sec=1.2, shw_thresh=0.5, max_width=None, work_fs=300):
        super().__init__()
        self.proc_id = proc_id
        self.shw_duration_sec = shw_duration_sec
        self.fs = fs
        self.max_width = max_width
        self.shw_w = int(shw_duration_sec * self.fs)
        self.shw_thresh = shw_thresh
        self.global_i_start = i_start
        self.work_fs = work_fs
        self.mfilt = mfilt
        self.v, self.t = v, t
        self.shw_df = None
        self.shw_recs = []
        self.lock = lock
        self.shared_shw_df_list = shw_dfs
        self.shared_shw_rec_list = shw_recs

    def run(self):
        try:
            q = int(len(self.v) / (self.fs / self.work_fs))
            # v_resampled, t_resampled = resample(self.v, q, t=self.t)
            v_resampled, t_resampled, resampled_fs = self.v, self.t, self.fs
            # resampled_fs = 1 / np.diff(t_resampled[1:-1]).mean()
            norm_power = self.get_sharp_waves(v_resampled, t_resampled, fs=resampled_fs)
            if self.shw_df.empty:
                return
            self.check_sharp_waves(v_resampled, t_resampled, norm_power)
            del v_resampled
            self.extract_original_shws(t_resampled)
            with self.lock:
                self.shared_shw_rec_list += self.shw_recs
                self.shared_shw_df_list.append(self.shw_df)
            # self.print(f'finished. SHW found: {len(self.shw_df)}; time taken: {(time.time()-t0)/60:.1f} minutes')
        except Exception as exc:
            self.print(f'ERROR: {exc}\n{traceback.format_exc()}')

    def get_sharp_waves(self, v, t, fs=None) -> (np.ndarray, list):
        fs = fs or self.fs
        # power = -filtfilt(self.mfilt, 1, v)
        # norm_power = power / power.max()
        m = self.mfilt.copy()
        # power = convolve(v, m, mode='same')# / sum(m)
        power = -filtfilt(m, 1, v)
        # self.print(f'power_max={power.max():.1f}, power_min={power.min():.1f}')
        norm_power = power / power.max()
        # norm_power = (power - power.min()) / (power.max() - power.min())
        # self.print(f'norm_power_max={norm_power.max():.1f}, norm_power_min={norm_power.min():.1f}')
        peaks, _ = find_peaks(norm_power, height=self.shw_thresh, distance=1*fs, prominence=0.6, width=self.max_width*fs)
        window = int(self.shw_duration_sec * fs)
        start_indices = np.array([peak - (window//2) for peak in peaks])
        if len(start_indices) > 0:
            t_ = t[start_indices]
            self.shw_df = pd.DataFrame({'start': start_indices, 'end': start_indices + window, 't_start': t_,
                                        't_end': t_ + self.shw_duration_sec})
        else:
            self.shw_df = pd.DataFrame()
        return norm_power

    def check_sharp_waves(self, v, t, norm_power):
        df = apply_parallel_dask(self.shw_df, row_shw_check, v, t)
        self.shw_df = pd.concat([self.shw_df, df], axis=1)
        # add the power values for each ShW
        self.shw_df['power'] = norm_power[self.shw_df.start + (self.shw_df.end - self.shw_df.start) // 2]
        if self.max_width:
            self.shw_df = self.shw_df.query(f'width<{self.max_width}')

    def extract_original_shws(self, t_resampled):
        for i, row in self.shw_df.copy().iterrows():
            t_start = t_resampled[int(row.start)]
            i_start = int(np.argmin(np.abs(self.t - t_start)))
            g_i_start = i_start + self.global_i_start
            self.shw_recs.append((g_i_start, self.v[i_start:(i_start + self.shw_w)], self.t[i_start:(i_start + self.shw_w)]))
            # fix shw_df (change to original values)
            self.shw_df.loc[i, 'start'] = g_i_start
            self.shw_df.loc[i, 'end'] = g_i_start + self.shw_w

    def print(self, msg):
        print(f'Process #{self.proc_id}: {msg}')


def row_shw_check(row, v, t):
    t_ = t[int(row.start):int(row.end)].flatten()
    v_ = v[int(row.start):int(row.end)].flatten()
    try:
        hmx = half_max_x(t_, -v_)
        s = {'width': hmx[1] - hmx[0], 'depth': max(-v_), 'p2v': max(v_) - min(v_)}
    except Exception:
        s = {'width': None, 'depth': None, 'p2v': None}
    return pd.Series(s)
