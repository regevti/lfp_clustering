import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mne.time_frequency import morlet
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, filtfilt
from readers.mat_files import MatRecordingsParser
from utils import half_max_x, apply_parallel


class SharpWavesFinder:
    def __init__(self, reader: (str, MatRecordingsParser), is_debug=True, shw_duration=1.2):
        self.reader = reader
        self.fs = reader.fs
        self.shw_w = int(self.fs * shw_duration)
        self.norm_conv = None
        self.thresh = None
        self.is_debug = is_debug
        self.v = None
        self.t = None
        self.sc = reader.load_slow_cycles()
        self.shw_df = None

    def train(self, cf=1, nc=1, thresh=0.15, mfilt=None, i_start=None, i_stop=None):
        self.v, self.t = self.reader.read(i_start, i_stop)
        self.get_sharp_waves(cf, nc, thresh=thresh, mfilt=mfilt)
        self.check_sharp_waves()
        self.thresh = thresh

    def get_sharp_waves(self, cf, nc, thresh, mfilt=None) -> (np.ndarray, list):
        peaks = []
        V, conv_start_indices, _ = self.reader.read_segmented(v=self.v, t=self.t)
        self.norm_conv = np.zeros(self.v.shape)
        for i, start_id in zip(range(V.shape[0]), conv_start_indices):
            if mfilt is None:
                # Convolve the wavelet and extract magnitude and phase
                wlt = morlet(self.reader.fs, [cf], n_cycles=nc)[0]
                analytic = np.convolve(V[i, :], wlt, mode='same')
            else:
                analytic = filtfilt(mfilt, 1, V[i, :])
            power = -analytic
            # norm_power_ = (power - power.min()) / (power.max() - power.min())
            norm_power_ = power / power.max()
            peaks_, _ = find_peaks(norm_power_, height=thresh, distance=self.fs*0.5, width=self.fs*0.1)
            peaks.extend([p + start_id for p in peaks_ if p > 0 and V[i, p] < -10])
            self.norm_conv[start_id:start_id+len(norm_power_)] = norm_power_

        half_w = int(self.shw_w / 2)
        # self.ShWs = np.vstack([self.v[peak - half_w:peak + half_w]
        #                        for peak in peaks if peak >= half_w and peak + half_w < len(self.v)])
        # start_indices = [peak - half_w for peak in peaks]
        start_indices = [peak - half_w for peak in peaks]
        self.print(f'Number of sharp waves found: {len(peaks)}')
        self.shw_df = self.reader.create_sig_df(self.t, start_indices, w=self.shw_w)

    def check_sharp_waves(self):
        self.print('start ShW checking...')
        # df = pd.DataFrame(index=self.shw_df.index, columns=['width'])
        df = apply_parallel(self.shw_df, row_shw_check, self.v, self.t)
        self.shw_df = pd.concat([self.shw_df, df], axis=1)
        # add the power values for each ShW
        self.shw_df['power'] = self.norm_conv[self.shw_df.start + (self.shw_df.end - self.shw_df.start) // 2]

    def calc_sw_rate(self, wt, overlap=0.75, label=None, lights_off_only=False) -> pd.DataFrame:
        """Calculate the rate of slow-waves across all recording"""
        noverlap = int(wt * overlap)
        rf = pd.DataFrame(columns=['rate', 'time', 'group_time', 'group', 'signal', 'datetime'])
        sig_df = self.shw_df.query(f'label=={label}') if label is not None else self.shw_df.copy()
        idx = (sig_df.start + self.shw_w).to_numpy().astype(int)
        idx = idx[idx < len(self.t)]  # remove out of range indices
        sws_times = self.t[idx]
        startT = 0
        endT = self.t.max()
        if lights_off_only:
            startT = self.t[self.reader.lights_off_id] - 2 * 60 * 60
            endT = self.t[self.reader.lights_on_id] + 2 * 60 * 60
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
            if self.reader.t_start is not None:
                dt = self.reader.t_start + pd.to_timedelta(t_start, unit='seconds')
            else:
                dt = None
            rf = rf.append({'rate': r, 'time': t_start + (wt/2), 'group_time': group_time, 'group': group,
                            'signal': sc_id, 'datetime': dt}, ignore_index=True)
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

    def plot_cycle_with_sharp_waves(self, cycle_id, split=1):
        v, t, start_id = self.reader.get_sleep_cycle(cycle_id, t=self.t, v=self.v)
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
        print(f'# of slow waves: {n_shw}')
        return V

    def plot_sharp_waves_detection(self, t_start, t_end, ax, ax2):
        idx = np.where((self.t >= t_start) & (self.t <= t_end))[0]
        v = self.v[idx]
        t = self.t[idx]
        sf = self.shw_df.query(f'start>={idx[0]} and end<={idx[-1]}')
        ax.plot(t, v, 'k')
        for i, row in sf.iterrows():
            ax.axvspan(self.t[int(row.start)], self.t[int(row.end)], facecolor='g', alpha=0.4)
        ax.axis('off')
        ylim = ax.get_ylim()
        print(f'Ylim: {ylim[1] - ylim[0]}')
        conv = self.norm_conv[idx]
        peaks_, _ = find_peaks(conv, height=self.thresh, distance=self.fs * 0.5, width=self.fs * 0.1)
        peaks_ = [p for p in peaks_ if p > 0 and v[p] < -10]
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

    def print(self, s):
        if self.is_debug:
            print(s)


def row_shw_check(row, v, t):
    t_ = t[int(row.start):int(row.end)].flatten()
    v_ = v[int(row.start):int(row.end)].flatten()
    try:
        hmx = half_max_x(t_, -v_)
        s = {'width': hmx[1] - hmx[0], 'depth': max(-v_)}
    except Exception:
        s = {'width': None, 'depth': None}
    return pd.Series(s)
