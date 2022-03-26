import pywt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.io import loadmat
from scipy.signal import decimate
from neo.rawio import NeuralynxRawIO
from utils import butter_lowpass_filter


class Reader:
    def __init__(self, root_dir=None, channel=None, is_debug=True,
                 window=None, overlap=None, lowpass=None, decimate_q=None, wavelet=None):
        assert overlap is None or 0 <= overlap <= 1, 'Overlap must be between 0-1'
        self.root_dir = Path(root_dir)
        self.channel = channel
        self.is_debug = is_debug
        self.fs = None
        self.load_metadata()
        assert self.fs is not None, 'No sampling frequency was loaded'
        self.w = int(self.fs * window)
        self.noverlap = int(self.w * overlap)
        self.lowpass = lowpass
        self.decimate_q = decimate_q
        self.wavelet = wavelet

    def read(self, i_start=None, i_stop=None):
        raise NotImplemented('No read function')

    def load_metadata(self):
        raise NotImplemented('No load_metadata function')

    def read_segmented(self, i_start=None, i_stop=None, v=None, t=None):
        """
        Read a segmented and processed signal
        :param w: Window time in seconds
        :param overlap: Overlap ratio (range: 0-1)
        :param i_start: Start index for reading
        :param i_stop: Stop index for reading
        :param lowpass: The lowpass value
        :param decimate_q: Decimation parameter
        :param wavelet: Name of wavelet to be used for continuous wavelet decomposition
        :return: Segmented matrix of the signal
        """
        if v is None or t is None:
            v, t = self.read(i_start, i_stop)
        if self.lowpass is not None:
            v = butter_lowpass_filter(v, self.lowpass, self.fs, order=5)

        V, start_indices = buffer(v, self.w, self.noverlap, self.decimate_q, self.is_debug)
        self.print(f'Number of segments after buffering: {len(start_indices)}')
        if self.decimate_q is not None:
            self.fs = self.fs / self.decimate_q

        if self.wavelet is not None:
            n_cpu = mp.cpu_count() - 1
            self.print('start parallel run of segment function...')
            scales = get_scales(self.wavelet, self.fs)
            self.print(f'scales used: {scales}')
            with mp.Pool(processes=n_cpu) as p:
                V = p.starmap(cwt_feature_extraction, ((v, scales, self.wavelet) for v in np.array_split(V, n_cpu)))
            V = np.vstack(V)

        sig_df = self.create_sig_df(t, start_indices)
        return V, start_indices, sig_df

    def create_sig_df(self, t, start_indices) -> pd.DataFrame:
        sig_df = []
        sc = self.load_slow_cycles()
        self.print('start creation of sig_df...')
        for startIndex in (tqdm(start_indices) if self.is_debug else start_indices):
            end_t_index = startIndex + self.w if startIndex + self.w < len(t) else len(t) - 1
            sc_id = np.where((sc['on'] <= t[startIndex]) &
                             (sc['off'] >= t[end_t_index]))[0]
            sc_id = int(sc_id[0]) if len(sc_id) == 1 else np.nan
            if isinstance(sc_id, int):
                group = 1 if t[startIndex] < sc.loc[sc_id, 'mid'] else 2
            else:
                group = np.nan
            sig_df.append((sc_id, startIndex, startIndex + self.w, group))

        return pd.DataFrame(sig_df, columns=['signal', 'start', 'end', 'group'])

    def load_slow_cycles(self) -> pd.DataFrame:
        assert self.channel, 'No channel was provided'
        assert self.root_dir, 'No root directory was provided'
        p = (self.root_dir / 'analysis')
        assert p.exists() and p.is_dir(), f'Analysis folder not exist in {self.root_dir}'
        files = list(p.rglob(f'slowCycles_ch{self.channel}.mat'))
        assert len(files) > 0, f'unable to find slow cycles file for channel {self.channel}'
        sc = loadmat(files[0].as_posix())
        return pd.DataFrame(np.vstack([sc[x] for x in ['TcycleOnset', 'TcycleMid', 'TcycleOffset']]).T,
                            columns=['on', 'mid', 'off']) / 1000

    def get_sleep_cycle(self, cycle_id, t, v, is_plot=False) -> (np.ndarray, np.ndarray, int):
        sc = self.load_slow_cycles()
        assert cycle_id < len(sc), f'Cycle {cycle_id} is out of sleep cycles range; ' \
                                   f'Number of sleep cycles: {len(sc)}'
        cycle_times = sc.iloc[cycle_id]
        start_id = np.argmin(np.abs(t - cycle_times.on))
        end_id = np.argmin(np.abs(t - cycle_times.off))
        v = v.flatten()[start_id:end_id]
        t = t[start_id:end_id]
        if is_plot:
            plt.figure(figsize=(25,5))
            plt.plot(t, v)
            plt.title(f'Cycle {cycle_id}')
        return v, t, start_id

    def print(self, s):
        if self.is_debug:
            print(s)


class NeoReader(Reader):
    _parser_cls = None

    def __init__(self, root_dir, channel):
        super().__init__(root_dir, channel)
        assert self._parser_cls is not None, 'You must use a specific reader'
        self.reader = self._parser_cls(root_dir)
        self.reader.parse_header()
        self.fs = self.reader.get_signal_sampling_rate()
        self.t_start = self.reader.get_signal_t_start(block_index=0, seg_index=0)
        self.units = self.reader.header['signal_channels'][0]['units']
        self.sc = self.load_slow_cycles()

    def read(self, i_start=None, i_stop=None):
        raw_sigs = self.reader.get_analogsignal_chunk(block_index=0, seg_index=0, i_start=i_start, i_stop=i_stop,
                                                      channel_indexes=[self.channel])
        return self.reader.rescale_signal_raw_to_float(raw_sigs, dtype='float64').flatten()

    @property
    def cache_dir_path(self):
        return self.root_dir / 'regev_cache'


class NeuralynxReader(NeoReader):
    _parser_cls = NeuralynxRawIO

    def read(self, i_start=None, i_stop=None):
        raw_sigs = self.reader.get_analogsignal_chunk(block_index=0, seg_index=0, i_start=i_start, i_stop=i_stop,
                                                      channel_names=[f'CSC{self.channel}'])
        return self.reader.rescale_signal_raw_to_float(raw_sigs, dtype='float64',
                                                       channel_names=[f'CSC{self.channel}']).flatten()


def buffer(X: np.ndarray, w, noverlap=0, decimate_q=None, is_debug=True):
    """buffers data vector X into length n column vectors with overlap p; excess data at the end of X is discarded"""
    w = int(w)  # length of each data vector
    noverlap = int(noverlap)  # overlap of data vectors, 0 <= p < n-1
    L = len(X)  # length of data to be buffered
    m = int(np.floor((L - w) / (w - noverlap)) + 1)  # number of sample vectors (no padding)
    start_indices = list(range(0, L - w, w - noverlap))
    all_iterations = list(zip(start_indices, range(0, m)))
    data = []
    for startIndex, segment_id in all_iterations:
        x = X[startIndex:startIndex + w]
        if decimate_q:
            x = decimate(x, decimate_q)
        data.append(x)
    data = np.vstack(data)
    if is_debug:
        print(f'Buffered Matrix size: {data.shape}')
    return data, start_indices


def get_scales(wavelet, fs, freqs=None):
    """Find the relevant scales to the desired frequencies"""
    f = []
    s = np.arange(1, 100000, 5)
    for i in s:
        f.append(pywt.scale2frequency(wavelet, i) * fs)
    scales = []
    desired_freqs = freqs if freqs is not None else np.arange(1, 40, 2)
    for j in desired_freqs:
        i = np.argmin(np.abs(np.array(f) - j))
        scales.append(s[i])
    return sorted(list(set(scales)))


def cwt_feature_extraction(V, scales, wavelet):
    res = []
    for i in range(V.shape[0]):
        v = V[i, :]
        coeffs, _ = pywt.cwt(v, scales, wavelet)
        res.append(coeffs.flatten())
    return np.vstack(res)


def sig_df_worker(startIndex, w, sc, t):
    end_t_index = startIndex + w if startIndex + w < len(t) else len(t) - 1
    sc_id = np.where((sc['on'] <= t[startIndex]) & (sc['off'] >= t[end_t_index]))[0]
    sc_id = int(sc_id[0]) if len(sc_id) == 1 else np.nan
    if isinstance(sc_id, int):
        group = 1 if t[startIndex] < sc.loc[sc_id, 'mid'] else 2
    else:
        group = np.nan
    return sc_id, startIndex, startIndex + w, group
