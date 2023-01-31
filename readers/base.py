import datetime
import re
import os
from typing import Union
import pywt
from dateutil import parser
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from IPython.display import display, HTML
from scipy.io import loadmat
from scipy.signal import resample
from dataclasses import dataclass
from neo.rawio import NeuralynxRawIO, RawBinarySignalRawIO
from readers.openephys import OpenEphysRawIO
from utils import butter_lowpass_filter, butter_bandpass_filter, butter_highpass_filter, apply_parallel_dask

DEFAULT_TABLE = '/media/sil2/Data/Lizard/Stellagama/brainStatesSS.xlsx'


@dataclass
class Reader:
    animal_id: str = None
    rec_id: str = None
    root_dir: [str, Path] = None
    channel: int = None
    is_debug: bool = True
    window: int = 2.5
    overlap: float = 0.5
    wavelet = None
    fs: int = None
    desired_fs: int = None
    recording_fs: int = None
    start_timestamp: datetime.datetime = None
    excel_table: Union[str, pd.DataFrame] = None  # path for alternative excel table or Dataframe
    use_multiprocessing = True
    use_slow_cycles = True
    is_flat_cwt = True
    _parser_cls = None

    def __post_init__(self):
        assert (self.animal_id and self.rec_id) or (self.root_dir and self.channel), \
            'You must specify either animal_id and rec_id or root_dir and channel'
        assert not self.root_dir or (self.root_dir and self.channel), 'you must provide channel with root_dir'
        assert self.overlap is None or 0 <= self.overlap <= 1, 'Overlap must be between 0-1'
        if isinstance(self.excel_table, str):
            assert Path(self.excel_table).exists(), 'Provided excel table path not exist'
        self.excel_table = self.load_excel_table()
        self.load_rec_params()
        self._init_reader()
        self.load_metadata()
        self.load_start_timestamp()
        assert self.fs is not None, 'No sampling frequency was loaded'
        self.w = int(self.fs * self.window)
        self.noverlap = int(self.w * self.overlap)

    def __str__(self):
        return f'{self.animal_id},{self.rec_id}'

    def load_rec_params(self):
        if not self.root_dir:
            res = self.excel_table.query(f'Animal=="{self.animal_id}" and recNames=="{self.rec_id}"')
            if res.empty:
                raise Exception(f'unable to find recording for animal_id: {self.animal_id} and rec: {self.rec_id}')
            if len(res) > 1:
                raise Exception(f'more than one option for animal_id: {self.animal_id} and rec: {self.rec_id}')
            self.excel_table = res.iloc[0]

            self.root_dir = self.excel_table.folder
            if not self.channel:
                self.channel = self.excel_table.defaulLFPCh

        self.root_dir = Path(self.root_dir)
        self.channel = int(self.channel)

    def read(self, i_start=None, i_stop=None, resample_q=None, t_start=None, t_stop=None,
             lowpass=None, bandpass=None, highpass=None, filter_order=5):
        assert sum(x is not None for x in [lowpass, bandpass, highpass]) <= 1, \
            'You must either choose lowpass, bandpass or highpass'
        if t_start:
            i_start = np.argmin(np.abs(self.time_vector - t_start))
        if t_stop:
            i_stop = np.argmin(np.abs(self.time_vector - t_stop))
        v, t = self._read(i_start, i_stop)

        if self.desired_fs:
            if resample_q or lowpass or bandpass or highpass:
                self.print('ignoring resample_q or lowpass or bandpass or highpass in desired_fs configuration')
            v = butter_lowpass_filter(v, self.desired_fs, self.recording_fs, order=filter_order)
            q = self.recording_fs / self.desired_fs
            v, t = self.resample(v, t, q)
        else:
            if resample_q:
                v, t = self.resample(v, t, resample_q)
            if lowpass is not None:
                v = butter_lowpass_filter(v, lowpass, self.fs, order=filter_order)
            elif bandpass is not None:
                assert isinstance(bandpass, (list, tuple)) and len(bandpass) == 2, 'bad bandpass. Should be like (20, 50)'
                v = butter_bandpass_filter(v, bandpass[0], bandpass[1], self.fs, order=filter_order)
            elif highpass is not None:
                v = butter_highpass_filter(v, highpass, self.fs, order=filter_order)

        return v, t

    def _read(self, i_start=None, i_stop=None):
        raise NotImplemented('No read function')

    def _init_reader(self):
        if self._parser_cls is not None:
            self.reader = self._parser_cls(self.root_dir)
        else:
            self.reader = None

    def load_metadata(self):
        self._parse_header()
        self.fs = self.get_sampling_frequency()
        self.recording_fs = self.fs
        if self.desired_fs:
            if self.desired_fs >= self.fs:
                self.print('desired fs is higher than recording fs; removing desired fs')
                self.desired_fs = 0
            else:
                self.fs = self.desired_fs

    def get_sampling_frequency(self):
        raise NotImplemented('No get_sampling_frequency function')

    def _parse_header(self):
        raise NotImplemented('No _parse_header function')

    def load_start_timestamp(self):
        pass

    def read_segmented(self, v):
        return buffer(v, self.w, self.noverlap, self.decimate_q)

    @staticmethod
    def resample(v, t, q) -> (np.ndarray, np.ndarray):
        return resample(v, num=int(len(v)/q), t=t)

    def read_segmented_old(self, i_start=None, i_stop=None, v=None, t=None, is_use_sc=True):
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

        V, start_indices = buffer(v, self.w, self.noverlap, self.decimate_q)
        self.print(f'Number of segments after buffering: {len(start_indices)}')

        if self.wavelet is not None:
            self.print('start run of segment function...')
            scales = get_scales(self.wavelet, self.fs)
            self.print(f'scales used: {scales}')
            if self.use_multiprocessing:
                self.print('using multi-processing for feature extraction')
                n_cpu = mp.cpu_count() - 1
                with mp.Pool(processes=n_cpu) as p:
                    V = p.starmap(cwt_feature_extraction, ((v, scales, self.wavelet) for v in np.array_split(V, n_cpu)))
            else:
                V = cwt_feature_extraction(V, scales, self.wavelet)
            if self.is_flat_cwt:
                V = np.vstack([v.flatten() for v in V])

        if is_use_sc:
            sig_df = self.create_sig_df(t, start_indices)
        else:
            sig_df = pd.DataFrame({})
        return V, start_indices, sig_df

    def create_sig_df(self, t, start_indices, w=None) -> pd.DataFrame:
        w = w or self.w
        sig_df = []
        sc = self.load_slow_cycles()
        self.print('start creation of sig_df...')
        for startIndex in (tqdm(start_indices) if self.is_debug else start_indices):
            end_t_index = startIndex + w if startIndex + w < len(t) else len(t) - 1
            if self.use_slow_cycles:
                sc_id = np.where((sc['on'] <= t[startIndex]) &
                                 (sc['off'] >= t[end_t_index]))[0]
                sc_id = int(sc_id[0]) if len(sc_id) == 1 else np.nan
                if isinstance(sc_id, int):
                    group = 1 if t[startIndex] < sc.loc[sc_id, 'mid'] else 2
                else:
                    group = np.nan
            else:
                sc_id, group = np.nan, np.nan
            sig_df.append((startIndex, startIndex + w, group, sc_id))
        sig_df = pd.DataFrame(sig_df, columns=['start', 'end', 'group', 'signal'])
        sig_df[['start', 'end']] = sig_df[['start', 'end']].astype('int')
        return sig_df

    def load_slow_cycles(self) -> pd.DataFrame:
        cols = ['on', 'mid', 'off']
        if not self.use_slow_cycles:
            return pd.DataFrame(columns=cols)
        assert self.channel, 'No channel was provided'
        assert self.root_dir, 'No root directory was provided'
        p = self.analysis_folder / f'slowCycles_ch{self.channel}.mat'
        assert p.exists(), f'cannot find slow cycles file in: {p}'
        sc = loadmat(p)
        sc = pd.DataFrame(np.vstack([sc[x] for x in ['TcycleOnset', 'TcycleMid', 'TcycleOffset']]).T,
                            columns=cols) / 1000
        sc = sc.astype({k: int for k in cols})
        return sc

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

    def load_excel_table(self, excel_path=None) -> pd.DataFrame:
        excel_path = excel_path or self.excel_table
        if isinstance(excel_path, pd.DataFrame):
            return excel_path
        path = Path(excel_path or DEFAULT_TABLE)
        return pd.read_excel(path)

    def p2v(self):
        ac_path = self.analysis_folder / f'dbAutocorr_ch{self.channel}.mat'
        if not ac_path.exists():
            print(f'Unable to find p2v value in {ac_path.parent}')
            return
        try:
            m = loadmat(ac_path)
            p2v = m['peak2VallyDiff'].item()
        except Exception as exc:
            self.print(f'unable to extract P2V value from analysis file; {exc}')
            p2v = None
        return p2v

    @property
    def analysis_folder(self):
        return self.root_dir / 'analysis' / f'Animal={self.animal_id},recNames={self.rec_id}'

    @property
    def t_lights_off(self):
        if self.start_timestamp:
            dt_off = datetime.datetime.combine(self.start_timestamp.date(), datetime.time(hour=19))
            return (dt_off - self.start_timestamp).total_seconds()

    @property
    def t_lights_on(self):
        if self.start_timestamp:
            next_day = self.start_timestamp.date() + datetime.timedelta(days=1)
            dt_on = datetime.datetime.combine(next_day, datetime.time(hour=7))
            return (dt_on - self.start_timestamp).total_seconds()


class NeoReader(Reader):
    """Readers based on the neo package. https://neo.readthedocs.io/"""

    def __init__(self, *args, **kwargs):
        assert self._parser_cls is not None, 'You must use a specific reader'
        self.reader = None
        self.time_vector = None
        self.read_channels = None  # will be assigned in load_rec_params.
        super().__init__(*args, **kwargs)
        self.units = self.reader.header['signal_channels'][0]['units']

    def load_metadata(self):
        super().load_metadata()
        self.time_vector = self.get_full_time_vector()

    def get_sampling_frequency(self):
        return self.reader.get_signal_sampling_rate()

    def _parse_header(self):
        self.reader.parse_header()

    def load_rec_params(self):
        super().load_rec_params()
        self.read_channels = [self.channel]

    def get_full_time_vector(self):
        t_start = self.reader.segment_t_start(block_index=0, seg_index=0)
        t_stop = self.reader.segment_t_stop(block_index=0, seg_index=0)
        return np.arange(t_start, t_stop, (1/self.recording_fs))

    def get_time_vector(self, v, i_start, i_stop):
        t = self.time_vector[i_start:i_stop]
        if len(t) > len(v):
            t = t[:len(v)]
        return t

    def _read(self, i_start=None, i_stop=None):
        raw_sigs = self.reader.get_analogsignal_chunk(block_index=0, seg_index=0, i_start=i_start, i_stop=i_stop,
                                                      channel_indexes=self.read_channels)
        v = self.reader.rescale_signal_raw_to_float(raw_sigs, dtype='float64').flatten()
        t = self.get_time_vector(v, i_start, i_stop)
        return v, t

    @property
    def cache_dir_path(self):
        return self.root_dir / 'regev_cache'


class NeuralynxReader(NeoReader):
    # TODO: fix the exception in nlxheader.py (line 253) for Mark's recordings
    _parser_cls = NeuralynxRawIO

    def _read(self, i_start=None, i_stop=None):
        raw_sigs = self.reader.get_analogsignal_chunk(block_index=0, seg_index=0, i_start=i_start, i_stop=i_stop,
                                                      channel_names=[f'CSC{self.channel}'])
        v = self.reader.rescale_signal_raw_to_float(raw_sigs, dtype='float64',
                                                       channel_names=[f'CSC{self.channel}']).flatten()
        t = self.get_time_vector(v, i_start, i_stop)
        return v, t

    def load_rec_params(self):
        pass


class OpenEphysReader(NeoReader):
    _parser_cls = OpenEphysRawIO

    def _init_reader(self):
        self.reader = self._parser_cls(dirname=self.root_dir, channels=[self.channel])

    def get_full_time_vector(self):
        t_start = self.reader.segment_t_start(block_index=0, seg_index=0)
        t_stop = self.reader.segment_t_stop(block_index=0, seg_index=0)
        return np.arange(0, t_stop - t_start, (1 / self.recording_fs))


class BinaryReader(NeoReader):
    _parser_cls = RawBinarySignalRawIO

    def __init__(self, animal_id, rec_id, fs=20000, **kwargs):
        super().__init__(animal_id, rec_id, fs=fs, **kwargs)
        self.nb_channel = 32

    def _init_reader(self):
        filename = self.excel_table['MEAfiles']
        if not filename:
            self.print('Unable to load binary file; MEAfiles is not specified in the excel')
            raise Exception('')
        if filename == 'ch1_32.bin':
            nb_channel = 32
        else:
            nb_channel = 1
            self.read_channels = None
            m = re.search(r'ch(\d+).bin', filename)
            if m:
                self.channel = int(m.group(1))
        self.binary_file_path = self.root_dir / filename
        alternative = self.root_dir / f'ch{self.channel}.bin'
        if not self.binary_file_path.exists():
            if alternative.exists():
                self.binary_file_path = alternative
            else:
                raise Exception(f'Both options for binary file do not exist: {self.binary_file_path} and {alternative}')

        self.reader = self._parser_cls(self.binary_file_path.as_posix(), nb_channel=nb_channel, sampling_rate=self.fs)

    @property
    def analysis_folder(self):
        rec_id = self.rec_id
        if rec_id.endswith('b'):
            rec_id = rec_id[:-1]
        analysis_dir = self.root_dir.parent / 'analysis'
        if not analysis_dir.exists():
            alternative = self.root_dir.parent.parent / 'analysis'
            if alternative.exists():
               analysis_dir = alternative
            else:
                raise Exception(f'Cannot find analysis folder in {analysis_dir} and {alternative}')

        return analysis_dir / f'Animal={self.animal_id},recNames={rec_id}'

    def load_start_timestamp(self):
        try:
            for p in self.root_dir.parts[:4:-1]:
                m_long = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', p)
                m_short = re.search(r'\d{2}_\d{2}_2\d', p)
                if not (m_long or m_short):
                    continue

                if m_long:
                    self.start_timestamp = datetime.datetime.strptime(m_long.group(), '%Y-%m-%d_%H-%M-%S')
                else:
                    # TODO: find a better way to load the start_timestamp for binary recordings with no timestamp in root_dir
                    self.start_timestamp = datetime.datetime.strptime(f'{m_short.group()}_17-00-00', '%d_%m_%y_%H-%M-%S')
        except:
            pass


class ExcelReader:
    reader_classes = {
        'binaryRecording': BinaryReader,
        'OERecording': OpenEphysReader
    }

    def __init__(self, xls: Union[str, pd.DataFrame], is_remove_excluded=True, **reader_kwargs):
        self.reader_kwargs = reader_kwargs
        if isinstance(xls, pd.DataFrame):
            self.rec_table = xls
        else:
            assert Path(xls).exists(), f'cannot find excel file: {xls}'
            self.rec_table = pd.read_excel(xls)
        if is_remove_excluded:
            self.rec_table = self.rec_table.query('Exclude!=Exclude')

    def _repr_html_(self):
        return self.rec_table._repr_html_()

    def get(self, animal_id, rec_id, desired_fs=None):
        # TODO: add option to change channel, also to multi channels
        res = self.get_row(animal_id, rec_id)
        rp = self[res.index[0]]
        if desired_fs:
            rp.desired_fs = desired_fs
        return rp

    def __getitem__(self, item):
        row = self.rec_table.iloc[item]
        reader_name = row.get('recFormat')
        assert reader_name in self.reader_classes, f'unknown recording format: {reader_name}'
        rp = self.reader_classes[reader_name](row['Animal'], row['recNames'],
                                              excel_table=self.rec_table, **self.reader_kwargs)
        return rp

    def __len__(self):
        return len(self.rec_table)

    def get_row(self, animal_id, rec_id) -> pd.Series:
        res = self.rec_table.query(f'Animal=="{animal_id}" and recNames=="{rec_id}"')
        if res.empty:
            raise Exception(f'No results for {animal_id} and rec: {rec_id}')
        elif len(res) > 1:
            raise Exception(f'More than one result for {animal_id}, recording: {rec_id}')
        return res

    def filter(self, recs):
        # rec format example: 'SA04, Night-8b'
        new_df = []
        for r in recs:
            animal_id, rec_id = r.split(',')
            new_df.append(self.get_row(animal_id, rec_id))

        return ExcelReader(pd.concat(new_df), **self.reader_kwargs)


def buffer(X: np.ndarray, w, noverlap=0, decimate_q=None):
    """buffers data vector X into length n column vectors with overlap p; excess data at the end of X is discarded"""
    w = int(w)  # length of each data vector
    noverlap = int(noverlap)  # overlap of data vectors, 0 <= p < n-1
    L = len(X)  # length of data to be buffered
    m = int(np.floor((L - w) / (w - noverlap)) + 1)  # number of sample vectors (no padding)
    start_indices = list(range(0, L - w, w - noverlap))
    all_iterations = list(zip(start_indices, range(0, m)))
    for startIndex, segment_id in all_iterations:
        x = X[startIndex:startIndex + w]
        if decimate_q:
            x = resample(x, len(x) // decimate_q)
        yield x, startIndex


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
        res.append(coeffs)
    return np.dstack(res).T  # output shape: (observations, timestamps, scales/features)


def sig_df_worker(startIndex, w, sc, t):
    end_t_index = startIndex + w if startIndex + w < len(t) else len(t) - 1
    sc_id = np.where((sc['on'] <= t[startIndex]) & (sc['off'] >= t[end_t_index]))[0]
    sc_id = int(sc_id[0]) if len(sc_id) == 1 else np.nan
    if isinstance(sc_id, int):
        group = 1 if t[startIndex] < sc.loc[sc_id, 'mid'] else 2
    else:
        group = np.nan
    return sc_id, startIndex, startIndex + w, group
