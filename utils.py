import numpy as np
import pandas as pd
from functools import partial
from joblib import Parallel, cpu_count
import multiprocessing as mp
from pathlib import Path
import dask.dataframe as dd
from scipy.stats import zscore, gaussian_kde, pearsonr
from scipy.special import betainc
from scipy.signal import decimate, butter, filtfilt, sosfilt, sosfreqz, find_peaks, correlate, iirnotch, lfilter, savgol_filter
from scipy.cluster.hierarchy import dendrogram
import matplotlib

ROOT_DIR = Path(__file__).parent.resolve()


def corrcoef(matrix):
    r = np.corrcoef(matrix)
    rf = r[np.triu_indices(r.shape[0], 1)]
    df = matrix.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = betainc(0.5 * df, 0.5, df / (df + ts))
    p = np.zeros(shape=r.shape)
    p[np.triu_indices(p.shape[0], 1)] = pf
    p[np.tril_indices(p.shape[0], -1)] = p.T[np.tril_indices(p.shape[0], -1)]
    p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])
    return r, p


def corrcoef_loop(matrix):
    rows, cols = matrix.shape[0], matrix.shape[1]
    r = np.ones(shape=(rows, rows))
    p = np.ones(shape=(rows, rows))
    for i in range(rows):
        for j in range(i+1, rows):
            r_, p_ = pearsonr(matrix[i], matrix[j])
            r[i, j] = r[j, i] = r_
            p[i, j] = p[j, i] = p_
    return r, p


def autocorr(x, t, ax=None):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n//2 + 1:]
    acorr_norm = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
    lag = np.abs(acorr_norm).argmax() + 1
    r = acorr_norm[lag-1]
    peaks, _ = find_peaks(acorr, prominence=0.1)
    if len(peaks) == 0:
        peaks = [len(acorr)-1]
    t_ = t[n//2 + 1:]
    t_cyc = t_[peaks][0] - t_[0]
    power = acorr[peaks][0] / acorr[0]
    power_norm = acorr_norm[peaks][0] / acorr_norm[0]
    if ax is not None:
        ax.plot(t_, acorr)
        ax.scatter(t_[peaks], acorr[peaks])
        ax.set_title(f'Tcyc = {t_cyc}, power = {power:.2}, power_norm = {power_norm:.2}\nr = {r:.2},  lag={lag}')
    return t_cyc, power, power_norm, r, lag


def cross_correlate(x, y):
    """Calculate and plot cross-correlation (full) between two signals."""
    N = max(len(x), len(y))
    n = min(len(x), len(y))
    if N == len(y):
        lags = np.arange(-N + 1, n)
    else:
        lags = np.arange(-n + 1, N)

    c = correlate(x / np.std(x), y / np.std(y), 'full') / n
    return c, lags


def butter_lowpass_filter(x, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, x)
    return y


def butter_highpass_filter(x, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, x)
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], analog=False, btype='band')
    y = filtfilt(b, a, data)
    # sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    # y = sosfilt(sos, data)
    return y


def notch_filter(data, fs, f0=50.0, Q=30.0):
    """
    Apply notch filter
    @param data: The signal
    @param fs: Sample frequency (Hz)
    @param f0: Frequency to be removed from signal (Hz)
    @param Q: Quality factor
    @return:
    """
    b, a = iirnotch(f0, Q, fs)
    y = lfilter(b, a, data)
    return y


def smooth_signal(v, window_size=51, poly_order=3):
    return savgol_filter(v, window_size, poly_order)


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def plot_spectrogram(signal, sf, ax, window_sec=0.1, maxy=10000):
    """Plot spectrogram and return the calculation values"""
    nfft = int(window_sec * sf)
    noverlap = int(nfft * 0.5)  # overlap set to be half of a segment
    nfft_padded = next_power_of_2(nfft)  # pad segment with zeros for making nfft of power of 2 (better performance)
    vmin = 20*np.log10(np.max(signal)) - 40  # hide anything below -90 dBc
    Sxx, f, t, img = ax.specgram(signal, Fs=sf, NFFT=nfft, noverlap=noverlap,
              pad_to=nfft_padded, mode='magnitude', vmin=vmin)
    # plt.colorbar(img, ax=ax)
    ax.set_ylim([0, maxy])
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title('Spectrogram')
    return Sxx, f, t


def listify(x):
    if isinstance(x, str) or not hasattr(x, '__iter__'):
        x = [x]
    return x


def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))


def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[-1], half)]


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def color_list(n):
    # normalize item number values to colormap
    norm = matplotlib.colors.Normalize(vmin=0, vmax=n)
    # colormap possible values = viridis, jet, spectral
    return [matplotlib.cm.jet(norm(i), bytes=True) for i in range(n)]


def apply_parallel_dask(df, func, *args, **kwargs):
    """Run parallel using Dask"""
    ddata = dd.from_pandas(df, npartitions=30)
    def apply_myfunc_to_DF(df_): return df_.apply(lambda row: func(row, *args, **kwargs), axis=1)
    def dask_apply(): return ddata.map_partitions(apply_myfunc_to_DF).compute(scheduler='processes')
    return dask_apply()


def run_parallel(func, array, n_jobs=None, **kwargs):
    """
    Run an array in parallel
    @param func: main function whose args are the array iterable and the rest are kwargs
    @param array: list or numpy array (if np array, so the array is splitted according to n_jobs)
    @param n_jobs: number of different jobs (CPUs)
    @param kwargs: kwargs for func
    @return: list of results
    """
    n_jobs = n_jobs or cpu_count() // 2
    if isinstance(array, np.ndarray):
        iterable = np.array_split(array, n_jobs)
    elif isinstance(array, list):
        iterable = array
    else:
        raise Exception(f'Bad type of array; {type(array)}')
    with mp.Pool(n_jobs) as pool:
        result = pool.map(partial(func, **kwargs), iterable)
    # result = Parallel(n_jobs=n_jobs)((func, a, kwargs) for a in iterable)
    return result
