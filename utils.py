import numpy as np
from scipy.stats import zscore, gaussian_kde
from scipy.signal import decimate, butter, filtfilt, sosfilt, sosfreqz


def butter_lowpass_filter(x, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, x)
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    y = sosfilt(sos, data)
    return y


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
