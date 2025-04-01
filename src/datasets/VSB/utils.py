import numpy as np
import pywt
from scipy import signal
import time
from scipy.signal import butter
from numpy.fft import rfft, rfftfreq, irfft


def low_pass(s, threshold=1e4):
    fourier = rfft(s)
    frequencies = rfftfreq(len(s), d=2e-2 / s.size)
    fourier[frequencies > threshold] = 0

    return irfft(fourier)


def get_crossing(x):
    x_1 = low_pass(x)
    x = x_1.reshape((-1,))
    zero_crossing = np.where(np.diff(np.sign(x)))[0]
    up_crossing = -1
    for zc in zero_crossing:
        if x[zc] < 0 and x[zc + 1] > 0:
            up_crossing = zc
    return up_crossing


def phase_shift(x, cross):
    if cross > 0:
        x = np.hstack([x[cross:], x[:cross]])
    return x


n_samples = 800000

# Sample duration is 20 miliseconds
sample_duration = 0.02

# Sample rate is the number of samples in one second
# Sample rate will be 40mhz
sample_rate = n_samples * (1 / sample_duration)


def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def high_pass_filter(x, low_cutoff=1000, sample_rate=sample_rate):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    """

    # nyquist frequency is half the sample rate https://en.wikipedia.org/wiki/Nyquist_frequency
    nyquist = 0.5 * sample_rate
    norm_low_cutoff = low_cutoff / nyquist

    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    # scipy version 1.2.0
    # sos = butter(10, low_freq, btype='hp', fs=sample_fs, output='sos')

    # scipy version 1.1.0
    sos = butter(10, Wn=[norm_low_cutoff], btype="highpass", output="sos")
    filtered_sig = signal.sosfilt(sos, x)

    return filtered_sig


def denoise_signal(x, wavelet="db4", level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """

    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")

    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1 / 0.6745) * maddest(coeff[-level])

    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])

    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode="per")


def dwt_signal_denoising(train_df, signal_ids, wavelet="haar", level=1):
    """
    Denoise the signals using wavelet transform
    """
    start_time = time.time()
    all_denoised_signals = []

    for index in signal_ids:
        if np.mod(index, 100) == 0:
            print(index)
            print("Elapsed time: {}".format(time.time() - start_time))

        signal = train_df[str(index)].values
        hp_signal = high_pass_filter(signal, low_cutoff=10000, sample_rate=sample_rate)
        denoised_signal = denoise_signal(hp_signal, wavelet=wavelet, level=level)

        all_denoised_signals.append(denoised_signal)

    return all_denoised_signals


def dwt_single_signal_aligned_denoising(signal, wavelet="haar", level=1, aligned=True):
    """
    Denoise the single signals using wavelet transform
    """
    if aligned:
        crossing = get_crossing(signal)
        signal = phase_shift(signal, crossing)
    hp_signal = high_pass_filter(signal, low_cutoff=10000, sample_rate=sample_rate)
    denoised_signal = denoise_signal(hp_signal, wavelet=wavelet, level=level)

    return denoised_signal
