from numpy import std, mean, max, min, std, sqrt
import numpy as np
from scipy.fftpack import fftfreq
from scipy.fftpack import fft
from scipy.stats import kurtosis
from nitime.algorithms import AR_est_LD

def signal_range(data):

    sig_range = max(data) - min(data)

    return sig_range


def signal_kurtosis(data):

    sig_kurtosis = kurtosis(data)

    return sig_kurtosis


def signal_mean(data):

    sig_mean = np.mean(data)

    return sig_mean


def signal_mean_abs(data):

    sig_abs = np.abs(data)
    sig_mean = signal_mean(sig_abs)

    return sig_mean

def a_histogram(data):

    descriptor = np.zeros(3)

    ncell = np.ceil(np.sqrt(len(data)))

    max_val = np.max(data)
    min_val = np.min(data)
    delta = (max_val - min_val) / (len(data) - 1)

    descriptor[0] = min_val - delta / 2
    descriptor[1] = max_val + delta / 2
    descriptor[2] = ncell

    h = np.histogram(data, np.int(ncell))

    return h[0], descriptor

def signal_entropy(data):

    h, d = a_histogram(data)

    lowerbound = d[0];
    upperbound = d[1];
    ncell = int(d[2]);

    estimate = 0
    sigma = 0
    count = 0

    for isess in range(ncell):
        if h[isess] != 0:
            logf = np.log(h[isess])
        else:
            logf = 0
        count = count + h[isess]
        estimate = estimate - h[isess] * logf
        sigma = sigma + h[isess] * logf ** 2

    nbias = -(float(ncell) - 1) / (2 * count)
    estimate = estimate / count
    estimate = estimate + np.log(count) + np.log((upperbound - lowerbound) / ncell) - nbias
    estimate = estimate / np.log(np.exp(1))

    return estimate

def a_calculate_powerspectrum(data, sample_rate, cutoff, padfactor=2):

    dim = data.shape
    if len(dim) > 1:
        if dim[1] > dim[0]:
            data = data.T
            nfft = 2 ** ((dim[1] * padfactor).bit_length())
        else:
            nfft = 2 ** ((dim[0] * padfactor).bit_length())
    else:
        nfft = 2 ** ((dim[0] * padfactor).bit_length())

    nfft_half = np.int(nfft / 2)
    freq_hat = fftfreq(nfft) * sample_rate
    freq = freq_hat[0:nfft_half]

    cutoff_freq_index = freq <= cutoff
    idx_cutoff = np.argwhere(cutoff_freq_index)

    sp_hat = fft(data, nfft)
    sp = sp_hat[0:nfft_half] * np.conjugate(sp_hat[0:nfft_half]);

    sp = sp[idx_cutoff]
    freq = freq[idx_cutoff]

    sp_norm = sp / sp.sum()

    return sp_norm, freq

def a_autocovariance_IQR(data):
    
    data = data - np.mean(data)
    dataCov = np.correlate(data, data, "full")
    dataCov = dataCov / max(dataCov)

    q25, q75 = np.percentile(dataCov, [25, 75])
    dataCovIQR = q75 - q25

    return dataCovIQR

def a_mean_cross_rate(data):

    MCR = 0;
    data = data - np.average(data)

    for isess in range(len(data) - 1):
        if np.sign(data[isess]) != np.sign(data[isess + 1]):
            MCR += 1

    MCR = float(MCR) / len(data)

    return MCR

def a_rms_value(data):

    data = data - np.mean(data)
    rms = np.std(data)

    return rms

def a_frequency_domain_features(data, sample_rate, cutoff, cutoff_low=3.5):
    
    # Calculate the power spectrum
    padfactor = 4
    sp_norm, freq = a_calculate_powerspectrum(data, sample_rate, padfactor, cutoff)

    # Find indexes of frequency values below the low cutoff frequency
    idx_low = freq <= cutoff_low
    idx_cutoff_low = np.argwhere(idx_low)

    # Power spectrum and the corresponding frequencies below low cutoff frequency
    sp_norm_low = sp_norm[idx_cutoff_low[:, 0]]
    freq_low = freq[idx_cutoff_low[:, 0]]

    # Calculate max frequency and its magnitude below the cutoff value
    max_freq = freq_low[sp_norm_low.argmax()]
    max_freq_val = sp_norm_low.max().real

    # Calculate dominant frequency ratio
    idx_band = (freq > max_freq - 1) * (freq < max_freq + 1) # Use a +-1 band
    idx_maxfreq_band = np.argwhere(idx_band)
    dom_freq_ratio = sum(sp_norm[idx_maxfreq_band].real)

    # Calculate the spectral entropy
    estimate = 0
    for ifreq in range(len(sp_norm)):
        if sp_norm[ifreq] != 0:
            logps = np.log(sp_norm[ifreq])
        else:
            logps = 0
        estimate = estimate - logps * sp_norm[ifreq].real

    estimate = estimate / np.log(len(sp_norm))
    estimate = (estimate - 0.5) / (1.5 - estimate)
    spec_entropy = float(estimate.real)
    
    return max_freq, max_freq_val, dom_freq_ratio[0], spec_entropy