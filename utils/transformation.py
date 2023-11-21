import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import torch, random
from utils.utils import normalize_arr_2d, robust_scaling_2d
import time


def gen_jitering(ts, window_size):
    num_samples = len(ts)
    noise_amplitude = (ts.max()-ts.min())*1/5  # Amplitude of the noise signal
    # Combine the original time series with the high frequency noise
    combined_signal = ts + noise_amplitude * np.random.randn(num_samples)
    # Random length of the frequency-based anomalies
    anom_len = np.random.randint(window_size//20, window_size//3)
    # Randomly select the range of indices for the part to modify
    start_index = np.random.randint(0, len(ts)-anom_len)
    end_index = start_index+anom_len
    # Revise portion of the original time series
    modified_ts = ts.copy()
    modified_ts[start_index:end_index] = combined_signal[start_index:end_index]

    return modified_ts, (start_index,end_index)


def gen_warping(ts, fft_values, window_size, verbose = False):
    # Compute the power spectral density
    psd_values = np.abs(fft_values) ** 2
    # Find the peak 30 frequencies
    peak_indices = np.argsort(psd_values)[-30:]

    frequencies = np.fft.fftfreq(len(ts), d=1)
    frequencies = frequencies[peak_indices]
    # Get the positive frequencies between (0,1)
    frequencies = np.unique(frequencies[frequencies>0])
    frequencies = np.sort(frequencies) # frequency sorted from lowest to highest

    # Randomly pick frequencies from the lower frequency range  
    pick_idx = np.arange(0, len(frequencies), 3)
    cutoff = np.random.choice(frequencies[pick_idx][0:4], size=2, replace=False)
    # Randomly select the frequency range you want to enhance
    low_freq = min(cutoff)  
    high_freq = max(cutoff)  
    b, a = signal.butter(4, [low_freq, high_freq], btype='band')
    # Apply the filter to the time series
    filtered_signal_lower = signal.lfilter(b, a, ts)

    # Scale the filtered signal to the orignal time series
    original = ts.reshape(-1, 1)
    filtered_signal_lower = filtered_signal_lower.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(original.min(), original.max()))
    scaler.fit(original)
    filtered_signal_lower = scaler.transform(filtered_signal_lower).flatten()

    # Random length of the frequency-based anomalies
    anom_len = np.random.randint(window_size//20, window_size//3)
    # Randomly select the range of indices for the part to modify
    start_index = np.random.randint(0, len(ts)-anom_len)
    end_index = start_index+anom_len

    # Copy the original array
    modified_ts = ts.copy()
    modified_ts[start_index:end_index] = filtered_signal_lower[start_index:end_index]
    if verbose:
        print(f'Lower frequency transformation: filter between {low_freq} and {high_freq}, anomaly length: {anom_len}')
    
    return modified_ts, (start_index,end_index)

def get_cross_domain_features(ts_slices, period_len, window_size):
    random.seed()
    tran_ts = []
    org_fft = []
    tran_fft = []
    org_resid = []
    tran_resid = []
    anom_indx = []
    h_freq_num = 0
    jittering = True
    for slice in ts_slices:
        # 1. Get the residuals - original
        org_result = seasonal_decompose(slice, model='additive', period=period_len, extrapolate_trend='freq')
        org_resid.append(org_result.resid)
        # 2. Get the frequency domain features - original
        fft_values = np.fft.fft(slice)
        amplitudes = np.abs(fft_values)  # Magnitude spectrum
        phases = np.angle(fft_values)  # Phase spectrum
        power = np.abs(fft_values) ** 2
        cat_fft = np.vstack((amplitudes, phases, power))
        cat_fft = robust_scaling_2d(cat_fft)  # 3*each row
        org_fft.append(cat_fft)
        
        #--------------------------------------------------
        # Transform the input time series (jittering and smoothing take turns)
        if jittering:
            modified, anom = gen_jitering(slice, window_size)
            h_freq_num += 1
            jittering = False
        else:
            modified, anom = gen_warping(slice, fft_values, window_size, verbose=False)
            jittering = True

        tran_ts.append(modified)
        anom_indx.append(anom)
        # 1. Get the residuals - transformed
        tran_result = seasonal_decompose(modified, model='additive', period=period_len,  extrapolate_trend='freq')
        tran_resid.append(tran_result.resid)

        # 2. Get the fft features - transformed
        fft_values = np.fft.fft(modified)
        amplitudes = np.abs(fft_values)  # Magnitude spectrum
        phases = np.angle(fft_values)  # Phase spectrum
        power = np.abs(fft_values) ** 2
        cat_fft = np.vstack((amplitudes, phases, power))
        cat_fft = robust_scaling_2d(cat_fft)  # 3*each row
        tran_fft.append(cat_fft)        

    slices = robust_scaling_2d(ts_slices)
    org_resid = robust_scaling_2d(org_resid)
    tran_ts = robust_scaling_2d(tran_ts)
    tran_resid = robust_scaling_2d(tran_resid)

    org_ts = torch.Tensor(slices).unsqueeze(dim=-1) # B*T*1
    org_fft = torch.Tensor(np.array(org_fft)).permute(0,2,1) # B*T*3
    org_resid = torch.Tensor(np.array(org_resid)).unsqueeze(dim=-1) # B*T*1

    tran_ts = torch.Tensor(np.array(tran_ts)).unsqueeze(dim=-1)
    tran_fft = torch.Tensor(np.array(tran_fft)).permute(0,2,1) 
    tran_resid = torch.Tensor(np.array(tran_resid)).unsqueeze(dim=-1)
    features = [org_ts, tran_ts, org_fft, tran_fft, org_resid, tran_resid]
    return features, h_freq_num, anom_indx


def get_test_features(ts_slices, period_len):
    org_fft = []
    org_resid = []
    
    for slice in ts_slices:
        # 1. Get the residuals - original
        org_result = seasonal_decompose(slice, model='additive', period=period_len, extrapolate_trend='freq')
        org_resid.append(org_result.resid)
        # 2. Get the fft features - original
        fft_values = np.fft.fft(slice)
        amplitudes = np.abs(fft_values)  # Magnitude spectrum
        phases = np.angle(fft_values)  # Phase spectrum
        power = np.abs(fft_values) ** 2
        cat_fft = np.vstack((amplitudes, phases, power))
        cat_fft = robust_scaling_2d(cat_fft)  # 3*each row
        org_fft.append(cat_fft)

    slices = robust_scaling_2d(ts_slices)
    org_resid = robust_scaling_2d(org_resid)

    org_ts = torch.Tensor(slices).unsqueeze(dim=-1)
    org_fft = torch.Tensor(np.array(org_fft)).permute(0,2,1) 
    org_resid = torch.Tensor(np.array(org_resid)).unsqueeze(dim=-1)

    features = [org_ts, org_fft, org_resid]
    return features 