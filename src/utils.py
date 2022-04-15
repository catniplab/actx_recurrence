import numpy as np
import os, pickle
import scipy
from scipy.optimize import curve_fit
from scipy.stats import norm, multivariate_normal
from dich_gauss.dichot_gauss import DichotGauss 
from dich_gauss.optim_dichot_gauss import get_bivargauss_cdf, find_root_bisection
import matplotlib.pyplot as plt

def numpify(tensor):
    return tensor.cpu().detach().numpy()

def spectral_resample(stimuli_spectrogram, time_bin, samplerate):
    time_bin_n = int(time_bin*samplerate) 
    num_bins = stimuli_spectrogram.shape[1]//time_bin_n
    stimuli_resampled = np.zeros((stimuli_spectrogram.shape[0],
        int(stimuli_spectrogram.shape[1]/time_bin_n)))
    for i in range(num_bins): 
        stimuli_resampled[:, i] = np.mean(stimuli_spectrogram[:, i*time_bin_n:(i+1)*time_bin_n], 1)
    return stimuli_resampled

def raster_full_to_events(raster_full, samplerate, sampletimespan):
    raster = []
    for i in range(raster_full.shape[0]):
        rowidx=np.array(np.nonzero(raster_full[i]))
        raster.append((rowidx/raster_full.shape[1])*(sampletimespan[1]-sampletimespan[0])+sampletimespan[0])
    return raster

def calculate_meanfiringrate(raster, sampletime):
    mfs = []
    for i in range(len(raster)):
       mfs.append(len(raster[i])/(sampletime[1]-sampletime[0]))#mfs across time
    return np.mean(mfs)#mean across trials

def calculate_fanofactor(isi, raster, samplerate, binsize):
    ## mean of fano factor on multiple trials
    # fanof = (np.var(data, ddof=1)/np.mean(data))
    fanofs = []
    # print("isi len", len(isi))
    for i in range(len(isi)):
        fanofi = np.var(isi[i],ddof=1)/(np.mean(isi[i])+1e-8)
        if(not np.isnan(fanofi)):
            fanofs.append(fanofi)
    fanof = np.sum(fanofs)/len(isi)
    # fanof = np.mean(fanofs)
    # print(fanofs)
    return fanof 

def calculate_coeffvar(isi):
    coeffvar = []
    for i in range(len(isi)):
        coeffvar.append(np.std(isi)/np.mean(isi))
    return coeffvar

class exponentialClass:
    def __init__(self):
        self.b = 0

    def exponential_func(self, t, tau, a):
        return a * np.exp(-t/tau) + self.b

def measure_isi(raster):
    isi_list = []
    isis_list = []
    for i in range(len(raster)):
        spktimesi = raster[i]
        isii =  np.asarray(spktimesi[1:]) - np.asarray(spktimesi[0:-1])
        isi_list.extend(isii.tolist())
        isis_list.append(isii)
    return isi_list, isis_list

def measure_psth(raster_full, binsizet, period, samplerate):
    binsize = int(samplerate*binsizet)
    totalbins = int((period*samplerate)/binsize)
    print(period, binsize, totalbins, raster_full.shape)
    normspikesperbin = []
    for i in range(totalbins):
        binslice = raster_full[:, i*binsize:(i+1)*binsize]
        # print(binslice.shape)
        spikecount = np.sum(binslice)
        normspikesperbin.append(spikecount/(raster_full.shape[0]*binsize))
    return normspikesperbin

def create_gaussian_filter(cfg, params):
    """
        Creates a test filter 
    """
    # REF: https://mne.tools/dev/auto_tutorials/machine-learning/30_strf.html
    sfreq = params['samplerate']
    n_freqs = params['freq_bins']
    tmin, tmax = params['time_span']

    # To simulate the data we'll create explicit delays here
    delays_samp = np.arange(np.round(tmin * sfreq),
                            np.round(tmax * sfreq)).astype(int)
    delays_sec = delays_samp / sfreq
    freqs = np.linspace(50, 5000, n_freqs)
    grid = np.array(np.meshgrid(delays_sec, freqs))

    # We need data to be shaped as n_epochs, n_features, n_times, so swap axes here
    grid = grid.swapaxes(0, -1).swapaxes(0, 1)

    # Simulate a temporal receptive field with a Gabor filter
    # TODO: add these configurations params in one of ther cfg or params
    means_high = [.1, 1500]
    means_low = [.2, 4000]
    cov = [[.001, 0], [0, 500000]]
    gauss_high = multivariate_normal.pdf(grid, means_high, cov)
    gauss_low = -1 * multivariate_normal.pdf(grid, means_low, cov)
    print("min max of gauss high:", np.min(gauss_high), np.max(gauss_high))
    print("min max of gauss low:", np.min(gauss_low), np.max(gauss_low))

    weights = 10*(gauss_high + gauss_low)  # Combine to create the "true" STRF
    print("min max of gauss total:", np.min(weights), np.max(weights))

    return weights

def calculate_rank(X):
    X_flat = torch.flatten(X, start_dim = 1).T
    X_flat = numpify(X_flat)
    print("x flat shape:", X_flat.shape)
    X_cov = np.cov(X_flat)
    print("x cov shape:", X_cov.shape)
    X_cov_rank = np.linalg.matrix_rank(X_cov)
    print("x cov rank:", X_cov_rank)


def stimuli_nme(params, weights):
    device = params['device']
    strf_bins = params['strf_bins']
    rng = np.random.RandomState(1337)

    ## test_stimuli -- load data from mne dataset
    path_audio = mne.datasets.mtrf.data_path()
    data = loadmat(path_audio + '/speech_data.mat')
    audio = data['spectrogram'].T
    print("OG audio shape:", audio.shape)
    sfreq = float(data['Fs'][0, 0])
    print("sfreq", sfreq)
    n_decim = params['n_decim']
    audio = mne.filter.resample(audio, down=n_decim, npad='auto')
    sfreq /= n_decim

    ## create a delay spectrogram
    # Reshape audio to split into epochs, then make epochs the first dimension.
    n_epochs, n_seconds = 1, 5*16
    audio = audio[:, :int(n_seconds * sfreq * n_epochs)]
    print("audio :", audio.shape)
    X = audio.reshape([n_freqs, n_epochs, -1]).swapaxes(0, 1)
    print("X: ", X.shape)
    X = np.expand_dims(scale(X[0]), 0)
    n_times = X.shape[-1]

    # Delay the spectrogram according to delays so it can be combined w/ the STRF
    # Lags will now be in axis 1, then we reshape to vectorize
    delays = np.arange(np.round(tmin * sfreq),
                       np.round(tmax * sfreq) + 1).astype(int)
    print("delays", len(delays), tmin, tmax)

    # Iterate through indices and append
    X_del = np.zeros((len(delays),) + X.shape)
    for ii, ix_delay in enumerate(delays):
        # These arrays will take/put particular indices in the data
        take = [slice(None)] * X.ndim
        put = [slice(None)] * X.ndim
        if ix_delay > 0:
            take[-1] = slice(None, -ix_delay)
            put[-1] = slice(ix_delay, None)
        elif ix_delay < 0:
            take[-1] = slice(-ix_delay, None)
            put[-1] = slice(None, ix_delay)
        X_del[ii][tuple(put)] = X[tuple(take)]

    # Now set the delayed axis to the 2nd dimension
    X_del = np.rollaxis(X_del, 0, 3)
    print("xdel shape", X_del.shape)
    X_del_return = torch.tensor(X_del, dtype=torch.float32, device=device)[0,:]
    X_del = X_del.reshape([n_epochs, -1, n_times])
    print("xdel shape", X_del.shape)
    n_features = X_del.shape[1]
    weights_sim = weights.ravel()

    # Simulate a neural response to the sound, given this STRF
    y = np.zeros((n_epochs, n_times))
    y_poss = np.zeros((n_epochs, n_times))
    for ii, iep in enumerate(X_del):
        # Simulate this epoch and add random noise
        noise_amp = .002
        y[ii] = np.dot(weights_sim, iep) + noise_amp * rng.randn(n_times)
        # print(y[ii].shape, np.exp(y[ii]).shape) 
        y_poss[ii] = np.random.poisson(lam=np.exp(y[ii]), size=(1, n_times)) 

    print("yii poss", y_poss)
    Y_return = torch.tensor(y_poss, dtype=torch.float32, device=device)
    print("non zero y", np.count_nonzero(y_poss[0]))
    usedidxs = torch.tensor([i for i in range(0, y.shape[1]-strf_bins)],
            device=device)
    return X_del_return, Y_return, usedidxs
