import scipy
import scipy.linalg
from scipy.stats import norm, multivariate_normal as mnorm
import numpy as np

from src.dich_gauss.optim_dichot_gauss import gauss_cov_func,find_root_bisection
import src.utils as utils

def calculate_dich_gauss_params(target_mfr, target_cov):
    dg_mfr = norm.ppf(target_mfr)
    dg_cov = np.zeros_like(target_cov)
    for i in range(target_cov.shape[0]):
        dg_cov[i] = find_root_bisection([target_mfr]*2, [dg_mfr]*2, target_cov[i])

    return dg_mfr, dg_cov

def generate_spike_trains(mfr, cov, n_neurons, len_spikes):
    cov_f = np.ones(cov.shape[0]+1)
    cov_f[1:] = cov
    cov_full = scipy.linalg.toeplitz(cov_f)
    mnorm_rv_samples = mnorm.rvs(mean = mfr, cov=cov_full, size=(len_spikes)).T
    spikes = np.zeros_like(mnorm_rv_samples)
    spikes[mnorm_rv_samples>0] = 1
    return spikes

def generate_spike_trains_temporal(mfr, cov, n_neurons, n_trials, len_spikes):
    cov_f = np.zeros(len_spikes)
    cov_f[1:1+cov.shape[0]] = cov
    cov_f[0] = 1
    cov_full = scipy.linalg.toeplitz(cov_f)
    print(cov_full)
    mnorm_rv_samples = mnorm.rvs(mean = mfr[0]+np.zeros(cov_full.shape[0]), cov=cov_full,\
            size=(n_trials))
    spikes = np.zeros_like(mnorm_rv_samples)
    print("spike shape", spikes.shape)
    spikes[mnorm_rv_samples>0] = 1
    return spikes

def check_spike_moments(spikes, sampletime, binsize):
    mfr_0 = utils.calculate_meanfiringrate_test2(np.expand_dims(spikes[0], axis=0), sampletime)
    mfr_1 = utils.calculate_meanfiringrate_test2(np.expand_dims(spikes[1], axis=0), sampletime)
    corr = utils.raw_correlation(np.expand_dims(spikes[0], axis=0),\
            np.expand_dims(spikes[1], axis=0), False)
    print("spike mfr: ", mfr_0*binsize, mfr_1*binsize)
    print("spike corr: ", corr[0, corr.shape[1]//2], corr.shape)

def check_spike_moments_temporal(spikes, sampletime, binsize, delays):
    mfr = utils.calculate_meanfiringrate_test2(spikes, sampletime)
    corr = utils.autocorrelation(spikes, delays, False)

    # n = np.arange(1, min(spikes.shape[1], spikes.shape[1])+1, 1)
    # n = np.concatenate((n, n[:-1][::-1]))
    # n = n.reshape( tuple([1] * (spikes.ndim - 1)) + (len(n),))
    # corr = scipy.signal.correlate(spikes, spikes, method='fft')
    # print("corr shape: ", corr.shape, np.argmax(corr, 1))
    # corr = corr/n
    # corr = np.mean(corr[:, corr.shape[1]//2:], 0)

    # print("delay len: ", len(delays))
    print("spike mfr: ", mfr*binsize)
    print("spike corr: ", corr[:10], corr.shape)

def spatial_check():
    mfr = np.array([0.4, 0.3]) # per bin
    cov = np.array([0.1]) # for cov_0,1
    print("actual mfr: ", mfr)
    print("cov: ", cov)

    len_spikes = 1000000 #bins
    binsize = 0.02 #s
    sampletime = [0*binsize, len_spikes*binsize]
    n_neurons = 2

    dg_mfr, dg_cov = calculate_dich_gauss_params(mfr, cov)
    print("dg mean: ", dg_mfr)
    print("dg cov", dg_cov)

    spike_trains = generate_spike_trains(dg_mfr, dg_cov, n_neurons, len_spikes)
    print(spike_trains.shape)
    
    check_spike_moments(spike_trains, sampletime, binsize)

def temporal_check():
    mfr = np.array([0.1481]) # per bin
    time_i = np.array([1,2,3,4,5,6,7])
    pedastal = 0.1481**2 #1e-4
    tau = 0.0387
    binsize = 0.02 #s
    a = 0.317
    auto_cov = (a)*np.exp(-1*(time_i*binsize)/tau) + pedastal
    # auto_cov = np.array([0.01, 0.005, 0.001, 0.0005, 0.0001]) # for cov_0,1
    delay_range = [0, 20]
    delays = [d for d in range(delay_range[0], delay_range[1])]
    print("actual mfr: ", mfr)
    print("auto cov: ", auto_cov)

    len_spikes = 50 #bins
    sampletime = [0*binsize, len_spikes*binsize]
    n_neurons = 1
    n_trials = 1000000

    dg_mfr, dg_cov = calculate_dich_gauss_params(mfr, auto_cov)
    print("dg mean: ", dg_mfr)
    print("dg cov", dg_cov)

    spike_trains = generate_spike_trains_temporal(dg_mfr, dg_cov, n_neurons, n_trials, len_spikes)
    print(spike_trains[0])
    print(spike_trains[1])
    
    check_spike_moments_temporal(spike_trains, sampletime, binsize, delays)

if __name__ == "__main__":
    # spatial_check()
    temporal_check()
