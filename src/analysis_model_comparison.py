import numpy as np
import os, pickle
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.fft import fft, fftfreq

from dich_gauss.dichot_gauss import DichotGauss 
from dich_gauss.optim_dichot_gauss import get_bivargauss_cdf, find_root_bisection

# from dataloader import loaddata_withraster, get_stimuli_spectrogram
# from dataloader_strf import loaddata_withraster_strf

from dataloaders.dataloader_fmsweep import Data_Loading_FMSweep, loaddata_withraster

from utils import raster_full_to_events, calculate_meanfiringrate, exponentialClass,\
    measure_isi, measure_psth, calculate_fanofactor, calculate_coeffvar, spectral_resample,\
    double_exponentialClass, autocorrelation, dichotomizedgaussian_surrogate, \
    resample, leastsquares_fit, leastsquares_fit_doubleexp

from plotting import plot_autocor, plot_neuronsummary, plot_utauests, plot_histdata,\
    plot_rasterpsth, plot_spectrogram, plot_psd, plot_psds

def single_vs_double_exponential_model(params, foldername, dataset_type):
    #params
    binsize = params['binsize']#s = 20ms
    delayrange = params['delayrange']#units
    samplerate = params['sample_rate']#samples per second
    sampletimespan = params['sampletimespan']
    minduration = params['minduration']

    #fetch raw data
    stimuli_df, spike_df, raster, raster_full = loaddata_withraster(foldername, params) 

    # measure isi and psth before resampling
    isi_list, _ = measure_isi(raster)
    psth_measure = measure_psth(raster_full, binsize, sampletimespan[1] - sampletimespan[0],
            samplerate)

    # resampling with wider bins and fitting exponential curve to og data
    raster, raster_full = resample(raster, raster_full, binsize, samplerate)#resize bins
    # stimuli = spectral_resample(stimuli_spectrogram, binsize, samplerate)
    # locspec='../outputs/spectogram.pdf'
    delay = [i for i in range(delayrange[0], delayrange[1])]#range of delays
    params['delay_times'] = [d*binsize for d in delay]
    mfr = calculate_meanfiringrate(raster, sampletimespan)#mean firing rate
    rv_mean = np.mean(raster_full)
    autocor = autocorrelation(raster_full, delay)#autocorr calculation

    # fit tau using a single exponential 
    if(not np.isnan(autocor).any()):
        b=(binsize*mfr)**2
        tau, a = leastsquares_fit(np.asarray(autocor), np.asarray(delay)*binsize, b)#least sq fit 
        print("mfr = {}, b = {}, a={}, tau={}".format(mfr, b, a, tau))
        ogest_exp = [a, b, tau]

    # create surrogate and estimate unbiased tau
    surrogate_taus = []
    surrogate_as = []
    surr_iters = 400
    for i in range(surr_iters):
        dgauss_surr = dichotomizedgaussian_surrogate(rv_mean, autocor, raster_full, delay)
        _ = dgauss_surr.dichotomized_gauss()
        tau_est, a_est = dgauss_surr.estimate_tau(binsize, samplerate, delayrange, sampletimespan)
        surrogate_taus.append(tau_est)
        surrogate_as.append(a_est)

    surrogate_taus = np.array(surrogate_taus)
    shape, loc, scale = scipy.stats.lognorm.fit(surrogate_taus)
    bias = np.log(scale) - np.log(tau)
    std_tau = shape
    logunbiasedtau = np.log(tau) - bias
    a_est = np.mean(surrogate_as)
    exp_model_mu, exp_model_std = logunbiasedtau, std_tau
    llh_exp = scipy.stats.lognorm.pdf(logunbiasedtau, loc=0, scale=np.exp(logunbiasedtau),
            shape=std_tau)

    # fit tau using two exponentials
    if(not np.isnan(autocor).any()):
        b=(binsize*mfr)**2
        tau, a, c, d = leastsquares_fit_doubleexp(np.asarray(autocor), np.asarray(delay)*binsize, b) 
        print("mfr = {}, b = {}, a={}, tau={}, c={}, d={}".format(mfr, b, a, tau, c, d))
        ogest_doubleexp = [a, b, tau, c, d]

    # create surrogate and estimate unbiased tau
    surrogate_taus = []
    surrogate_as = []
    surrogate_cs = []
    surrogate_ds = []
    surr_iters = 400
    for i in range(surr_iters):
        dgauss_surr = dichotomizedgaussian_surrogate(rv_mean, autocor, raster_full, delay)
        _ = dgauss_surr.dichotomized_gauss()
        tau_est, a_est, c_est, d_est = dgauss_surr.estimate_tau_doubleexp(binsize, samplerate,\
                delayrange, sampletimespan)
        surrogate_taus.append(tau_est)
        surrogate_as.append(a_est)
        surrogate_cs.append(c_est)
        surrogate_ds.append(d_est)

    surrogate_taus = np.array(surrogate_taus)
    shape, loc, scale = scipy.stats.lognorm.fit(surrogate_taus)
    bias = np.log(scale) - np.log(tau)
    std_tau = shape
    logunbiasedtau = np.log(tau) - bias
    a_est = np.mean(surrogate_as)
    dblexp_model_mu, dblexp_model_std = logunbiasedtau, std_tau
    llh_dblexp = scipy.stats.lognorm.pdf(logunbiasedtau, loc=0, scale=np.exp(logunbiasedtau),
            shape=std_tau)

    # plot exp vs double exp
    # compute bayes factor on these two fit models
    # bic of exp
    num_params = 2
    num_data = surr_items # is this correct?
    bic_exp = num_params*np.log(num_data) - 2*np.log(llh_exp)

    # bic for double exp
    num_params = 4
    num_data = surr_items # is this correct?
    bic_dblexp = num_params*np.log(num_data) - 2*np.log(llh_dblexp)

    return llh_exp, llh_dblexp, bic_exp, bic_dblexp

if(__name__=="__main__"):
    # prestrf dataset
    foldername = "../data/prestrf_data/ACx_data_{}/ACx{}/"
    datafiles = [1,2,3]
    cortexside = ["Calyx", "Thelo"]
    dataset_type = 'prestrf'

    #params
    params = {}
    params['binsize'] = 0.02#s = 20ms
    params['delayrange'] = [0, 50]#units
    params['sample_rate'] = 10000#samples per second
    sampletimespan = [0, 1.640]#s
    params['rng'] = [0, 1.640]
    params['sampletimespan'] = [100, 300]
    params['minduration'] = 1.640
    params['freqrange'] = [0, 45000]
    params['freqbin'] = 10 #heuristic/random?
    # sampletimespan *= 10 #100ms time units

    # single datafile test
    # foldername = "../data/prestrf_data/ACx_data_1/ACxCalyx/20170909-010/"
    # figloc = "../outputs/{}.pdf".format("20170909-010")
    # dataset_type = 'prestrf'
    # foldername = "../data/strf_data/20210825-xxx999-002-001/"
    # dataset_type = 'strf'
    # figloc = "../outputs/{}.pdf".format("oscillation_singleneuron_test_neuronsummary")
    # raster, raster_full, isi_list, psth_measure, delay, autocor, mfr, ogest, dichgaussest =\
        # estimate_ogtau(foldername, dataset_type, params)
    # plot_neuronsummary(autocor, delay, raster, isi_list, psth_measure, ogest, dichgaussest,\
            # foldername, figloc)

    labels = []
    foldernames = []
    cortexsides = []
    filenames = []
    llh_exps = []
    llh_dblexps = []
    bic_exps = []
    bic_dblexps = []

    for dfs in datafiles:
        for ctxs in cortexside:
            fname = foldername.format(dfs, ctxs)
            foldersinfname = os.listdir(fname)
            for f in foldersinfname:
                foldernames.append(fname+f+'/')
                cortexsides.append(ctxs)
                filenames.append(f)

    datafiles = {'folderloc':foldernames, 'label':cortexsides, 'filenames':filenames}
    # print(datafiles)

    for count, dfs in enumerate(datafiles['folderloc']):
        print("dfs", dfs)
        print("label: ", datafiles['label'][count])
        labels.append(datafiles['label'][count])

        llh_exp, llh_dblexp, bic_exp, bic_dblexp = single_vs_double_exponential_model(params,\
                dfs, dataset_type)
        llh_exps.append(llh_exp)
        llh_dblexps.append(llh_dblexp)
        bic_exps(bic_exp)
        bic_dblexps(bic_dblexp)
        break

    #dump data in a pickle file
    data_dump = {'llh_exp':llh_exps,
            'llh_dblexps':llh_dblexps,
            'bic_exps':bic_exps,
            'bic_dblexps':bic_dblexps}

    with open('../outputs/analysis_model_comparison.pkl', 'wb') as handle:
        pickle.dump(data_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../outputs/analysis_model_comparison.pkl', 'rb') as handle:
        data_dump = pickle.load(handle)

    print(f'mean BIC for exponential model {np.mean(bic_exps)}, mean BIC for double exponential\
        models {np.mean(bic_dblexps)}')
