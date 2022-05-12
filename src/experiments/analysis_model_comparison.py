import numpy as np
import os, pickle
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.fft import fft, fftfreq

from src.dich_gauss.dichot_gauss import DichotGauss 
from src.dich_gauss.optim_dichot_gauss import get_bivargauss_cdf, find_root_bisection

# from dataloader import loaddata_withraster, get_stimuli_spectrogram
# from dataloader_strf import loaddata_withraster_strf

from src.dataloaders.dataloader_fmsweep import Data_Loading_FMSweep, loaddata_withraster

from src.utils import raster_full_to_events, calculate_meanfiringrate, exponentialClass,\
    measure_isi, measure_psth, calculate_fanofactor, calculate_coeffvar, spectral_resample,\
    double_exponentialClass, autocorrelation, dichotomizedgaussian_surrogate, \
    resample, leastsquares_fit, leastsquares_fit_doubleexp

from src.plotting import plot_autocor, plot_neuronsummary, plot_utauests, plot_histdata,\
    plot_rasterpsth, plot_spectrogram, plot_psd, plot_psds, plot_neuronsummary_with_doubleexp

def single_vs_double_exponential_model(params, foldername, dataset_type):
    #params
    binsize = params['binsize']#s = 20ms
    delayrange = params['delayrange']#units
    samplerate = params['samplerate']#samples per second
    sampletimespan = params['sampletimespan']
    minduration = params['minduration']

    c_list = params['c_list']
    d_list = params['d_list']

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
    p0 = [0.2, 0.1]
    if(not np.isnan(autocor).any()):
        b=(binsize*mfr)**2
        print("b values -- ", b, mfr, binsize)
        tau, a = leastsquares_fit(np.asarray(autocor), np.asarray(delay)*binsize, b, p0)#least sq fit 
        print("single exponential -- mfr = {}, b = {}, a={}, tau={}".format(mfr, b, a, tau))
        ogest_exp = [a, b, tau]

    p0 = [tau, a]
    # p0 = [1,1]
    # create surrogate and estimate unbiased tau
    surrogate_taus = []
    surrogate_as = []
    surr_iters = 400
    for i in range(surr_iters):
        dgauss_surr = dichotomizedgaussian_surrogate(rv_mean, autocor, raster_full, delay)
        _ = dgauss_surr.dichotomized_gauss()
        tau_est, a_est = dgauss_surr.estimate_tau(binsize, samplerate, delayrange, sampletimespan,
                p0=p0)
        surrogate_taus.append(tau_est)
        surrogate_as.append(a_est)

    surrogate_taus = np.array(surrogate_taus)
    shape, loc, scale = scipy.stats.lognorm.fit(surrogate_taus)
    bias = np.log(scale) - np.log(tau)
    std_tau = shape
    logunbiasedtau = np.log(tau) - bias
    a_est = np.mean(surrogate_as)
    exp_model_mu, exp_model_std = logunbiasedtau, std_tau
    dichgaussest = [a_est, b, np.exp(logunbiasedtau), std_tau]
    llh_exp = scipy.stats.lognorm.pdf(tau, scale=np.exp(logunbiasedtau),
            s=std_tau)

    max_llh = -10000000.0
    best_est = None
    best_logunbiasedtau = None

    # fit tau using two exponentials
    for c_0 in c_list:
        for d_0 in d_list:
            p0 = [ogest_exp[0], ogest_exp[1], c_0, d_0]
            # p0 = [1, 1, 1, 1]
            if(not np.isnan(autocor).any()):
                b=(binsize*mfr)**2
                try:
                    tau, a, c, d = leastsquares_fit_doubleexp(np.asarray(autocor),\
                            np.asarray(delay)*binsize, b, p0) 
                except:
                    print("welp that didn't work -- curve fit")
                    continue

                print("double exponential -- mfr = {}, b = {}, a={}, tau={}, c={}, d={}".\
                        format(mfr, b, a, tau, c, d))
                ogest_doubleexp = [a, b, tau, c, d]
            else:
                continue

            p0 = [tau, a, c, d]
            # p0 = [1, 1, 1, 1]
            # create surrogate and estimate unbiased tau
            surrogate_taus = []
            surrogate_as = []
            surrogate_cs = []
            surrogate_ds = []
            surr_iters = 400
            for i in range(surr_iters):
                dgauss_surr = dichotomizedgaussian_surrogate(rv_mean, autocor, raster_full, delay)
                _ = dgauss_surr.dichotomized_gauss()
                try:
                    tau_est, a_est, c_est, d_est = dgauss_surr.estimate_tau_doubleexp(binsize,\
                            samplerate, delayrange, sampletimespan, p0)
                except:
                    print("welp that didn't work -- dgauss dblexp")
                    continue

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
            c_est = np.mean(surrogate_as)
            shape_d, loc_d, scale_d = scipy.stats.lognorm.fit(surrogate_ds)
            bias_d = np.log(scale_d) - np.log(d)
            logunbiased_d = np.log(d) - bias

            dblexp_model_mu, dblexp_model_std = logunbiasedtau, std_tau
            llh_dblexp = scipy.stats.lognorm.pdf(tau, scale=np.exp(logunbiasedtau),
                    s=std_tau)
            print("llh for dbl exp: ", llh_dblexp)
            dichgaussest_dexp = [a_est, b, np.exp(logunbiasedtau), std_tau, c_est, logunbiased_d]

            if(max_llh<llh_dblexp):
                max_llh = llh_dblexp
                best_est = ogest_doubleexp
                best_dichgauss_dexp = dichgaussest_dexp

    print("best estimates:")
    print("max llh :", max_llh)
    print("best est: ", best_est)
    print("best dich gaus est: ", best_dichgauss_dexp)

    # plot exp vs double exp
    # compute bayes factor on these two fit models
    # bic of exp
    num_params = 2
    num_data = surr_iters # is this correct?
    bic_exp = num_params*np.log(num_data) - 2*np.log(llh_exp)

    # bic for double exp
    num_params = 4
    num_data = surr_iters # is this correct?
    bic_dblexp = num_params*np.log(num_data) - 2*np.log(max_llh)

    return llh_exp, max_llh, bic_exp, bic_dblexp, autocor, delay, raster, isi_list,\
        psth_measure, ogest_exp, dichgaussest, best_est, best_dichgauss_dexp

if(__name__=="__main__"):
    # prestrf dataset
    foldername = "../../data/prestrf_data/ACx_data_{}/ACx{}/"
    datafiles = [1,2,3]
    cortexside = ["Calyx", "Thelo"]
    dataset_type = 'prestrf'

    #params
    params = {}
    params['binsize'] = 0.02#s = 20ms
    params['delayrange'] = [0, 20]#units
    params['samplerate'] = 10000#samples per second
    sampletimespan = [0, 1.640]#s
    params['rng'] = [0, 1.640]
    # params['sampletimespan'] = [0, 1640]
    params['sampletimespan'] = sampletimespan
    params['minduration'] = 1.640
    params['freqrange'] = [0, 45000]
    params['freqbin'] = 10 #heuristic/random?
    # sampletimespan *= 10 #100ms time units

    # parameter initialization for c and d
    params['c_list'] = np.random.uniform(low=-0.5, high=0.5, size=(1,)) # range like for a
    params['d_list'] = np.random.uniform(low=0.005, high=0.3, size=(1,)) # tau value range in ms

    # # single datafile test
    foldername = "../../data/prestrf_data/ACx_data_1/ACxCalyx/20170909-010/"
    figloc = "../outputs/{}.pdf".format("20170909-010")
    dataset_type = 'prestrf'
    figloc = "../../outputs/{}.pdf".format("model_comparison_singleneuron_summary")
    llh_exp, llh_dblexp, bic_exp, bic_dblexp, autocor, delay, raster, isi_list, psth_measure,\
        ogest, dichgaussest, ogest_dexp, dichgaussest_dexp =\
        single_vs_double_exponential_model(params, foldername, dataset_type)
    print(f' BIC for single exp: {bic_exp}; BIC for double exp: {bic_dblexp}; LLH for single exp:\
        {llh_exp}; LLH for double exp: {llh_dblexp}')

    plot_neuronsummary_with_doubleexp(autocor, params, raster, isi_list, psth_measure, ogest,\
            dichgaussest, ogest_dexp, dichgaussest_dexp, foldername, figloc)

    # labels = []
    # foldernames = []
    # cortexsides = []
    # filenames = []
    # llh_exps = []
    # llh_dblexps = []
    # bic_exps = []
    # bic_dblexps = []

    # for dfs in datafiles:
        # for ctxs in cortexside:
            # fname = foldername.format(dfs, ctxs)
            # foldersinfname = os.listdir(fname)
            # for f in foldersinfname:
                # foldernames.append(fname+f+'/')
                # cortexsides.append(ctxs)
                # filenames.append(f)

    # datafiles = {'folderloc':foldernames, 'label':cortexsides, 'filenames':filenames}
    # # print(datafiles)

    # for count, dfs in enumerate(datafiles['folderloc']):
        # print("dfs", dfs)
        # print("label: ", datafiles['label'][count])
        # labels.append(datafiles['label'][count])

        # llh_exp, llh_dblexp, bic_exp, bic_dblexp = single_vs_double_exponential_model(params,\
                # dfs, dataset_type)
        # llh_exps.append(llh_exp)
        # llh_dblexps.append(llh_dblexp)
        # bic_exps(bic_exp)
        # bic_dblexps(bic_dblexp)
        # break

    # #dump data in a pickle file
    # data_dump = {'llh_exp':llh_exps,
            # 'llh_dblexps':llh_dblexps,
            # 'bic_exps':bic_exps,
            # 'bic_dblexps':bic_dblexps}

    # with open('../outputs/analysis_model_comparison.pkl', 'wb') as handle:
        # pickle.dump(data_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('../outputs/analysis_model_comparison.pkl', 'rb') as handle:
        # data_dump = pickle.load(handle)

    # print(f'Exponential model -- mean BIC: {np.mean(bic_exps)}, std BIC: {np.std(bic_exps)}, n =\
            # {len(bic_exps)}')
    # print(f'Double exponential model -- mean BIC: {np.mean(bic_dblexps)}, std BIC: \
            # {np.std(bic_dblexps)}, n = {len(bic_dblexps)}')
