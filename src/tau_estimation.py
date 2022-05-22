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
from default_cfg import get_cfg_defaults

import src.utils as utils
from utils import raster_full_to_events, calculate_meanfiringrate, exponentialClass,\
    calculate_fanofactor, calculate_coeffvar, spectral_resample,\
    double_exponentialClass, autocorrelation, dichotomizedgaussian_surrogate, \
    leastsquares_fit, leastsquares_fit_doubleexp

from plotting import plot_autocor, plot_utauests, plot_histdata,\
    plot_rasterpsth, plot_spectrogram, plot_psd, plot_psds
import src.plotting as plotting

def testingfunc(foldername, dataset_type, filename):
    #params
    binsize = 0.02#s = 20ms
    delayrange = [1, 20]#units
    samplerate = 10000#samples per second
    # sampletimespan = [0, 1.640]#s
    sampletimespan = [-0.5, 1.640]
    minduration = 1.640
    # sampletimespan *= 10 #100ms time units

    if(dataset_type == 'strf'):
        stimuli_df, spike_df, raster, raster_full = loaddata_withraster_strf(foldername,
                sampletimespan, minduration)#fetch raw data
    else:
        stimuli_df, spike_df, raster, raster_full = loaddata_withraster(foldername, sampletimespan,
                minduration)#fetch raw data

    # measure isi and psth before resampling
    isi_list, isis_list = utils.measure_isi(raster)
    psth_measure = utils.measure_psth(raster_full, binsize, sampletimespan[1]-sampletimespan[0],
            samplerate)
    fanof = calculate_fanofactor(isis_list, raster, samplerate, binsize)

    plot_rasterpsth(raster, psth_measure, isi_list, "../outputs/rasterpsth_{}.pdf".format(filename),
            binsize*1000, sampletimespan, fanof)

def testingfunc_onetrial(foldername, dataset_type, params, filename):
    #params
    binsize = params['binsize']#s = 20ms
    delayrange = params['delayrange']#units
    samplerate = params['samplerate']#samples per second
    # sampletimespan = [0, 1.640]#s
    sampletimespan = params['sampletimespan']
    minduration = params['minduration']
    # sampletimespan *= 10 #100ms time units

    if(dataset_type == 'strf'):
        stimuli_df, spike_df, raster, raster_full, sampletimespan = loaddata_withraster_strf(foldername,
                sampletimespan, minduration)#fetch raw data
    else:
        stimuli_df, spike_df, raster, raster_full, sampletimespan = loaddata_withraster(foldername, sampletimespan,
                minduration)#fetch raw data

    # measure isi and psth before resampling
    isi_list, isis_list = utils.measure_isi(raster)
    psth_measure = utils.measure_psth(raster_full, binsize, sampletimespan[1]-sampletimespan[0],
            samplerate)
    fanof = calculate_fanofactor(isis_list, raster, samplerate, binsize)
    coeffvar = calculate_coeffvar(isis_list)

    # plot_rasterpsth(raster, psth_measure, isi_list, "../outputs/rasterpsth_{}.pdf".format(filename),
            # binsize*1000, sampletimespan, fanof)

    delay = [i for i in range(delayrange[0], delayrange[1])]#range of delays
    mfr = calculate_meanfiringrate(raster, sampletimespan)#mean firing rate
    rv_mean = np.mean(raster_full)
    autocor = autocorrelation(raster_full, delay)#autocorr calculation
    b=(binsize*mfr)**2
    tau, a = leastsquares_fit(np.asarray(autocor), np.asarray(delay)*binsize, b)#least sq fit 
    print("mfr = {}, b = {}, a={}, tau={}".format(mfr, b, a, tau))
    print("coefficient of variance = {}".format(coeffvar))
    ogest = [a, b, tau]

    dichgaussest = None
    figtitle = foldername +" mfr={} ".format(mfr)+" coeffvar={} ".format(coeffvar)+" mean ISI={} "\
        .format(np.mean(isi_list)+" tau est={} ".format(tau))
    return raster, raster_full, isi_list, psth_measure, autocor, mfr, ogest, dichgaussest, figtitle

def estimate_ogtau(cfg, params, foldername):
    #params
    binsize = cfg.DATASET.binsize #sec 
    delayrange = cfg.DATASET.delayrange # bin units
    samplerate = cfg.DATASET.samplerate #samples per second
    # TODO: check this variable's use
    sampletimespan = cfg.DATASET.sampletimespan
    trial_minduration = cfg.DATASET.trial_minduration
    obs_window_range = cfg.DATASET.window_range

    #fetch raw data
    stimuli_df, spike_df, raster, raster_full = loaddata_withraster(cfg, params, foldername)
    # stimuli_spectrogram = get_stimuli_spectrogram(stimuli_df, samplerate, binsize,
                # params['freqbin'], params['freqrange'])

    # measure isi and psth before resampling
    isi_list, _ = utils.measure_isi(raster)
    psth_measure = utils.measure_psth(raster_full, binsize,\
            obs_window_range[1]-obs_window_range[0], samplerate)

    # resampling with wider bins and fitting exponential curve to og data
    raster, raster_full = utils.resample(raster, raster_full, binsize, samplerate) #resize bins
    # stimuli = spectral_resample(stimuli_spectrogram, binsize, samplerate)
    # locspec='../outputs/spectogram.pdf'
    # plot_spectrogram(stimuli, locspec)
    # delay = np.linspace(delayrange[0], delayrange[1], 20)
    delay = [i for i in range(delayrange[0], delayrange[1])] #range of delays
    params['delay_times'] = [d*binsize for d in delay]
    mfr = utils.calculate_meanfiringrate(raster, sampletimespan) #mean firing rate
    rv_mean = np.mean(raster_full)
    autocor = utils.autocorrelation(raster_full, delay) #autocorr calculation
    b=(binsize*mfr)**2
    tau, a = utils.leastsquares_fit(np.asarray(autocor), np.asarray(delay)*binsize, b)#least sq fit 
    print("mfr = {}, b = {}, a={}, tau={}".format(mfr, b, a, tau))
    ogest = [a, b, tau]

    # plot_autocor(np.array(autocor), np.asarray(delay)*binsize, a, b, tau)#plotting autocorr
    # return [a, b, tau, mfr, binsize, delay, binsize, autocor]

    ## create surrogate and estimate unbiased tau
    surrogate_taus = []
    surrogate_as = []
    surr_iters = 400
    for i in range(surr_iters):
        dgauss_surr = dichotomizedgaussian_surrogate(rv_mean, autocor, raster_full, delay)
        _ = dgauss_surr.dichotomized_gauss()
        tau_est, a_est = dgauss_surr.estimate_tau(binsize, samplerate, delayrange, sampletimespan)
        surrogate_taus.append(tau_est)
        surrogate_as.append(a_est)

    # check surrogate a's distribution by plotting it's hist
    # plot_histdata(surrogate_as)

    surrogate_taus = np.array(surrogate_taus)
    params = scipy.stats.lognorm.fit(surrogate_taus)
    bias = params[0]-np.log(tau)
    std_tau = params[1]
    # print(np.exp(params[0]))
    logunbiasedtau = np.log(tau) - bias
    a_est = np.mean(surrogate_as)

    # intergrate the posteriors
    dichgaussest = [a_est, b, np.exp(logunbiasedtau), std_tau]
    print("dich gaus estimates a={}, b={}, tau={}, std={}, bias={}".format(a_est, b,
        np.exp(logunbiasedtau), std_tau, bias))
    figtitle = foldername + " mfr={} ".format(mfr)+" mean ISI={} "\
        .format(np.mean(isi_list))+" tau est={} ".format(tau)

    output = {
            'raster':raster, 
            'raster_full':raster_full, 
            'isi_list':isi_list,
            'psth_measure':psth_measure,
            'delays':np.asarray(delay)*binsize,
            'mean_autocorr':autocor, 
            'mean_fr':mfr, 
            'fit_est':ogest,
            'dichgauss_est':dichgaussest,
            'fig_title':figtitle
        }

    return output

if(__name__=="__main__"):
    # configuration
    cfg = get_cfg_defaults()
    cfg.freeze()

    # extra params if needed
    params = {}

    # prestrf dataset
    # foldername = "../data/prestrf_data/ACx_data_{}/ACx{}/"
    # datafiles = [1,2,3]
    # cortexside = ["Calyx", "Thelo"]
    # dataset_type = 'prestrf'

    # #params
    # params = {}
    # params['binsize'] = 0.02#s = 20ms
    # params['delayrange'] = [0, 50]#units
    # params['sample_rate'] = 10000#samples per second
    # sampletimespan = [0, 1.640]#s
    # params['rng'] = [0, 1.640]
    # params['sampletimespan'] = [100, 300]
    # params['minduration'] = 1.640
    # params['freqrange'] = [0, 45000]
    # params['freqbin'] = 10 #heuristic/random?
    # # sampletimespan *= 10 #100ms time units

    # single datafile test
    trial_neuron = "20200717-xxx999-002-001"
    trial_foldernum = 3
    trial_hemi = "Calyx"
    foldername = cfg.DATASET.foldername.format(trial_foldernum, trial_hemi) +\
            trial_neuron + "/"
    figloc = "../outputs/{}.pdf".format(trial_neuron+"_single_neuron")
    estimate_outputs = estimate_ogtau(cfg, params, foldername)

    # raster, raster_full, isi_list, psth_measure, delay, autocor, mfr, ogest, dichgaussest, figtitle\
            # = estimate_ogtau(foldername, dataset_type, params)
    # plot_autocor(autocor, delay, ogest[0], ogest[1], ogest[2], figloc)
    # plot_summary_single_neuron(autocor, delay, raster, isi_list, psth_measure,\
            # ogest, dichgaussest, foldername, figloc)
    plotting.plot_summary_single_neuron(cfg, params, estimate_outputs, foldername, figloc)

    # dichgaussests = []
    # labels = []
    # mfrs = []
    # foldernames = []
    # cortexsides = []
    # filenames = []
    # for dfs in datafiles:
        # for ctxs in cortexside:
            # fname = foldername.format(dfs, ctxs)
            # foldersinfname = os.listdir(fname)
            # for f in foldersinfname:
                # foldernames.append(fname+f+'/')
                # cortexsides.append(ctxs)
                # filenames.append(f)

    # datafiles = {'folderloc':foldernames, 'label':cortexsides, 'filenames':filenames}

    # for count, dfs in enumerate(datafiles['folderloc']):
        # print("dfs", dfs)
        # print("label: ", datafiles['label'][count])
        # labels.append(datafiles['label'][count])
        # raster, raster_full, isi_list, psth_measure, autocor, mfr, ogest, dichgaussest, figtitle =\
            # estimate_ogtau(dfs, dataset_type, params)
        # # raster, raster_full, isi_list, psth_measure, autocor, mfr, ogest, dichgaussest,\
            # # figtitle = testingfunc_onetrial(dfs, dataset_type, params, datafiles['filenames'][count])
        # mfrs.append(mfr)
        # dichgaussests.append(dichgaussest)
        # figloc = "../outputs/{}.pdf".format(datafiles['filenames'][count])
        # figtitle = foldername
        # plot_neuronsummary(autocor, params, raster, isi_list, psth_measure, ogest, dichgaussest,\
                # figtitle, figloc)




