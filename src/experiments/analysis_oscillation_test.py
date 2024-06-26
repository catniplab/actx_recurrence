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
    double_exponentialClass, autocorrelation, resample, leastsquares_fit,\
    calculate_meanfiringrate_test
from src.plotting import plot_autocor, plot_neuronsummary, plot_utauests, plot_histdata,\
    plot_rasterpsth, plot_spectrogram, plot_psd, plot_psds, plot_psds_3d
import src.utils as utils

def oscillation_test(foldername, dataset_type, params):
    #params
    binsize = params['binsize']#s = 20ms
    delayrange = params['delayrange']#units
    samplerate = params['samplerate']#samples per second
    sampletimespan = params['sampletimespan']
    minduration = params['minduration']

    stimuli_df, spike_df, raster, raster_full = loaddata_withraster(foldername, params) #fetch raw data


    # measure isi and psth before resampling
    isi_list, _ = measure_isi(raster)
    psth_measure = measure_psth(raster_full, binsize, sampletimespan[1] - sampletimespan[0],
            samplerate)

    # resampling with wider bins and fitting exponential curve to og data
    raster, raster_full = resample(raster, raster_full, binsize, samplerate)#resize bins
    mfr = calculate_meanfiringrate_test(raster, sampletimespan)#mean firing rate
    # mfr = utils.calculate_meanfiringrate_test2(raster, raster_full, sampletimespan)#mean firing rate

    print("file: ", foldername, "raster size: ", len(raster), "mfr: ", mfr)

    # stimuli = spectral_resample(stimuli_spectrogram, binsize, samplerate)
    # locspec='../outputs/spectogram.pdf'
    # delay = np.linspace(delayrange[0], delayrange[1], 20)
    delay = [i for i in range(delayrange[0], delayrange[1])]#range of delays
    params['delay_times'] = [d*binsize for d in delay]
    rv_mean = np.mean(raster_full)
    autocor = autocorrelation(raster_full, delay)#autocorr calculation

    if(not np.isnan(autocor).any()):
        b=(binsize*mfr)**2
        tau, a = leastsquares_fit(np.asarray(autocor), np.asarray(delay)*binsize, b)#least sq fit 
        # print("mfr = {}, b = {}, a={}, tau={}".format(mfr, b, a, tau))
        ogest = [a, b, tau]

    # --- new code part --- 
    N = params['delayrange'][1]
    # estimate the Power spectrum of each neuron ~ FFT on the autocorrelation would give us this by
    # weiner - khinchin theorem
    psd = fft(autocor)
    # print(psd.shape)
    psd = np.abs(psd[0:N//2])
    psd = 2.0/N * psd #TODO: check this validity; only for printing?
    total_power = np.sum(psd)

    logpsd = 10*np.log10(psd/total_power)
    # print("fft -----", psd)

    # extract the peak oscillation power from the PSD
    fr_idx_log = np.argmax(logpsd[1:])
    fr_idx = np.argmax(psd[1:])
    max_osc_power = psd[fr_idx+1]
    # frequencies = fftfreq(params['delay_times'], N)[:N//2]
    frequencies = fftfreq(N, params['binsize'])[:N//2]
    power_fraction = max_osc_power/total_power

    # print("freq ---", frequencies)
    # print(f' freq index: {fr_idx+1}, freq idx with log psd: {fr_idx_log+1}')
    # print(f' max oscillation power: {max_osc_power}, total signal power: {total_power}, fraction:\
        # {power_fraction}')
    # Find the ratio ~ power of oscillation / power of the signal -- what is mean??
    return psd, total_power, frequencies, power_fraction, logpsd

if(__name__=="__main__"):
    # prestrf dataset
    foldername = "../../data/prestrf_data/ACx_data_{}/ACx{}/"
    filename = "ACx_data_{}/ACx{}/{}"
    datafiles = [1,2,3]
    cortexside = ["Calyx", "Thelo"]
    dataset_type = 'prestrf'

    rows_to_be_dropped = ['ACx_data_1/ACxCalyx/20191010-004',
                         'ACx_data_1/ACxThelo/20180322-f002',
                         'ACx_data_1/ACxThelo/20180329-f003',
                         'ACx_data_2/ACxThelo/20171208-f007',
                         'ACx_data_3/ACxCalyx/20200718-xxx999-002-001',
                         'ACx_data_3/ACxThelo/20180220f007',
                         'ACx_data_3/ACxThelo/20180310f003',
                         'ACx_data_3/ACxThelo/20200213-xxx999-005']
    
    rows_to_use = ['ACx_data_1/ACxCalyx/20080930-002',
                 'ACx_data_1/ACxCalyx/20081104-002',
                 'ACx_data_1/ACxCalyx/20100428-001',
                 'ACx_data_1/ACxCalyx/20170903-001-017',
                 'ACx_data_1/ACxCalyx/20170903-001-019',
                 'ACx_data_1/ACxCalyx/20170909-007_file001',
                 'ACx_data_1/ACxCalyx/20170909-010',
                 'ACx_data_1/ACxCalyx/20190904-001-016',
                 'ACx_data_1/ACxCalyx/20191010-005',
                 'ACx_data_1/ACxCalyx/20200107-009-003',
                 'ACx_data_1/ACxCalyx/20200108-008-002',
                 'ACx_data_1/ACxThelo/20171114-d003-f004',
                 'ACx_data_1/ACxThelo/20171204-f010',
                 'ACx_data_1/ACxThelo/20171205-f002',
                 'ACx_data_1/ACxThelo/20171205-f007',
                 'ACx_data_1/ACxThelo/20171208-f015',
                 'ACx_data_1/ACxThelo/20171208-f016',
                 'ACx_data_1/ACxThelo/20180219-f014',
                 'ACx_data_1/ACxThelo/20180219-f017',
                 'ACx_data_1/ACxThelo/20180309-f005',
                 'ACx_data_1/ACxThelo/20180322-f006',
                 'ACx_data_2/ACxCalyx/20190918-xxx999-001-004',
                 'ACx_data_2/ACxThelo/20171120-f003',
                 'ACx_data_2/ACxThelo/20171208-f011',
                 'ACx_data_2/ACxThelo/20171208-f018',
                 'ACx_data_2/ACxThelo/20200116-xxx999-005',
                 'ACx_data_3/ACxCalyx/20200717-xxx999-002-001',
                 'ACx_data_3/ACxCalyx/20200718-xxx999-001-001',
                 'ACx_data_3/ACxCalyx/20200718-xxx999-001-002',
                 'ACx_data_3/ACxCalyx/20200720-xxx999-001-001',
                 'ACx_data_3/ACxCalyx/20200720-xxx999-001-002',
                 'ACx_data_3/ACxCalyx/20200720-xxx999-005-003',
                 'ACx_data_3/ACxCalyx/20200721-xxx999-001-001',
                 'ACx_data_3/ACxCalyx/20200721-xxx999-001-002',
                 'ACx_data_3/ACxCalyx/20200721-xxx999-002-001',
                 'ACx_data_3/ACxCalyx/20200721-xxx999-002-002',
                 'ACx_data_3/ACxThelo/20171205f009',
                 'ACx_data_3/ACxThelo/20180219f012',
                 'ACx_data_3/ACxThelo/20180219f016',
                 'ACx_data_3/ACxThelo/20180301f006',
                 'ACx_data_3/ACxThelo/20180322f012',
                 'ACx_data_3/ACxThelo/20180329f005',
                 'ACx_data_3/ACxThelo/20200115-xxx999-004-001',
                 'ACx_data_3/ACxThelo/20200708-xxx999-003-003',
                 'ACx_data_3/ACxThelo/20200710-xxx999-004-002']

    #params
    params = {}
    params['binsize'] = 0.02#s = 20ms
    params['delayrange'] = [0, 50]#units
    params['samplerate'] = 10000#samples per second
    sampletimespan = [0, 1.640]#s
    params['rng'] = [0, 1.640]
    params['sampletimespan'] = [0, 1.640]
    params['minduration'] = 1.640
    params['freqrange'] = [0, 45000]
    params['freqbin'] = 10 #heuristic/random?
    # sampletimespan *= 10 #100ms time units

    # single datafile test
    # foldername = "../../data/prestrf_data/ACx_data_1/ACxCalyx/20170903-001-017/"
    # figloc = "../../outputs/{}.pdf".format("20170903-001-017")
    # dataset_type = 'prestrf'
    # figloc = "../../outputs/{}.pdf".format("psd_possible_culprit")
    # psd, total_power, freqs, frac, logpsd = oscillation_test(foldername, dataset_type, params)
    # plot_psd(logpsd, freqs, figloc)

    labels = []
    names = []
    foldernames = []
    cortexsides = []
    filenames = []
    psds = []
    logpsds = []
    total_powers = []
    frequencies = []
    power_frac = []
    for dfs in datafiles:
        for ctxs in cortexside:
            fname = foldername.format(dfs, ctxs)
            foldersinfname = os.listdir(fname)
            for f in foldersinfname:
                filename_to_check = filename.format(dfs, ctxs, f) 
                # print(filename_to_check)
                if(filename_to_check not in rows_to_use):
                    continue
                foldernames.append(fname+f+'/')
                cortexsides.append(ctxs)
                filenames.append(f)

    datafiles = {'folderloc':foldernames, 'label':cortexsides, 'filenames':filenames}
    print("total files: ", len(datafiles['folderloc']))

    for count, dfs in enumerate(datafiles['folderloc']):
        # print("dfs", dfs)
        psd, total_power, freqs, frac, logpsd = oscillation_test(dfs, dataset_type, params)
        if(np.isnan(np.asarray(psd)).any()):
            print("skipped", psd)
            continue
        psds.append(psd)
        logpsds.append(logpsd)
        frequencies.append(freqs)
        total_powers.append(total_power)
        power_frac.append(frac)
        labels.append(datafiles['label'][count])
        names.append(dfs)
        # print("label: ", datafiles['label'][count])

    #dump data in a pickle file
    data_dump = {'psds':psds,
            'frequencies':frequencies,
            'total_powers':total_powers,
            'power_frac':power_frac,
            'labels':labels,
            'logpsd':logpsds,
            'foldername':names}

    with open('../../outputs/psds_plot_pickle.pkl', 'wb') as handle:
        pickle.dump(data_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../../outputs/psds_plot_pickle.pkl', 'rb') as handle:
        data_dump = pickle.load(handle)

    n = len(data_dump['power_frac'])
    mean_power_frac = np.mean(data_dump['power_frac'])
    std_power_frac = np.std(data_dump['power_frac'])
    # print(f'mean power fraction = {mean_power_frac}, std = {std_power_frac}, n = {n}')

    # finding the high freq component:
    # high_idx = np.argmax(np.asarray(data_dump['logpsd'])[:, 4:], axis=1)
    # print(high_idx, data_dump['foldername'])
    # print("name: ", data_dump['foldername'][high_idx],\
            # " labels: ", data_dump['labels'][high_idx])

    figloc = "../../outputs/{}.pdf".format("neuron_summary_psds")
    plot_psds(data_dump['logpsd'], data_dump['frequencies'], data_dump['labels'], params, data_dump, figloc)




