import numpy as np
import matplotlib.pyplot as plt
import os
import scipy

from dataloader import load_data, get_eventraster, get_stimulifreq_barebone
from dataloader_strf import loaddata_withraster_strf
from utils import raster_fulltoevents, exponentialClass, spectral_resample
from plotting import plot_autocor, plot_neuronsummary, plot_utauests, plot_histdata,\
    plot_rasterpsth, plot_spectrogram

class strfestimation():
    def __init__(self, params):
        self.params = params
        self.frequency_bins =\
            int((params['freqrange'][1]-params['freqrange'][0])/params['freqbinsize'])
        self.time_bins =\
            int((params['strf_timerange'][1]-params['strf_timerange'][0])/params['strf_timebinsize'])
        self.strf_params = np.random.rand((self.frequency_bins, self.time_bins))
        self.hist_bins = int(params['hist_size']/params['strf_timebinsize'])
        self.history_filter = np.random.rand((self.hist_bins, 1))
    
    def binned_spikes(self, params, spiketimes):
        total_time = np.ceil(spiketimes[-1]+1) #s
        binned_spikes = np.zeros((1, int(total_time / params['strf_timebinsize'])))
        for count, spiket in enumerate(spiketimes):
            binned_spikes[0, int(spiket/params['strf_timebinsize'])]+=1
        return binned_spikes
    
    def stimuli_baretospectrogram(self, spikebin, binned_freq, binned_amp):
        num_timebins =\
            int((params['strf_timerange'][1]-params['strf_timerange'][0])/params['strf_timebinsize'])
        num_freqbins = int((params['freqrange'][1]-params['freqrange'][0])/params['freqbinsize'])
        stimuli_spectrogram = np.zeros((num_freqbins, num_timebins))
        timeslice = range(spikebin-1-num_timebins, spikebin-1)

        for idx, ts in enumerate(timeslice):
           stimuli_spectrogram[binned_freq[0, ts]//params['freqbinsize'], idx] = binned_amp[0, ts]
        return stimuli_spectrogram

    def run(self, stimuli_df, spike_df):
        spiketimes = spike_df['timestamps'].to_numpy() # in seconds
        total_time = np.ceil(spiketimes[-1]+1) #s
        samples_per_bin = int(params['samplerate']*params['strf_timebinsize']) #num samples per bin
        spikes_binned = self.binned_spikes(params, spiketimes)
        binned_freq, binned_amp = get_stimulifreq_barebone(stimuli_df, total_time, timebin, params['samplerate'])

        # for count, spikebin in enumerate(spikes_binned):
            # stimuli_spectro = self.stimuli_baretospectrogram(count, binned_freq, binned_amp)
            # spike_history = spiked_binned[0, count-params['hist_size']:(count-1)]

        print("to complete")

def estimate_strf(foldername, dataset_type, params):
    #params
    binsize = params['binsize']#s = 20ms
    delayrange = params['delayrange']#units
    samplerate = params['samplerate']#samples per second
    sampletimespan = params['sampletimespan']
    minduration = params['minduration']

    if(dataset_type == 'strf'):
        stimuli_df, spike_df= loaddata_withraster_strf(foldername)#fetch raw data
    else:
        stimuli_df, spike_df, raster, raster_full = loaddata_withraster(foldername, sampletimespan,
                minduration)#fetch raw data

    strfest = strfestimation(params)
    strfest.run(stimuli_df, raster_df)


if(__name__=="__main__"):
    # prestrf dataset
    foldername = "../data/prestrf_data/ACx_data_{}/ACx{}/"
    datafiles = [1,2,3]
    cortexside = ["Calyx", "Thelo"]
    dataset_type = 'prestrf'
    #params
    params = {}
    params['binsize'] = 0.02#s = 20ms
    params['strf_timebinsize'] = 0.001#s = 1ms
    params['strf_timerange'] = [0, 0.25] #s - 0 to 250ms
    params['delayrange'] = [1, 30]#units
    params['samplerate'] = 10000#samples per second
    params['sampletimespan'] = [0, 100] #sec
    params['minduration'] = 1.640
    params['freqrange'] = [0, 41000]
    params['freqbinsize'] = 10 #hz/bin -- heuristic/random?
    params['hist_size'] = 0.02 #s = 20ms

    # #strf dataset
    # foldername = "../data/strf_data/"
    # cortexside = ["Calyx", "Thelo"]
    # dataset_type = 'strf'

    # #params
    # params = {}
    # params['binsize'] = 0.02#s = 20ms
    # params['delayrange'] = [1, 300]#units
    # params['samplerate'] = 10000#samples per second
    # # sampletimespan = [0, 1.640]#s
    # params['sampletimespan'] = [100, 300]
    # params['minduration'] = 1.640
    # # sampletimespan *= 10 #100ms time units

    ## single datafile test
    # foldername = "../data/prestrf_data/ACx_data_1/ACxCalyx/20170909-010/"
    # figloc = "../outputs/{}.png".format("20170909-010")
    # dataset_type = 'prestrf'
    # foldername = "../data/strf_data/20210825-xxx999-002-001/"
    # dataset_type = 'strf'
    # figloc = "../outputs/{}.png".format("20210825-xxx999-002-001")
    # raster, raster_full, isi_list, psth_measure, delay, autocor, mfr, ogest, dichgaussest =\
        # estimate_ogtau(foldername, dataset_type)
    # plot_neuronsummary(autocor, delay, raster, isi_list, psth_measure, ogest, dichgaussest,\
            # foldername, figloc)

    labels = []

    foldernames = []
    cortexsides = []
    filenames = []
    if(dataset_type=='prestrf'):
        for dfs in datafiles:
            for ctxs in cortexside:
                fname = foldername.format(dfs, ctxs)
                foldersinfname = os.listdir(fname)
                for f in foldersinfname:
                    foldernames.append(fname+f+'/')
                    cortexsides.append(ctxs)
                    filenames.append(f)
    else:
        foldersinfname = os.listdir(foldername)
        for f in foldersinfname:
            foldernames.append(foldername+f+'/')
            cortexsides.append('NA')
            filenames.append(f)

    datafiles = {'folderloc':foldernames, 'label':cortexsides, 'filenames':filenames}
    # print(datafiles)

    for count, dfs in enumerate(datafiles['folderloc']):
        figloc = "../outputs/strf_{}.pdf".format(datafiles['filenames'][count])
        print("dfs", dfs)
        labels.append(datafiles['label'][count])
        # figtitle = foldername
        estimate_strf()
        print("label: ", datafiles['label'][count])
    
