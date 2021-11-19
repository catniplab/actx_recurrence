import numpy as np
import matplotlib.pyplot as plt
import os
import scipy

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.utils.data import Dataset, DataLoader

from dataloader import load_data, get_eventraster, get_stimulifreq_barebone
from dataloader_strf import loaddata_withraster_strf
from utils import raster_fulltoevents, exponentialClass, spectral_resample
from plotting import plot_autocor, plot_neuronsummary, plot_utauests, plot_histdata,\
    plot_rasterpsth, plot_spectrogram, plot_strf

class strfdataset(Dataset):
    def __init__(self, params, stimuli_df, spikes_df):
        self.params = params
        self.device = params['device']
        self.spikes_df = spikes_df
        self.stimuli_df = stimuli_df
        spiketimes = spike_df['timestamps'].to_numpy() # in seconds
        total_time = np.ceil(spiketimes[-1]+1) #s
        samples_per_bin = int(params['samplerate']*params['strf_timebinsize']) #num samples per bin
        self.spikes_binned = torch.tensor(self.binned_spikes(params, spiketimes),
                device=self.device)
        self.binned_freq, self.binned_amp = get_stimulifreq_barebone(stimuli_df, total_time, timebin,
                params['samplerate'], params['freqrange'])
        self.binned_freq = torch.tensor(self.binned_freq, device=self.device)
        self.binned_amp = torch.tensor(self.binned_amp, device=self.device)
        self.spiking_bins = torch.nonzero(self.spikes_binned)

    def __len__(self):
        return self.spiking_bins.shape[0]

    def __getitem__(self, idx):
        stimuli_spectro = self.stimuli_baretospectrogram(self.spiking_bins[idx], self.binned_freq, self.binned_amp)
        spike_history = self.spikes_binned[0, self.spiking_bins[idx] -
                params['hist_size']:(self.spiking_bins[idx]-1)]
        stimuli_spectro = torch.tensor(stimuli_spectro, device=self.device)
        spike_history = torch.tensor(spike_history, device=self.device)
        eta = torch.tensor(np.random.normal(0, 1, 1), device=self.device)
        return stimuli_spectro, spike_history, eta, self.spikes_binned[0, self.spiking_bins[idx]]

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

class strfestimation(nn.Module):
    def __init__(self, params):
        self.params = params
        self.device = params['device']
        self.frequency_bins =\
            int((params['freqrange'][1]-params['freqrange'][0])/params['freqbinsize'])
        self.time_bins =\
            int((params['strf_timerange'][1]-params['strf_timerange'][0])/params['strf_timebinsize'])
        self.strf_params = torch.tensor(np.random.rand((self.frequency_bins,
            self.time_bins)), requires_grad=True, device = self.device)
        self.hist_bins = int(params['hist_size']/params['strf_timebinsize'])
        self.history_filter = torch.tensor(np.random.rand((self.hist_bins, 1)),
                requires_grad=True, device=self.device)
        self.bias = torch.randn(1, requires_grad=True, device=self.device)
        self.optimizer = torch.optim.SGD([self.strf_params, self.history_filter, self.bias],
                lr=params['lr'])

    def run(self, dataloader):
        epochs = params['epochs']

        for e in epochs:
            for ibatch, batchsample in enumerate(tqdm(dataloader)):
                self.optimizer.zero_grad()
                Xin, Yhist, eta, Yt = batchsample
                linsum = torch.mul(Xin, self.strf_params[None, :,:]) + torch.mul(Yhist,
                        self.history_filter[None, :]) + self.bias[None, :] + eta
                linsumexp = torch.exp(linsum)
                LLh =  Yt*linsum - linsumexp - (Yt+1).lgamma().exp()
                loss = -1*LLh
                loss.backward()
            print("loss at epoch {} = {}".format(e, loss))

    def plotstrf(self, figloc):
        plot_strf(self.strf_params.detach().cpu().numpy(),
                self.history_filter.detach().cpu().numpy(), figloc)


def estimate_strf(foldername, dataset_type, params,  figloc):
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

    strf_dataset = strfdataset(params, stimuli_df, spike_df)
    strf_dataloader = DataLoader(strfdataset, batch_size=params['batchsize'], shuffle=False,
            num_workers=10)
    strfest = strfestimation(params)
    strfest.run(strf_dataloader)
    strfest.plotstrf(figloc)


if(__name__=="__main__"):
    # torch params
    device = torch.device('cuda:0')

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

    params['device'] = device
    params['batchsize'] = 32
    params['epochs'] = 10

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
        estimate_strf(dfs, dataset_type, params, figloc)
        print("label: ", datafiles['label'][count])
    
