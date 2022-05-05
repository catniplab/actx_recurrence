import numpy as np
import os, pickle, random, math
import scipy.io, scipy
import pandas as pd

import matplotlib.pyplot as plt
from .dataloader_base import Data_Loading
from src.plotting import plot_raster
from torch.utils.data import Dataset
import torch

class Data_Loading_STRF(Data_Loading):
    def __init__(self, PARAMS, foldername):
        super().__init__(PARAMS, foldername)

    def get_loaded_data(self):
        raster, raster_full, rng = get_event_raster_one_trial(self.stimuli_df, self.spike_df,\
                self.PARAMS)
        return self.stimuli_df, self.spike_df, raster, raster_full, rng

    def load_data(self, foldername, PARAMS):
        """
            Data loading for frequency modulated stimulus recordings
            We need three file in a folder: stimuli.mat, tt_spikes.mat, and stimuli.dat
            Collects stimuli points, creates the stimuli, and returns the spikes and the stimuli raw
                details 
            Returns: Raw stimuli information data frame and raw spiking time stamp dataframe
        """
        fileslist = [f for f in os.listdir(foldername) if os.path.isfile(foldername+f)]
        sample_rate = PARAMS['samplerate'] #sampling rate of neuron activity

        ## opening -stimulti.mat
        stimuli_raw_file = [f for f in fileslist if "-stimuli.mat" in f][0]
        stimuli_raw = scipy.io.loadmat(foldername+stimuli_raw_file)

        # stimuli has the following fields in the initial dataset
        stimuli_raw_data = stimuli_raw['stimuli']
        stimuli = {'type':[], 'stim_length':[], 'trigger':[], 'datafile':[], 'param':[]}

        for i in range(stimuli_raw_data.shape[0]):
            stimuli['type'].append(stimuli_raw_data[i]['type'][0][0])
            stimuli['stim_length'].append(stimuli_raw_data[i]['stimlength'][0][0][0])
            stimuli['trigger'].append(stimuli_raw_data[i]['trigger'][0][0][0]/sample_rate)
            stimuli['datafile'].append(stimuli_raw_data[i]['datafile'][0][0][0])
            
            # for STRF cloud data in STRF specifc stimuli data
            if(stimuli_raw_data[i]['type'][0][0] == 'strfcloud'):
                stimuli_param = {'durPipZZstrf':[], 'rampLenZZstrf':[], 'freqsZZstrf':[],
                        'ordZZstrf':[], 'ampsZZstrf':[], 'counter':[], 'duration':[], 'next':[],
                        'empiricalDur':[]}
                stimuli_param['freqsZZstrf'] = stimuli_raw_data[i]\
                        [0]['param']['freqsZZstrf'][0][0][0][0]
                stimuli_param['rampLenZZstrf'] = stimuli_raw_data[i]\
                        [0]['param']['rampLenZZstrf'][0][0][0][0]
                stimuli_param['durPipZZstrf'] = stimuli_raw_data[i]\
                        [0]['param']['durPipZZstrf'][0][0][0][0]
                stimuli_param['ordZZstrf'] = stimuli_raw_data[i][0]['param']['ordZZstrf'][0][0][0][0]
                stimuli_param['ampsZZstrf'] = stimuli_raw_data[i][0]['param']['ampsZZstrf'][0][0][0][0]
                stimuli_param['counter'] = stimuli_raw_data[i][0]['param']['counter'][0][0][0][0]
                stimuli_param['next'] = stimuli_raw_data[i][0]['param']['next'][0][0][0][0]
                stimuli_param['empiricalDur'] = stimuli_raw_data[i]\
                        [0]['param']['empiricalDur'][0][0][0][0]
                stimuli_param['duration'] = stimuli_raw_data[i][0]['param']['duration'][0][0][0][0]

            elif(stimuli_raw_data[i]['type'][0][0] == 'tone'):
                stimuli_param = {'frequency':[], 'amplitude':[], 'ramp':[], 'duration':[]}
                stimuli_param['frequency'] = stimuli_raw_data[i][0]['param']['frequency'][0][0][0][0]
                stimuli_param['amplitude'] = stimuli_raw_data[i][0]['param']['amplitude'][0][0][0][0]
                stimuli_param['ramp'] = stimuli_raw_data[i][0]['param']['ramp'][0][0][0][0]
                stimuli_param['duration'] = stimuli_raw_data[i][0]['param']['duration'][0][0][0][0]

            elif(stimuli_raw_data[i]['type'][0][0] == 'whitenoise'):
                stimuli_param = {'amplitude':[], 'ramp':[], 'duration':[]}
                stimuli_param['amplitude'] = stimuli_raw_data[i][0]['param']['amplitude'][0][0][0][0]
                stimuli_param['ramp'] = stimuli_raw_data[i][0]['param']['ramp'][0][0][0][0]
                stimuli_param['duration'] = stimuli_raw_data[i][0]['param']['duration'][0][0][0][0]

            elif(stimuli_raw_data[i]['type'][0][0] == 'fmsweep'):
                stimuli_param = {'amplitude':[], 'ramp':[], 'duration':[], 'speed':[], 'method':[],
                        'next':[], 'start_frequency':[], 'stop_frequency':[]}
                stimuli_param['amplitude'] = stimuli_raw_data[i][0]['param']['amplitude'][0][0][0][0]
                stimuli_param['ramp'] = stimuli_raw_data[i][0]['param']['ramp'][0][0][0][0]
                stimuli_param['duration'] = stimuli_raw_data[i][0]['param']['duration'][0][0][0][0]
                stimuli_param['speed'] = stimuli_raw_data[i][0]['param']['speed'][0][0][0][0]
                stimuli_param['method'] = stimuli_raw_data[i][0]['param']['method'][0][0][0][0]
                stimuli_param['next'] = stimuli_raw_data[i][0]['param']['next'][0][0][0][0]
                stimuli_param['start_frequency'] = stimuli_raw_data[i][0]\
                    ['param']['start_frequency'][0][0][0][0]
                stimuli_param['stop_frequency'] = stimuli_raw_data[i][0]\
                    ['param']['stop_frequency'][0][0][0][0]

            # for clicktrain data
            elif(stimuli_raw_data[i]['type'][0][0] == 'clicktrain'):
                stimuli_param = {'amplitude':[], 'ramp':[], 'duration':[], 'next':[], 'start':[],
                        'isi':[], 'clickduration':[], 'nclicks':[]}
                stimuli_param['amplitude'] = stimuli_raw_data[i][0]['param']['amplitude'][0][0][0][0]
                stimuli_param['ramp'] = stimuli_raw_data[i][0]['param']['ramp'][0][0][0][0]
                stimuli_param['duration'] = stimuli_raw_data[i][0]['param']['duration'][0][0][0][0]
                stimuli_param['next'] = stimuli_raw_data[i][0]['param']['next'][0][0][0][0]
                stimuli_param['start'] = stimuli_raw_data[i][0]['param']['start'][0][0][0][0]
                stimuli_param['isi'] = stimuli_raw_data[i][0]['param']['isi'][0][0][0][0]
                stimuli_param['clickduration'] = stimuli_raw_data[i][0]['param']['clickduration'][0][0][0][0]
                stimuli_param['nclicks'] = stimuli_raw_data[i][0]['param']['nclicks'][0][0][0][0]

            stimuli['param'].append(stimuli_param)

        stimuli_df = pd.DataFrame(stimuli)

        ## opening -tt_spikes.dat
        spikes_raw_file = [f for f in fileslist if "-tt_spikes.dat" in f][0]
        spikes_raw = scipy.io.loadmat(foldername+spikes_raw_file)

        spike_data = {'timestamps':[], 'waveforms':[], 'waveforms_raw':[]}
        for i in range(spikes_raw['timestamps'].shape[0]):
            spike_data['timestamps'].append(spikes_raw['timestamps'][i][0]/sample_rate)
            spike_data['waveforms'].append(spikes_raw['waveforms'][i])
            spike_data['waveforms_raw'].append(spikes_raw['waveforms_raw'][i])

        spike_data_df = pd.DataFrame(spike_data)
        # print("spike data:", spike_data_df)

        ## opening -data.dat file
        # data_raw_file = [f for f in fileslist if "-data.dat" in f][0]
        # data_raw = scipy.io.loadmat(foldername+data_raw_file)
        # print(data_raw)

        return stimuli_df, spike_data_df

    # def get_sorted_event_raster(stimuli_df, spikes_df, PARAMS):
        # ## select stimuli 
        # fieldname = 'fmsweep'
        # sample_rate = PARAMS['sample_rate']
        # rng = PARAMS['rng']

        # stimuli_field = stimuli_df.loc[stimuli_df['type'] == 'fmsweep']
        # stimuli_field_param = stimuli_field.param.dropna().apply(pd.Series)

        # # sort the trials on the basis of the sweep direcions
        # sweepdir = stimuli_field_param['start_frequency'].to_numpy()
        # speeds = stimuli_field_param['speed']
        # sorted_idxs = np.argsort(sweepdir) 
        # df_index_sorted = stimuli_field.index.to_numpy()[sorted_idxs]
        # stimuli_field_sorted = stimuli_field.reindex(df_index_sorted)
        # triggers = stimuli_field_sorted['trigger'].to_numpy()

        # # each trial start and stop bounds
        # start_samples = round(rng[0]*sample_rate)
        # stop_samples = round(rng[1]*sample_rate)
        # n_triggers = triggers.shape[0]
        # n_samples = stop_samples - start_samples
        # raster_full = np.zeros([n_triggers, n_samples])
        # raster = []

        # # for each stimuli trigger create a new trial
        # # from trigger -rng[0] to trigger + rng[1] for all triggers
        # for i in range(n_triggers):
            # spikes = spikes_df.loc[(spikes_df['timestamps']>triggers[i]+rng[0]) &
                                    # (spikes_df['timestamps']<triggers[i]+rng[1])]
            # spikes = spikes['timestamps'].to_numpy()
            # if(spikes.shape[0] > 0):
                # spike_pos = np.floor((spikes - triggers[i])*sample_rate) + 1 - start_samples
                # raster_full[i, spike_pos.astype(int)] = 1
                # raster.append(spike_pos/sample_rate)
            # else:
                # raster.append([])

        # raster = np.asarray(raster)
        # return raster, raster_full

    # def get_event_raster(stimuli_df, spikes_df, PARAMS):
        # """
            # It selects all the events which are atleast $minduration$ long
                # > for each stimuli check for a 1640ms open gap 
                # > if open gap then take the spikes in that time frame
                # > pass each such open time frame as one trial
            # Returns: event raster and a full binary raster
        # """
        # sample_rate = PARAMS['sample_rate']
        # minduration = PARAMS['minduration']
        # rng = PARAMS['rng']
        # numstimulis = stimuli_df.size
        # triggertimes = []

        # for i in range(numstimulis-1):
            # stimuli = stimuli_df[i:i+1]
            # stimuli_next = stimuli_df[i+1:i+2]
            # stimuli_param = stimuli.param.dropna().apply(pd.Series) 
            # stimuli_trigger = stimuli['trigger'].to_numpy()
            # stimuli_next_trigger = stimuli_next['trigger'].to_numpy()
            # stimuli_duration = stimuli['stim_length'].to_numpy()/1000
            # # # if((stimuli_next_trigger - (stimuli_trigger+stimuli_duration))>minduration):
                # # # triggertimes.append([stimuli_trigger+stimuli_duration,
                    # # # stimuli_trigger+stimuli_duration+minduration])
            # if((stimuli_next_trigger - stimuli_trigger)>minduration):
                # triggertimes.append([stimuli_trigger, stimuli_trigger+minduration])

        # n_triggers = len(triggertimes)
        # print("trigger trials", n_triggers)
        # start_samples = round(rng[0]*sample_rate)
        # stop_samples = round(rng[1]*sample_rate)
        # n_samples = stop_samples - start_samples
        # raster_full = np.zeros([n_triggers, n_samples])
        # raster = []

        # for i in range(n_triggers):
            # spikes = spikes_df.loc[(spikes_df['timestamps']>(triggertimes[i][0][0]+rng[0])) &
                                    # (spikes_df['timestamps']<(triggertimes[i][0][0]+rng[1]))]
            # spikes = spikes['timestamps'].to_numpy()
            # if(spikes.shape[0] > 0):
                # spike_pos = np.floor((spikes - triggertimes[i][0][0])*sample_rate) + 1\
                        # - start_samples
                # raster_full[i, spike_pos.astype(int)] = 1
                # spike_pos = np.floor((spikes - triggertimes[i][0][0])*sample_rate) + 1
                # raster.append(spike_pos/sample_rate)
            # else:
                # raster.append([])

        # raster = np.asarray(raster)
        # return raster, raster_full


class STRF_Dataset_randomchords(Dataset):
    def __init__(self, params, stimuli_df, spikes_df, fmsweepdata):
        self.params = params
        self.device = params['device']
        self.spikes_df = spikes_df
        self.stimuli_df = stimuli_df
        self.strf_bins = int(self.params['strf_timerange'][1]/self.params['strf_timebinsize'])
        self.hist_bins = int(params['hist_size']/params['strf_timebinsize'])
        spiketimes = spikes_df['timestamps'].to_numpy() # in seconds
        total_time = np.ceil(spiketimes[-1]+1) #s ##??? why not take the total time from stimuli?
        params['total_time'] = total_time
        params['timebin'] = params['strf_timebinsize']
        samples_per_bin = int(params['samplerate']*params['strf_timebinsize']) #num samples per bin
        self.spikes_binned = torch.tensor(self.binned_spikes(params, spiketimes),
                device=self.device)
        print("spike binned: ", self.spikes_binned.shape)
        self.binned_freq, self.binned_amp = fmsweepdata.get_stimuli_barebone(stimuli_df, params)
        self.binned_freq = torch.tensor(self.binned_freq, device=self.device)
        self.binned_amp = torch.tensor(self.binned_amp, device=self.device)
        self.spiking_bins = torch.tensor([i for i in range(self.strf_bins+1,\
            self.strf_bins+self.params['batchsize']*1000)], device = self.device)
        # self.spiking_bins = torch.tensor([i for i in range(self.strf_bins+1,\
            # self.spikes_binned.shape[1])], device = self.device)
        # self.spiking_bins = torch.nonzero(self.spikes_binned)[:,1]

    def __len__(self):
        return self.spiking_bins.shape[0]

    def __getitem__(self, idx):
        stimuli_spectro = self.stimuli_baretospectrogram(int(self.spiking_bins[idx]),\
                self.binned_freq, self.binned_amp, self.params)
        spike_history = self.spikes_binned[0, int(self.spiking_bins[idx] -\
            self.hist_bins-1):int(self.spiking_bins[idx]-1)]
        stimuli_spectro = torch.tensor(stimuli_spectro, device=self.device)
        spike_history = torch.tensor(spike_history, device=self.device)
        eta = torch.tensor(np.random.normal(0, 1, 1), device=self.device)
        return stimuli_spectro.type(torch.float32), spike_history.type(torch.float32), eta,\
            self.spikes_binned[0, self.spiking_bins[idx]].type(torch.float32)

    def binned_spikes(self, params, spiketimes):
        total_time = np.ceil(spiketimes[-1]+1) #s
        binned_spikes = np.zeros((1, int(total_time / params['strf_timebinsize'])))
        for count, spiket in enumerate(spiketimes):
            binned_spikes[0, int(spiket/params['strf_timebinsize'])]+=1
        return binned_spikes
    
    def stimuli_baretospectrogram(self, spikebin, binned_freq, binned_amp, params):
        num_timebins =\
            int((params['strf_timerange'][1]-params['strf_timerange'][0])/params['strf_timebinsize'])
        num_freqbins = int((params['freqrange'][1]-params['freqrange'][0])/params['freqbinsize'])
        stimuli_spectrogram = np.zeros((num_freqbins, num_timebins))
        timeslice = range(spikebin-1-num_timebins, spikebin-1)

        for idx, ts in enumerate(timeslice):
            # print("freq bin size", params['freqbinsize'], "binner freq", binned_freq[0, ts])
            stimuli_spectrogram[int(binned_freq[0, ts]//params['freqbinsize']), idx] = binned_amp[0, ts]
        return stimuli_spectrogram

def loaddata_withraster(foldername, PARAMS):
    fmsdata = Data_Loading_STRF(PARAMS, foldername)

    # raster, raster_full = get_sorted_event_raster(stimuli_df, spike_df, rng)
    raster, raster_full = fmsdata.get_event_raster(fmsdata.stimuli_df, fmsdata.spike_data_df, PARAMS)
    return fmsdata.stimuli_df, fmsdata.spike_data_df, raster, raster_full, fmsdata

if (__name__ == "__main__"):
    PARAMS = {
            'rng': [-0.5, 2], 
            'sample_rate': 10000, 
            'minduration': 1.640 #s
        }
    foldername = "../../data/strf_data/20210825-xxx999-002-001/"
    stimuli_df, spike_df, raster, raster_full = loaddata_withraster(foldername, PARAMS)
    plot_raster(raster)
 
