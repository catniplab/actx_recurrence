import numpy as np
import os, pickle, random, math
import scipy.io, scipy
import pandas as pd

import matplotlib.pyplot as plt
from src.plotting import plot_raster
from torch.utils.data import Dataset
import torch

class Data_Loading():
    def __init__(self, PARAMS, foldername):
        self.PARAMS = PARAMS
        self.stimuli_df, self.spike_data_df = self.load_data(foldername, PARAMS)

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

            # for pure tone part of the stimulus
            if(stimuli_raw_data[i]['type'][0][0] == 'tone'):
                stimuli_param = {'frequency':[], 'amplitude':[], 'ramp':[], 'duration':[]}
                stimuli_param['frequency'] = stimuli_raw_data[i][0]['param']['frequency'][0][0][0][0]
                stimuli_param['amplitude'] = stimuli_raw_data[i][0]['param']['amplitude'][0][0][0][0]
                stimuli_param['ramp'] = stimuli_raw_data[i][0]['param']['ramp'][0][0][0][0]
                stimuli_param['duration'] = stimuli_raw_data[i][0]['param']['duration'][0][0][0][0]
            
            # for whitenoise part of the stimulus
            elif(stimuli_raw_data[i]['type'][0][0] == 'whitenoise'):
                stimuli_param = {'amplitude':[], 'ramp':[], 'duration':[]}
                stimuli_param['amplitude'] = stimuli_raw_data[i][0]['param']['amplitude'][0][0][0][0]
                stimuli_param['ramp'] = stimuli_raw_data[i][0]['param']['ramp'][0][0][0][0]
                stimuli_param['duration'] = stimuli_raw_data[i][0]['param']['duration'][0][0][0][0]

            # for the frequency modulated sweeps part of the stimulus
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

        # raw stimuli dataframe
        stimuli_df = pd.DataFrame(stimuli)

        ## opening -tt_spikes.dat which gives us the raw spiking data 
        spikes_raw_file = [f for f in fileslist if "-tt_spikes.dat" in f][0]
        spikes_raw = scipy.io.loadmat(foldername+spikes_raw_file)

        # spiking data has single neuron spiking time points and the raw waveforms
        # we only require the spike times for our computations
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

    def get_sorted_event_raster(self, stimuli_df, spikes_df, PARAMS):
        """
            Generates the spike raster from spikes_df and sort it according to the sweep direction
            Aligns the trials, and cut the trials in the specified time interval
            Returns: an event raster with all the spike times for each sample; a full binary spiking
                raster matrix with value 1 for the time bin which actually spikes
        """

        ## select stimuli type
        if 'stimuli_type' in PARAMS:
            stimuli_type = PARAMS['stimuli_type']
        else:
            stimuli_type = 'fmsweep'
        sample_rate = PARAMS['samplerate']
        rng = PARAMS['rng']

        stimuli_field = stimuli_df.loc[stimuli_df['type'] == stimuli_type]
        stimuli_field_param = stimuli_field.param.dropna().apply(pd.Series)

        # sort the trials on the basis of the sweep direcions
        sweepdir = stimuli_field_param['start_frequency'].to_numpy()
        speeds = stimuli_field_param['speed']
        sorted_idxs = np.argsort(sweepdir) 
        df_index_sorted = stimuli_field.index.to_numpy()[sorted_idxs]
        stimuli_field_sorted = stimuli_field.reindex(df_index_sorted)
        triggers = stimuli_field_sorted['trigger'].to_numpy()

        # each trial start and stop bounds
        start_samples = round(rng[0]*sample_rate)
        stop_samples = round(rng[1]*sample_rate)
        n_triggers = triggers.shape[0]
        n_samples = stop_samples - start_samples
        raster_full = np.zeros([n_triggers, n_samples])
        raster = []

        # for each stimuli trigger create a new trial
        # from trigger -rng[0] to trigger + rng[1] for all triggers
        for i in range(n_triggers):
            spikes = spikes_df.loc[(spikes_df['timestamps']>triggers[i]+rng[0]) &
                                    (spikes_df['timestamps']<triggers[i]+rng[1])]
            spikes = spikes['timestamps'].to_numpy()
            if(spikes.shape[0] > 0):
                spike_pos = np.floor((spikes - triggers[i])*sample_rate) + 1 - start_samples
                raster_full[i, spike_pos.astype(int)] = 1
                raster.append(spike_pos/sample_rate)
            else:
                # if there were no spike in a trial the raster for that trial will be empty
                raster.append([])

        raster = np.asarray(raster)
        return raster, raster_full

    def get_event_raster(self, stimuli_df, spikes_df, PARAMS):
        """
            It selects all the events which are atleast $minduration$ long
                > for each stimuli check for a 1640ms open gap 
                > if open gap then take the spikes in that time frame
                > pass each such open time frame as one trial
            Returns: event raster and a full binary raster
        """

        sample_rate = PARAMS['samplerate']
        minduration = PARAMS['minduration']
        rng = PARAMS['rng']
        numstimulis = stimuli_df.size
        triggertimes = []
        del_time = 0.001 #s

        # find all trials which are minduration long
        for i in range(numstimulis-1):
            stimuli = stimuli_df[i:i+1]
            stimuli_next = stimuli_df[i+1:i+2]
            stimuli_param = stimuli.param.dropna().apply(pd.Series) 
            stimuli_trigger = stimuli['trigger'].to_numpy()
            stimuli_next_trigger = stimuli_next['trigger'].to_numpy()
            stimuli_duration = stimuli['stim_length'].to_numpy()/1000
            # if((stimuli_next_trigger - (stimuli_trigger+stimuli_duration))>minduration):
                # triggertimes.append([stimuli_trigger+stimuli_duration,
                    # stimuli_trigger+stimuli_duration+minduration])
            if((stimuli_next_trigger - stimuli_trigger)>minduration+del_time):
                # triggertimes.append([stimuli_trigger, stimuli_trigger+minduration])
                triggertimes.append([stimuli_next_trigger-minduration-del_time,\
                    stimuli_next_trigger-del_time])

        n_triggers = len(triggertimes)
        start_samples = round(rng[0]*sample_rate)
        stop_samples = round(rng[1]*sample_rate)
        n_samples = stop_samples - start_samples
        raster_full = np.zeros([n_triggers, n_samples])
        raster = []

        for i in range(n_triggers):
            spikes = spikes_df.loc[(spikes_df['timestamps']>triggertimes[i][0][0]+rng[0]) &
                                    (spikes_df['timestamps']<triggertimes[i][0][0]+rng[1])]
            spikes = spikes['timestamps'].to_numpy()
            if(spikes.shape[0] > 0):
                # raster.append(spike_pos/sample_rate)
                spike_pos = np.floor((spikes - triggertimes[i][0][0])*sample_rate)\
                        - start_samples
                if((spike_pos>=n_samples).any()):
                    print(spike_pos, start_samples, triggertimes[i][0][0], spikes)
                raster_full[i, spike_pos.astype(int)] = 1
                spike_pos = np.floor((spikes - triggertimes[i][0][0])*sample_rate) 
                raster.append(spike_pos/sample_rate)
            else:
                raster.append([])

        raster = np.asarray(raster)
        return raster, raster_full

    def get_event_raster_single_trial(self, stimuli_df, spikes_df, PARAMS):
           # rng, minduration=1.640):
        """
            Load all the data as a single trial instead of num_trigger trials
            Returns: Spike event raster in timestamps; Full spike raster with binary spike
            representation; updated range wrt max input data shape
        """
        ## load the entire data as a single trial
        sample_rate = PARAMS['samplerate']
        numstimulis = stimuli_df.size
        triggertimes = []
        raster = []

        # updating/setting range to be the full data as it will be a single trial
        rng = [0, spikes_df['timestamps'].tolist()[-1]]

        spikes = spikes_df['timestamps'].loc[(spikes_df['timestamps']>rng[0]) &
                (spikes_df['timestamps']<rng[1])]
        # print(rng)
        # spikes = spikes/sample_rate
        raster.append(spikes.tolist())
        total_len = raster[0][-1]
        # print(total_len)
        raster_full = np.zeros([1, int(total_len*sample_rate)+1])
        for i in range(len(raster[0])):
            # raster_full[0, int(spikes_df['timestamps'][i]/sample_rate)]=1
            raster_full[0, int((raster[0][i]-rng[0])*sample_rate)]=1

        raster = np.asarray(raster)
        return raster, raster_full, rng

    def get_stimuli_barebone(self, stimuli_df, PARAMS):
            # total_time, timebin, sample_rate, freqrange, max_amp): #timebin in s
        """
            Get a barebone description of the stimulus instead of the whole spectrogram as it can be
            very memory intensive to manipulate all the time
            It also resamples the stimulus to a binsize defined in params

            Return: Time varying frequency and time varying amplitude
        """
        numstimulis = stimuli_df.shape[0]
        total_time = PARAMS['total_time']
        timebin = PARAMS['timebin']
        sample_rate = PARAMS['samplerate']
        freqrange = PARAMS['freqrange']
        max_amp = PARAMS['max_amp']

        stimuli_bare_freq = np.zeros((1, int(total_time*sample_rate)))
        stimuli_bare_amp = np.zeros((1, int(total_time*sample_rate)))

        for i in range(numstimulis):
            # for each stimulus trial
            stim_i = stimuli_df.iloc[i]
            stim_i_params = stimuli_df.iloc[i]['param']
            stim_i_start = int(stim_i['trigger']*sample_rate)
            if(stim_i_start>total_time*sample_rate):
                # if incorrect stim start outside the total time being considered
                break
            stim_i_length = int(stim_i_params['duration']*(sample_rate/1000)) # ms->s->samplenum

            # creating the stimuli for each trial
            for j in range(stim_i_length):

                # pure tone type input
                if(stim_i['type']=='tone'):
                    freq_i = stim_i['param']['frequency']
                    amp_i = stim_i_params['amplitude']
                    # spectrogram_full[stim_i_start+j, freq_i] = amp_i
                    stimuli_bare_freq[0, stim_i_start+j] = freq_i
                    stimuli_bare_amp[0, stim_i_start+j] = amp_i

                # frequency sweep type stimuli
                # TODO: add the ramp from pure tone to min/max frequency of the sweep
                elif(stim_i['type']=='fmsweep'):
                    start_freq_i = stim_i_params['start_frequency']
                    stop_freq_i = stim_i_params['stop_frequency']
                    if(start_freq_i<stop_freq_i):
                        freq_speed = stim_i_params['speed'] # octaves per sec
                    else:
                        freq_speed = -1*stim_i_params['speed'] # octaves per sec
                    amp_i = stim_i_params['amplitude']
                    freq_j = int(start_freq_i*(2**(freq_speed*(j/sample_rate))))
                    stimuli_bare_freq[0, stim_i_start+j] = freq_j
                    stimuli_bare_amp[0, stim_i_start+j] = amp_i

                # for white noise input like in later datasets
                elif(stim_i['type']=='whitenoise'):
                    amp_i = stim_i_params['amplitude']
                    rand_freq = np.random.uniform(freqrange[0], freqrange[1], (1,1))
                    stimuli_bare_freq[0, stim_i_start+j] = rand_freq
                    stimuli_bare_amp[0, stim_i_start+j] = amp_i

        binsize = int(timebin*sample_rate)
        binned_stimuli_bare_freq = np.zeros((1, int((total_time*sample_rate)/binsize)))
        binned_stimuli_bare_amp = np.zeros((1, int((total_time*sample_rate)/binsize)))

        # resizing the bin size where the new bin is given mean stimulus values on all the values in
        # that window
        import torch
        for bn in range(int((total_time*sample_rate)/binsize)):
            binned_stimuli_bare_freq[0, bn] = np.mean(stimuli_bare_freq[0, bn*binsize:(bn+1)*binsize])
            binned_stimuli_bare_amp[0, bn] = np.mean(stimuli_bare_amp[0,
                bn*binsize:(bn+1)*binsize])/max_amp

        return binned_stimuli_bare_freq, binned_stimuli_bare_amp
 
