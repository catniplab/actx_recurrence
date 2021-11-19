import numpy as np
import os, pickle, random, math
import scipy.io, scipy
import pandas as pd

import matplotlib.pyplot as plt
from plotting import plot_raster

def load_data(foldername):
    fileslist = [f for f in os.listdir(foldername) if os.path.isfile(foldername+f)]
    sample_rate = 10000.0
    ## opening -stimulti.mat
    stimuli_raw_file = [f for f in fileslist if "-stimuli.mat" in f][0]
    stimuli_raw = scipy.io.loadmat(foldername+stimuli_raw_file)
    # print("data parameters: ", stimuli_raw['param'])
    stimuli_raw_data = stimuli_raw['stimuli']
    stimuli = {'type':[], 'stim_length':[], 'trigger':[], 'datafile':[], 'param':[]}
    for i in range(stimuli_raw_data.shape[0]):
        stimuli['type'].append(stimuli_raw_data[i]['type'][0][0])
        stimuli['stim_length'].append(stimuli_raw_data[i]['stimlength'][0][0][0])
        stimuli['trigger'].append(stimuli_raw_data[i]['trigger'][0][0][0]/sample_rate)
        stimuli['datafile'].append(stimuli_raw_data[i]['datafile'][0][0][0])
        if(stimuli_raw_data[i]['type'][0][0] == 'tone'):
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

        stimuli['param'].append(stimuli_param)

    stimuli_df = pd.DataFrame(stimuli)
    # print("stimuli:", stimuli_df)

    ## opening -tt_spikes.dat
    spikes_raw_file = [f for f in fileslist if "-tt_spikes.dat" in f][0]
    spikes_raw = scipy.io.loadmat(foldername+spikes_raw_file)
    # print("spike file keys:", spikes_raw.keys())
    # print("spike file params:", spikes_raw['param'])
    # print("spike waveforms shape:", spikes_raw['waveforms'])
    spike_data = {'timestamps':[], 'waveforms':[], 'waveforms_raw':[]}
    for i in range(spikes_raw['timestamps'].shape[0]):
        spike_data['timestamps'].append(spikes_raw['timestamps'][i][0]/sample_rate)
        spike_data['waveforms'].append(spikes_raw['waveforms'][i])
        spike_data['waveforms_raw'].append(spikes_raw['waveforms_raw'][i])

    spike_data_df = pd.DataFrame(spike_data)
    # print("spike data:", spike_data_df)


    # print("spike waveform data shape:", len(spike_data['waveforms'][0]))
    # print("spike waveform data :", len(spike_data['waveforms']))
    # print("waveform data at idx 0", spike_data['waveforms'][0].shape)

    ## plotting the waveforms of neuron 0
    # y = spike_waveforms_df["waveforms"]
    # x = [i for i in range(len(y))]
    # multichannel_waveform_plot(y)
    # waveform_plot(x,y)

    ## opening -data.dat file
    # data_raw_file = [f for f in fileslist if "-data.dat" in f][0]
    # data_raw = scipy.io.loadmat(foldername+data_raw_file)
    # print(data_raw)

    return stimuli_df, spike_data_df



def sort_eventraster(stimuli_df, spikes_df, rng):
    ## select stimuli 
    fieldname = 'fmsweep'
    sample_rate = 10000
    stimuli_field = stimuli_df.loc[stimuli_df['type'] == 'fmsweep']
    # stimuli_field = stimuli_df[stimuli_field_idxs]
    stimuli_field_param = stimuli_field.param.dropna().apply(pd.Series)

    sweepdir = stimuli_field_param['start_frequency'].to_numpy()
    speeds = stimuli_field_param['speed']
    sorted_idxs = np.argsort(sweepdir) 
    df_index_sorted = stimuli_field.index.to_numpy()[sorted_idxs]
    stimuli_field_sorted = stimuli_field.reindex(df_index_sorted)
    triggers = stimuli_field_sorted['trigger'].to_numpy()

    start_samples = round(rng[0]*sample_rate)
    stop_samples = round(rng[1]*sample_rate)
    n_triggers = triggers.shape[0]
    n_samples = stop_samples - start_samples
    raster_full = np.zeros([n_triggers, n_samples])
    raster = []

    # rows = []
    # columns = []
    for i in range(n_triggers):
        spikes = spikes_df.loc[(spikes_df['timestamps']>triggers[i]+rng[0]) &
                                (spikes_df['timestamps']<triggers[i]+rng[1])]
        spikes = spikes['timestamps'].to_numpy()
        if(spikes.shape[0] > 0):
            spike_pos = np.floor((spikes - triggers[i])*sample_rate) + 1 - start_samples
            # print(spike_pos)
            raster_full[i, spike_pos.astype(int)] = 1
            raster.append(spike_pos/sample_rate)
            # rows.extend([i]*spike_pos.shape[0])
            # columns.extend(spike_pos.astype(int))
        else:
            raster.append([])

    raster = np.asarray(raster)
    # data = np.ones([len(rows)])
    # raster = scipy.sparse.coo_matrix((data, (rows, columns)), shape = (n_triggers,
        # n_samples)).toarray()
    # print(np.sum(raster[30,:]))
    return raster, raster_full

def get_eventraster(stimuli_df, spikes_df, rng, minduration=1.640, sample_rate = 10000):
    # for each stimuli check for a 1640ms open gap
    # if open gap then take the spikes in that time frame
    # pass each such open time frame as one trial
    sample_rate = sample_rate
    numstimulis = stimuli_df.size
    triggertimes = []

    for i in range(numstimulis-1):
        stimuli = stimuli_df[i:i+1]
        stimuli_next = stimuli_df[i+1:i+2]
        stimuli_param = stimuli.param.dropna().apply(pd.Series) 
        stimuli_trigger = stimuli['trigger'].to_numpy()
        stimuli_next_trigger = stimuli_next['trigger'].to_numpy()
        stimuli_duration = stimuli['stim_length'].to_numpy()/1000
        if((stimuli_next_trigger - (stimuli_trigger+stimuli_duration))>minduration):
            triggertimes.append([stimuli_trigger+stimuli_duration,
                stimuli_trigger+stimuli_duration+minduration])

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
            spike_pos = np.floor((spikes - triggertimes[i][0][0])*sample_rate) + 1 - start_samples
            # print(spike_pos)
            raster_full[i, spike_pos.astype(int)] = 1
            raster.append(spike_pos/sample_rate)
            # rows.extend([i]*spike_pos.shape[0])
            # columns.extend(spike_pos.astype(int))
        else:
            raster.append([])

    raster = np.asarray(raster)
    return raster, raster_full

def get_stimulifreq_barebone(stimuli_df, total_time, timebin, samplerate, freqrange): #timebin in s
    numstimulis = stimuli_df.size
    stimuli_bare_freq = np.zeros((1, int(total_time*samplerate)))
    stimuli_bare_amp = np.zeros((1, int(total_time*samplerate)))
    for i in range(numstimulis):
        stim_i = stimuli_df.iloc[i]
        stim_i_params = stimuli_df.iloc[i]['param']
        stim_i_start = int(stim_i['trigger']*sample_rate)
        if(stim_i_start>total_time*sample_rate):
            break
        stim_i_length = int(stim_i_params['duration']*(sample_rate/1000)) #ms->s->samplenum
        for j in range(stim_i_length):
            if(stim_i['type']=='tone'):
                freq_i = stim_i['param']['frequency']
                amp_i = stim_i_params['amplitude']
                # spectrogram_full[stim_i_start+j, freq_i] = amp_i
                stimuli_bare_freq[0, stim_i_start+j] = freq_i
                stimuli_bare_amp[0, stim_i_start+j] = amp_i
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
            elif(stim_i['type']=='whitenoise'):
                amp_i = stim_i_params['amplitude']
                rand_freq = np.random.uniform(freqrange[0], freqrange[1], (1,1))
                stimuli_bare_freq[0, stim_i_start+j] = rand_freq
                stimuli_bare_amp[0, stim_i_start+j] = amp_i

    binsize = int(timebin*samplerate)
    binned_stimuli_bare_freq = np.zeros((1, int(total_time*samplerate)/binsize))
    binned_stimuli_bare_amp = np.zeros((1, int(total_time*samplerate)/binsize))
    for bn in range(int(total_time*samplerate)/binsize):
        binned_stimuli_bare_freq[0, bn] = mean(stimuli_bare_freq[0, bn*binsize:(bn+1)*binsize])
        binned_stimuli_bare_amp[0, bn] = mean(stimuli_bare_amp[0, bn*binsize:(bn+1)*binsize])

    return binned_stimuli_bare_freq, binned_stimuli_bare_amp

def loaddata_withraster(foldername, rng, minduration):
    stimuli_df, spike_df = load_data(foldername)
    # rng = [-0.5, 2]
    # rng = [0, 1.640]
    # minduration = 1.640
    # raster, raster_full = sort_eventraster(stimuli_df, spike_df, rng)
    raster, raster_full = get_eventraster(stimuli_df, spike_df, rng, minduration)
    return stimuli_df, spike_df, raster, raster_full

if (__name__ == "__main__"):
    # foldername = "..//data/ACx_data_3/ACxCalyx/20200717-xxx999-002-001/"
    foldername = "..//data/ACx_data_1/ACxCalyx/20080930-002/"
    stimuli_df, spike_df = load_data(foldername)
    rng = [-0.5, 2]
    raster, raster_full = sortnplot_eventraster(stimuli_df, spike_df, rng)
    plot_raster(raster)
 
