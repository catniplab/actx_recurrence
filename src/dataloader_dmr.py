import numpy as np
import os, pickle, random, math
import scipy.io, scipy
import pandas as pd

import matplotlib.pyplot as plt

def load_data(foldername):
    fileslist = [f for f in os.listdir(foldername) if os.path.isfile(foldername+f)]
    sample_rate = 10000.0
    ## opening -stimulti.mat
    stimuli_raw_file = [f for f in fileslist if "stimuli.mat" in f][0]
    stimuli_raw = scipy.io.loadmat(foldername+stimuli_raw_file)
    # print("data parameters: ", stimuli_raw['param'])
    stimuli_raw_data = stimuli_raw['stimuli']
    stimuli = {'type':[], 'stim_length':[], 'trigger':[], 'datafile':[], 'param':[]}
    for i in range(stimuli_raw_data.shape[0]):
        stimuli['type'].append(stimuli_raw_data[i]['type'][0][0])
        stimuli['stim_length'].append(stimuli_raw_data[i]['stimlength'][0][0][0])
        stimuli['trigger'].append(stimuli_raw_data[i]['trigger'][0][0][0]/sample_rate)
        stimuli['datafile'].append(stimuli_raw_data[i]['datafile'][0][0][0])
        if(stimuli_raw_data[i]['type'][0][0] == 'naturalsound'):
            stimuli_param = {'file':[], 'ramp':[], 'description':[], 'duration':[]}
            stimuli_param['file'] = stimuli_raw_data[i][0]['param']['file'][0][0][0][0]
            stimuli_param['ramp'] = stimuli_raw_data[i][0]['param']['ramp'][0][0][0][0]
            stimuli_param['description'] = stimuli_raw_data[i][0]['param']['description'][0][0][0][0]
            stimuli_param['duration'] = stimuli_raw_data[i][0]['param']['duration'][0][0][0][0]
        if(stimuli_raw_data[i]['type'][0][0] == 'strfcloud'):
            stimuli_param = {'durPipZZstrf':[], 'rampLenZZstrf':[], 'freqsZZstrf':[],
                    'ordZZstrf':[], 'ampsZZstrf':[], 'counter':[], 'duration':[], 'next':[],
                    'empiricalDur':[]}
            stimuli_param['freqsZZstrf'] = stimuli_raw_data[i][0]['param']['freqsZZstrf'][0][0][0][0]
            stimuli_param['rampLenZZstrf'] = stimuli_raw_data[i][0]['param']['rampLenZZstrf'][0][0][0][0]
            stimuli_param['durPipZZstrf'] = stimuli_raw_data[i][0]['param']['durPipZZstrf'][0][0][0][0]
            stimuli_param['ordZZstrf'] = stimuli_raw_data[i][0]['param']['ordZZstrf'][0][0][0][0]
            stimuli_param['ampsZZstrf'] = stimuli_raw_data[i][0]['param']['ampsZZstrf'][0][0][0][0]
            stimuli_param['counter'] = stimuli_raw_data[i][0]['param']['counter'][0][0][0][0]
            stimuli_param['next'] = stimuli_raw_data[i][0]['param']['next'][0][0][0][0]
            stimuli_param['empiricalDur'] = stimuli_raw_data[i][0]['param']['empiricalDur'][0][0][0][0]
            stimuli_param['duration'] = stimuli_raw_data[i][0]['param']['duration'][0][0][0][0]
        if(stimuli_raw_data[i]['type'][0][0] == 'tone'):
            stimuli_param = {'frequency':[], 'amplitude':[], 'ramp':[], 'duration':[]}
            stimuli_param['frequency'] = stimuli_raw_data[i][0]['param']['frequency'][0][0][0][0]
            stimuli_param['amplitude'] = stimuli_raw_data[i][0]['param']['amplitude'][0][0][0][0]
            stimuli_param['ramp'] = stimuli_raw_data[i][0]['param']['ramp'][0][0][0][0]
            stimuli_param['duration'] = stimuli_raw_data[i][0]['param']['duration'][0][0][0][0]
        if(stimuli_raw_data[i]['type'][0][0] == 'whitenoise'):
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
    spikes_raw_file = [f for f in fileslist if "spikesnormalized.mat" in f][0]
    spikes_raw = scipy.io.loadmat(foldername+spikes_raw_file)
    spike_data = {'timestamps':[]}
    for i in range(spikes_raw['spikesout'].shape[0]):
        spike_data['timestamps'].append(spikes_raw['spikesout'][i][0])
        # spike_data['waveforms'].append(spikes_raw['waveforms'][i])
        # spike_data['waveforms_raw'].append(spikes_raw['waveforms_raw'][i])

    spike_data_df = pd.DataFrame(spike_data)
    # print("spike data:", spike_data_df)

    dmr_stamps_file = [f for f in fileslist if "DMRstamps.mat" in f][0]
    dmr_stamps = scipy.io.loadmat(foldername+dmr_stamps_file)
    dmr_stamps = stimuli_actual['DMRstamps']
    # print(dmr_stamps)
    # print(dmr_stamps.shape)
    # stimuli_raw_data = stimuli_raw['stimuli']
    # stimuli = {'type':[], 'stim_length':[], 'trigger':[], 'datafile':[], 'param':[]}


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

    return stimuli_df, spike_data_df, dmr_stamps

def multichannel_waveform_plot(recordings):
    fig, ax = plt.subplots(5,5)
    toshow = random.sample(range(len(recordings)), 25)

    for i in range(5):
        for j in range(5):
            y = recordings[i*5 + j]
            x = [i for i in range(len(y))]
            ax[i,j].plot(x, y)
            # ax[i,j].xlabel("timestamp")
            # ax[i,j].ylabel("voltage")
    plt.show()

def waveform_plot(timestamp, recording):
    plt.plot(timestamp, recording)
    plt.xlabel("timestamp")
    plt.ylabel("voltage")
    plt.show()

def plot_raster(event_data):
    print("event data shape:", event_data.shape)
    plt.eventplot(event_data, linelengths=0.9, linewidth = 0.4)
    plt.xlabel("time")
    plt.ylabel("event count")
    plt.show()

def sortnplot_eventraster(stimuli_df, spikes_df, rng):
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

def get_eventraster(stimuli_df, spikes_df, rng, minduration=1.640):
    # for each stimuli check for a 1640ms open gap
    # if open gap then take the spikes in that time frame
    # pass each such open time frame as one trial
    sample_rate = 10000
    numstimulis = stimuli_df.size
    triggertimes = []

    for i in range(numstimulis-1):
        stimuli = stimuli_df[i:i+1]
        stimuli_next = stimuli_df[i+1:i+2]
        stimuli_param = stimuli.param.dropna().apply(pd.Series) 
        stimuli_trigger = stimuli['trigger'].to_numpy()
        stimuli_next_trigger = stimuli_next['trigger'].to_numpy()
        stimuli_duration = stimuli['stim_length'].to_numpy()/1000
        # # if((stimuli_next_trigger - (stimuli_trigger+stimuli_duration))>minduration):
            # # triggertimes.append([stimuli_trigger+stimuli_duration,
                # # stimuli_trigger+stimuli_duration+minduration])
        if((stimuli_next_trigger - stimuli_trigger)>minduration):
            triggertimes.append([stimuli_trigger, stimuli_trigger+minduration])

    n_triggers = len(triggertimes)
    print("trigger trials", n_triggers)
    start_samples = round(rng[0]*sample_rate)
    stop_samples = round(rng[1]*sample_rate)
    n_samples = stop_samples - start_samples
    raster_full = np.zeros([n_triggers, n_samples])
    raster = []

    for i in range(n_triggers):
        spikes = spikes_df.loc[(spikes_df['timestamps']>(triggertimes[i][0][0]+rng[0])) &
                                (spikes_df['timestamps']<(triggertimes[i][0][0]+rng[1]))]
        spikes = spikes['timestamps'].to_numpy()
        if(spikes.shape[0] > 0):
            spike_pos = np.floor((spikes - triggertimes[i][0][0])*sample_rate) + 1 +\
                np.abs(start_samples)
            # print(spike_pos)
            raster_full[i, spike_pos.astype(int)] = 1
            spike_pos = np.floor((spikes - triggertimes[i][0][0])*sample_rate) + 1
            raster.append(spike_pos/sample_rate)
            # rows.extend([i]*spike_pos.shape[0])
            # columns.extend(spike_pos.astype(int))
        else:
            raster.append([])

    raster = np.asarray(raster)
    return raster, raster_full

def get_eventraster_onetrial(stimuli_df, spikes_df, dmr_stamps, rng, minduration=1.640):
    ## load the entire data as a single trial
    sample_rate = 10000
    numstimulis = stimuli_df.size
    triggertimes = []
    raster = []
    rng = [0, spikes_df['timestamps'].tolist()[-1]]

    spikes = spikes_df['timestamps'].loc[(spikes_df['timestamps']>rng[0]) &
            (spikes_df['timestamps']<rng[1])]
    # print(spikes)
    print(rng)
    # spikes = spikes/sample_rate
    raster.append(spikes.tolist())
    total_len = raster[0][-1]
    print(total_len)
    raster_full = np.zeros([1, int(total_len*sample_rate)+1])
    for i in range(len(raster[0])):
        # raster_full[0, int(spikes_df['timestamps'][i]/sample_rate)]=1
        raster_full[0, int((raster[0][i]-rng[0])*sample_rate)]=1

    raster = np.asarray(raster)
    return raster, raster_full, rng

class dmrdataset(Dataset):
    def __init__(self, params, stimuli_df, spikes_df, dmrstamps):
        self.params = params
        self.device = params['device']
        self.spikes_df = spikes_df
        self.stimuli_df = stimuli_df
        self.dmrstamps = dmrstamps
        self.strf_bins = int(self.params['strf_timerange'][1]/self.params['strf_timebinsize'])
        self.hist_bins = int(params['hist_size']/params['strf_timebinsize'])
        total_time = np.ceil(spiketimes[-1]+1) #s ##??? why not take the total time from stimuli?
        spiketimes = spikes_df['timestamps'].to_numpy() # in seconds
        samples_per_bin = int(params['samplerate']*params['strf_timebinsize']) #num samples per bin
        self.spikes_binned = torch.tensor(self.binned_spikes(params, spiketimes),
                device=self.device)
        
        #constants
        self.spectralmodulation = 0  ## is this correct?
        self.amplitudeconst = 1 ## is this correct?
        self.modulationdepth = 2 ## is this correct?
        self.phase_mu = 0
        self.phase_sigma = 1

    def create_stimuli_envelope(self, timepoints):
        


def loaddata_withraster_dmr(foldername, rng, minduration):
    stimuli_df, spike_df, dmr_stamps = load_data(foldername)
    # rng = [-0.5, 2]
    # rng = [0, 1.640]
    # minduration = 1.640
    # raster, raster_full = sort_eventraster(stimuli_df, spike_df, rng)
    # raster, raster_full = get_eventraster(stimuli_df, spike_df, rng, minduration)
    raster, raster_full, rng = get_eventraster_onetrial(stimuli_df, spike_df, dmr_stamps, rng, minduration)
    return stimuli_df, spike_df, raster, raster_full, rng

if (__name__ == "__main__"):
    # foldername = "..//data/ACx_data_3/ACxCalyx/20200717-xxx999-002-001/"
    foldername = "../data/dmr_data/20211126-xxx999-001-001/"
    # stimuli_df, spike_df = load_data(foldername)
    rng = [-0.5, 2]
    minduration = 10
    loaddata_withraster_dmr(foldername, rng, minduration)
    # raster, raster_full = sortnplot_eventraster(stimuli_df, spike_df, minduration)
    # plot_raster(raster)
 
