import numpy as np
import os, pickle, random, math
import scipy.io, scipy
import pandas as pd
import math

import matplotlib.pyplot as plt
from .dataloader_base import Data_Loading
from src.plotting import plot_raster, plot_spectrogram

class Data_Loading_DMR(Data_Loading):

    def __init__(self, PARAMS, foldername, specfilename = None):
        self.PARAMS = PARAMS
        self.stimuli_df, self.spike_data_df, self.dmr_stamps = self.load_data(foldername, PARAMS)
        self.spectrogram_data = self.load_spectrogram(specfilename)

    def load_spectrogram(self, spectrogram_file = None):
        if(spectrogram_file is not None):
            specdata = np.fromfile(spectrogram_file, dtype=np.float32)
            specdata = self.reshape_spectrogram(specdata)
        else:
            specdata = None
        return specdata

    def reshape_spectrogram(self, specdata):
        spectrshape = [self.PARAMS['num_freqs'], specdata.shape[0]//self.PARAMS['num_freqs']]
        # spectrshape = [specdata.shape[0]//self.PARAMS['num_freqs'], self.PARAMS['num_freqs']]
        # specdata = np.transpose(np.reshape(specdata, spectrshape))
        specdata = np.reshape(specdata, spectrshape)
        return specdata

    def load_data(self, foldername, PARAMS):
        """
            Data loading for frequency modulated stimulus recordings
            We need three file in a folder: stimuli.mat, tt_spikes.mat, and stimuli.dat
            Collects stimuli points, creates the stimuli, and returns the spikes and the stimuli raw
                details 
            Returns: Raw stimuli information data frame and raw spiking time stamp dataframe, and DMR
                stimuli stamps
        """
        fileslist = [f for f in os.listdir(foldername) if os.path.isfile(foldername+f)]
        sample_rate = PARAMS['sample_rate'] #sampling rate of neuron activity

        ## opening -stimulti.mat
        stimuli_raw_file = [f for f in fileslist if "stimuli.mat" in f][0]
        stimuli_raw = scipy.io.loadmat(foldername+stimuli_raw_file)

        stimuli_raw_data = stimuli_raw['stimuli']
        stimuli = {'type':[], 'stim_length':[], 'trigger':[], 'datafile':[], 'param':[]}

        for i in range(stimuli_raw_data.shape[0]):
            stimuli['type'].append(stimuli_raw_data[i]['type'][0][0])
            stimuli['stim_length'].append(stimuli_raw_data[i]['stimlength'][0][0][0])
            stimuli['trigger'].append(stimuli_raw_data[i]['trigger'][0][0][0]/sample_rate)
            stimuli['datafile'].append(stimuli_raw_data[i]['datafile'][0][0][0])

            # this data has different stimuli type tags, this is for natural sound
            if(stimuli_raw_data[i]['type'][0][0] == 'naturalsound'):
                stimuli_param = {'file':[], 'ramp':[], 'description':[], 'duration':[]}
                stimuli_param['file'] = stimuli_raw_data[i][0]['param']['file'][0][0][0][0]
                stimuli_param['ramp'] = stimuli_raw_data[i][0]['param']['ramp'][0][0][0][0]
                stimuli_param['description'] =\
                        stimuli_raw_data[i][0]['param']['description'][0][0][0][0]
                stimuli_param['duration'] = stimuli_raw_data[i][0]['param']['duration'][0][0][0][0]

            elif(stimuli_raw_data[i]['type'][0][0] == 'strfcloud'):
                stimuli_param = {'durPipZZstrf':[], 'rampLenZZstrf':[], 'freqsZZstrf':[],
                        'ordZZstrf':[], 'ampsZZstrf':[], 'counter':[], 'duration':[], 'next':[],
                        'empiricalDur':[]}
                stimuli_param['freqsZZstrf'] =\
                        stimuli_raw_data[i][0]['param']['freqsZZstrf'][0][0][0][0]
                stimuli_param['rampLenZZstrf'] = stimuli_raw_data[i]\
                        [0]['param']['rampLenZZstrf'][0][0][0][0]
                stimuli_param['durPipZZstrf'] = stimuli_raw_data[i]\
                        [0]['param']['durPipZZstrf'][0][0][0][0]
                stimuli_param['ordZZstrf'] = stimuli_raw_data[i][0]['param']['ordZZstrf'][0][0][0][0]
                stimuli_param['ampsZZstrf'] =\
                        stimuli_raw_data[i][0]['param']['ampsZZstrf'][0][0][0][0]
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

            stimuli['param'].append(stimuli_param)

        stimuli_df = pd.DataFrame(stimuli)

        ## opening normalized spiking file in other naming format of some dataset
        # spikes_raw_file = [f for f in fileslist if "spikesnormalized.mat" in f][0]
        # spikes_raw = scipy.io.loadmat(foldername+spikes_raw_file)
        # spike_data = {'timestamps':[]}
        # for i in range(spikes_raw['spikesout'].shape[0]):
            # spike_data['timestamps'].append(spikes_raw['spikesout'][i][0]/sample_rate)
            # # spike_data['waveforms'].append(spikes_raw['waveforms'][i])
            # # spike_data['waveforms_raw'].append(spikes_raw['waveforms_raw'][i])

        ## opening -tt_spikes.dat which gives us the raw spiking data 
        spikes_raw_file = [f for f in fileslist if "-tt_spikes.dat" in f][0]
        spikes_raw = scipy.io.loadmat(foldername+spikes_raw_file)
        spike_data = {'timestamps':[], 'waveforms':[], 'waveforms_raw':[]}
        for i in range(spikes_raw['timestamps'].shape[0]):
            spike_data['timestamps'].append(spikes_raw['timestamps'][i][0]/sample_rate)
            spike_data['waveforms'].append(spikes_raw['waveforms'][i])
            spike_data['waveforms_raw'].append(spikes_raw['waveforms_raw'][i])


        spike_data_df = pd.DataFrame(spike_data)

        # dmr_stamps_file = [f for f in fileslist if "DMRstamps.mat" in f][0]
        # stimuli_actual = scipy.io.loadmat(foldername+dmr_stamps_file)
        # dmr_stamps = stimuli_actual['DMRstamps']

        dmr_stamps = None

        ## opening -data.dat file
        # data_raw_file = [f for f in fileslist if "-data.dat" in f][0]
        # data_raw = scipy.io.loadmat(foldername+data_raw_file)
        # print(data_raw)

        return stimuli_df, spike_data_df, dmr_stamps

class DMR_dataset():
    def __init__(self, params, stimuli_df, spikes_df, dmrstamps, specdata):
        self.params = params
        self.device = params['device']
        self.spikes_df = spikes_df
        self.stimuli_df = stimuli_df
        self.dmrstamps = dmrstamps
        self.strf_bins = int(self.params['strf_timerange'][1]/self.params['strf_timebinsize'])
        self.hist_bins = int(params['hist_size']/params['strf_timebinsize'])
        spiketimes = spikes_df['timestamps'].to_numpy() # in seconds
        total_time = np.ceil(spiketimes[-1]+1) #s ##??? why not take the total time from stimuli?
        samples_per_bin = int(params['samplerate']*params['strf_timebinsize']) #num samples per bin
        # self.spikes_binned = torch.tensor(self.binned_spikes(params, spiketimes),
                # device=self.device)
        
        #constants -- same as used in the Eschebi et al paper
        dmr_consts = {
            'f1' : 200,  ##Lowest carrier frequency
            'f2' : 48000,  ##Maximum carrier frequency
            'fRD' : 1.5,  ##Maximum rate of change for RD
            'fFM' : 3,  ##Maximum rate of change for FM
            'MaxRD' : 4,  ##Maximum ripple density (cycles/oct)
            'MaxFM' : 50,  ##Maximum temporal modulation rate (Hz)
            'App' : 30,  ##Peak to peak amplitude of the ripple in dB
            'Fs' : 200e3,  ##Sampling rate
            'NCarriersPerOctave' : 100,
            'NB' : 1,  ##For NB' : 1 genarets DMR
            'Axis' : 'log',
            'Block' : 'n',
            # 'DF' : round(Fs/1000),   ##Sampling rate for envelope single is 3kHz (48kHz/16)
            'AmpDist' : 'dB',
            'seed' : 789
        }

        M = dmr_consts['Fs']*60*5,  ##5 minute long sounds
        self.NS  = math.ceil(dmr_consts['NCarriersPerOctave']*math.log2(dmr_consts['f2']/dmr_consts['f1']))  ##Number of sinusoid carriers. ~100 sinusoids / octave
        shape = [self.NS, int(specdata.shape[0]/self.NS)]
        spectrogram = np.reshape(specdata, shape)
        
        print(f'spectrogram shape: {spectrogram.shape}; spikes shape: {spiketimes.shape}')

    # def create_stimuli_envelope(self, timepoints):


def loaddata_withraster(foldername, PARAMS, specfilename):
    dmrdata = Data_Loading_DMR(PARAMS, foldername, specfilename)

    # raster, raster_full = get_sorted_event_raster(stimuli_df, spike_df, rng)
    raster, raster_full = dmrdata.get_event_raster(dmrdata.stimuli_df, dmrdata.spike_data_df, PARAMS)
    return dmrdata.stimuli_df, dmrdata.spike_data_df, dmrdata.spectrogram_data, raster, raster_full

if (__name__ == "__main__"):
    foldername = "../../data/dmr_data2/20220120-xxx999-006-001/"
    spectrogram_file = "../../data/StimulusFiles_DMR/APR21_DMR50ctx.spr"
    # stimuli_df, spike_df = load_data(foldername)

    # ---- parameters ---- #
    rng = [-0.5, 2]
    device = 'cpu'
    PARAMS = {}
    PARAMS['device'] = device
    PARAMS['strf_timerange'] = [0, 1] #seconds
    PARAMS['strf_timebinsize'] = 0.01 #s = 10ms
    PARAMS['samplerate'] = 10000#samples per second
    PARAMS['hist_size'] = 0.02 #s = 20ms
    PARAMS['rng'] = rng
    PARAMS['sample_rate'] = 10000
    PARAMS['minduration'] = 1.640
    PARAMS['num_freqs'] = 80 # log2(48000/200) + 1

    # stimuli_df, spike_df, raster, raster_full, rng, dmr_stamps, specdata =\
        # loaddata_withraster_dmr(foldername, rng, minduration, spectrogram_file)

    # params['rng'] = rng
    # dmr_dataset = DMR_dataset(params, stimuli_df, spike_df, dmr_stamps, specdata)

    stimuli_df, spike_df, spectrogram, raster, raster_full = loaddata_withraster(foldername, PARAMS,
            spectrogram_file)

    # plot_raster(raster)
    print(f'spectrogram info -- shape: {spectrogram.shape},  min: {np.min(spectrogram)},\
            max: {np.max(spectrogram)}')
    # plot_spectrogram(spectrogram[:, 30000:31000], "../../outputs/spectrogram_test.pdf")



 
