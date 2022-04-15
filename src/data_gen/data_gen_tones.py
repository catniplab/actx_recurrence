import numpy as np
import matplotlib.pyplot as plt
import os,sys
import scipy
from tqdm import tqdm
import h5py
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_minimize.optim import MinimizeWrapper
from scipy.stats import multivariate_normal, poisson
from scipy.io import loadmat
from scipy import stats
from scipy.interpolate import interp1d

import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge
from sklearn.preprocessing import scale

# from dataloader import load_data, get_eventraster, get_stimulifreq_barebone
# from dataloader_strf import loaddata_withraster_strf
from utils import raster_fulltoevents, exponentialClass, spectral_resample, numpify
from plotting import plot_autocor, plot_neuronsummary, plot_utauests, plot_histdata,\
    plot_rasterpsth, plot_spectrogram, plot_strf

np.random.seed(100)
torch.manual_seed(100)

class Gen_Data_RandomPureTone(Dataset):
    def __init__(self, cfg, params):
        self.device = params['device']
        self.params = params
        self.cfg = cfg
        self.strf_bins = int(self.params['strf_timerange'][1]/self.params['strf_timebinsize'])
        self.num_timebins = 30
        self.hist_len = self.params['hist_len']
        self.weights = self.create_filter(params)
        # self.stimuli_nme(strf_filter)
        self.X, self.Y, self.usedidxs, full_spec = self.create_stimuli()
        print("number of non zero spike bins:", torch.nonzero(self.Y).shape)
        print("number of spikes:", torch.sum(self.Y))
        print("full_spec: ", full_spec.shape)

        # print("weights:", self.weights)
        self.calculate_rank(self.X)
        # np.set_printoptions(threshold=sys.maxsize)
        # plot_spectrogram(full_spec[:, 6400:7000].T, "../outputs/spec1.pdf")

    def __create_stimuli(self, cfg, params):
        samplerate = 100
        time_length = 1000 #sec
        freq_range = [500, 5000] #hz
        num_timebins = 30
        num_freqbins = 20
        delay_gap = 0 # int(0.01*samplerate)
        start_point = int(0.1*time_length)
        stim_len = 7
        p_type = [0,1,2]
        w_type = [1, 0.0, 0.0]
        amplitudes = np.asarray([1, 2, 3, 4, 5, 6])
        bias = -3
        # amplitudes /= np.mean(amplitudes)
        noise_amp = .002
        discrete_freq = np.linspace(start = freq_range[0], stop = freq_range[1], num = num_freqbins)
        total_bins = int(time_length*samplerate)
        bins_per_trail = delay_gap+stim_len
        fmsweep = freq_range[0]*np.logspace(0, 0.99, num=stim_len, base=10.0)
        fmsweep = ((fmsweep - freq_range[0])//((freq_range[1]-freq_range[0])/num_freqbins)).astype('int')
        # print(fmsweep)

        full_spec = np.zeros((num_freqbins, total_bins))
        freq_time = np.zeros((1, total_bins))
        freq_time_idx = np.zeros((1, total_bins), dtype=np.int64)
        amp_time = np.zeros((1, total_bins))

        for t in range(start_point, total_bins, bins_per_trail):
            if(t+num_timebins>total_bins):
                break
            stimuli_type = np.random.choice(p_type, size=1, replace=True, p=w_type)
            amp = np.random.choice(amplitudes, size=1)[0]
            # print(amp)

            if(stimuli_type==0): #flat tone
                freq = np.random.choice(discrete_freq.shape[0], size=1)
                freq_time_idx[0, t:t+stim_len] = freq[0].astype('int')
                full_spec[freq, t:t+stim_len] = amp
                freq = discrete_freq[freq][0]
                freq_time[0, t:t+stim_len] = freq
                amp_time[0, t:t+stim_len] = amp
            elif(stimuli_type == 1): #FM sweep
                fmsweep_len = fmsweep.shape[0]
                freq_time_idx[0, t:t+fmsweep_len] = fmsweep
                full_spec[fmsweep, t:t+stim_len] = amp
                freq_time[0, t:t+fmsweep_len] = discrete_freq[fmsweep]
                amp_time[0, t:t+fmsweep_len] = amp
            elif(stimuli_type==2): #white noise
                freq = np.random.choice(discrete_freq.shape[0], size=stim_len)
                full_spec[freq, t:t+stim_len] = amp
                freq_time_idx[0, t:t+stim_len] = freq.astype('int')
                freq_time[0, t:t+stim_len] = discrete_freq[freq]
                amp_time[0, t:t+stim_len] = amp

        ## creating spectrograms
        Xs = np.zeros((total_bins-num_timebins, num_freqbins, num_timebins))
        Ys = np.zeros((1, total_bins))
        linsums = []
        linsumsexp = []
        usedidxs = torch.tensor(np.arange(Xs.shape[0]), device=self.device)
        for t in range(Xs.shape[0]):
            Xs[t, freq_time_idx[0, t:t+num_timebins], :] = amp_time[0, t:t+num_timebins]
            linsum = np.sum(np.multiply(Xs[t,:,:], self.weights)) + bias # add noise
            Ys[0, t+num_timebins] = np.random.poisson(lam=np.exp(linsum), size=(1)) 
            linsums.append(linsum)
            linsumsexp.append(np.exp(linsum))
            # print("fr: ", np.exp(linsum), linsum, Ys[0,t+num_timebins])
        
        linsumsexp = np.asarray(linsumsexp)
        linsums = np.asarray(linsums)
        print("mean fr: ", np.mean(linsumsexp))
        print("total linsums: ", np.sum(linsums))

        return torch.tensor(Xs, device=self.device), torch.tensor(Ys, device=self.device),\
    usedidxs, full_spec

    def __len__(self):
        # print("useidxs", self.usedidxs.shape[0])
        return self.usedidxs.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx,:,:].type(torch.float32)
        # print(X.shape)
        Yhist = self.Y[0, idx+self.num_timebins-self.hist_len-1:idx+self.num_timebins-1]
        # print(Yhist.shape)
        return X, Yhist, torch.tensor([0.0]), self.Y[0, idx+self.num_timebins] 

    def __save_dataset(self, datafile):

    def __load_dataset(self, datafile):

