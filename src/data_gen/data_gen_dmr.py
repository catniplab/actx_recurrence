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

np.random.seed(100)
torch.manual_seed(100)

class Gen_Data_DynamicMovingRipple(Dataset):
    def __init__(self):
        self.params = self.dataset_params()
        self.dmr_params = self.dmr_params()
        self.filter = self.create_filter()
        # self.device = self.params['device']
        self.seed = self.dmr_params['seed']
        self.total_stimuli, self.total_response = self.create_stimuli_wholetrial()
        print('min: ', np.min(self.total_stimuli), 'max :', np.max(self.total_stimuli))

    def dataset_params(self):
        params = {
            'samplingrate' : 1000,
            'binsize' : 1, #ms
            'envduration': 100, #ms
            'total_time': 30, #s
            # 'device' : 'cuda:0',
            'omega_sr': 3, #Hz
            'fomega_sr': 6, #Hz
            'bias' : -1
        }
        return params

    def dmr_params(self):
        dmr_consts = {
            'f1' : 200,  ##Lowest carrier frequency
            'f2' : 48000,  ##Maximum carrier frequency
            'fRD' : 1.5,  ##Maximum rate of change for RD
            'fFM' : 3,  ##Maximum rate of change for FM
            'MaxRD' : 4,  ##Maximum ripple density (cycles/oct)
            'MaxFM' : 50,  ##Maximum temporal modulation rate (Hz)
            'App' : 45,  ##Peak to peak amplitude of the ripple in dB
            'Fs' : 200e3,  ##Sampling rate
            'NCarriersPerOctave' : 10,
            'NB' : 1,  ##For NB' : 1 genarets DMR
            'Axis' : 'log',
            'Block' : 'n',
            # 'DF' : round(Fs/1000),   ##Sampling rate for envelope single is 3kHz (48kHz/16)
            'AmpDist' : 'dB',
            'seed' : 789,
            'sigmax' : 3.0,
            'sigmat' : 3.0,
        }
        M = dmr_consts['Fs']*60*self.params['total_time']  ##5 minute long sounds
        NS  = np.ceil(dmr_consts['NCarriersPerOctave']*np.log2(dmr_consts['f2']/dmr_consts['f1']))  ##Number of sinusoid carriers. ~100 sinusoids / octave
        dmr_consts['M'] = M
        dmr_consts['NS'] = NS
        dmr_consts['freq_bins'] = int(NS)
        dmr_consts['time_bins'] = int(self.params['envduration']*\
                (self.params['samplingrate']/1000)*(1/self.params['binsize']))
        return dmr_consts
    
    def create_filter(self):
        timebins = self.dmr_params['time_bins']
        freq_maxoct = np.log2(self.dmr_params['f2']/self.dmr_params['f1'])
        freqs = np.linspace(0, freq_maxoct, int(self.dmr_params['NS']))
        times = np.linspace(0, self.params['envduration'], timebins)
        grid = np.array(np.meshgrid(times, freqs))
        # print(grid.shape)

        # We need data to be shaped as n_epochs, n_features, n_times, so swap axes here
        grid = grid.swapaxes(0, -1).swapaxes(0, 1)
        # Simulate a temporal receptive field with a Gabor filter
        means_high = [20, 1.5]
        means_low = [40, 3]
        cov = [[10, 0], [0, 3]]
        gauss_high = multivariate_normal.pdf(grid, means_high, cov)
        gauss_low = -1 * multivariate_normal.pdf(grid, means_low, cov)
        # print("min max of gauss high:", np.min(gauss_high), np.max(gauss_high))
        # print("min max of gauss low:", np.min(gauss_low), np.max(gauss_low))
        weights = 3*(gauss_high + gauss_low)  # Combine to create the "true" STRF
        return weights

    def plot_twoenvandfilter(self, env, filt):
        timebins = self.dmr_params['time_bins']
        freq_maxoct = np.log2(self.dmr_params['f2']/self.dmr_params['f1'])
        freqs = np.linspace(0, freq_maxoct, int(self.dmr_params['NS']))
        times = np.linspace(0, self.params['envduration'], timebins)
        kwargs = dict(cmap='RdBu_r', shading='gouraud')
        fig, ax = plt.subplots(2, 1)
        ax[0].pcolormesh(times, freqs, filt, **kwargs)
        ax[0].set(title='Simulated STRF', xlabel='Time Lags (s)', ylabel='Frequency (Hz)')
        ax[1].pcolormesh(times, freqs, env, **kwargs)
        ax[1].set(title='DMR envelope', xlabel='Time Lags (s)', ylabel='Frequency (Hz)')
        # plt.setp(ax.get_xticklabels(), rotation=45)
        plt.autoscale(tight=True)
        plt.show()

    def smooth_walk(self, points, dur):
        f = interp1d(np.linspace(0, dur, len(points)), np.reshape(points, (points.shape[0], )), "cubic")
        return f(np.linspace(0, dur, dur * self.params['samplingrate']))

    def create_stimuli_wholetrial(self):
        phi = np.random.rand(1)*np.pi*2
        total_bins = int(self.params['samplingrate']*\
                self.params['total_time']/self.params['binsize'])
        freq_maxoct = np.log2(self.dmr_params['f2']/self.dmr_params['f1'])

        S = np.zeros((int(self.dmr_params['NS']), int(self.params['samplingrate']\
                    *self.params['total_time']/self.params['binsize'])))
        Ws_list = self.dmr_params['MaxRD']*np.random.randn(\
                int(self.params['omega_sr']*self.params['total_time']), 1)
        Fws_list = self.dmr_params['MaxFM']*np.random.randn(\
                int(self.params['fomega_sr']*self.params['total_time']), 1)

        ## interpolate the Ws and Fws
        Ws = self.smooth_walk(Ws_list, self.params['total_time'])
        Fws = self.smooth_walk(Fws_list, self.params['total_time'])

        ## Frequency vector 
        xs = np.linspace(0, freq_maxoct, int(self.dmr_params['NS']))
        ts = np.linspace(0, self.params['total_time'],\
                self.params['samplingrate']*self.params['total_time'])

        ## calculate the whole stimuli in one go using the interpolated parameters
        Xs = np.tile(xs, (total_bins, 1)).T # F x T; xs~(F,)
        Ts = np.tile(ts, (xs.shape[0], 1))
        Fws = np.tile(Fws, (xs.shape[0], 1)) # F x T
        Ws = np.tile(Ws, (xs.shape[0], 1)) # F x T

        S = (self.dmr_params['App']/20)*np.sin(2*np.pi*Ws*Xs + 2*np.pi*Fws*Ts + phi)
        S_norm = (1/np.sqrt(np.pi*self.dmr_params['sigmax']*self.dmr_params['sigmat']))*\
                np.exp((-(Ts**2)/2)/(self.dmr_params['sigmat']**2))*\
                np.exp((-(Xs**2)/2)/(self.dmr_params['sigmax']**2))

        # total_stimuli = 1 * (0.5 + (S * S_norm)/2)
        total_stimuli = (S * S_norm)
        # total_stimuli = S/2 + 0.5
        # print(np.min(total_bins), np.max(total_bins))
        total_response = []
        length = int(self.params['total_time']*self.params['samplingrate']/self.params['binsize']) -\
            int(self.params['envduration']) + 1 #time in ms
        ## calculating response vector
        for i in range(0, length):
            stim_env = total_stimuli[:, i:i+int(self.params['envduration'])]
            response_linsum = np.sum(self.filter * stim_env) + self.params['bias']
            # if(i == 40):
                # print(stim_env)
                # print(self.filter)
                # print(stim_env * self.filter)
                # print(response_linsum)
                # self.plot_twoenvandfilter(stim_env, self.filter)
            response = np.random.poisson(lam=np.exp(response_linsum), size=(1))
            total_response.append(response)

        avg_fr = np.sum(total_response)/self.params['total_time']
        print('firing rate: ', avg_fr)
        print('max spikes per bin', np.max(total_response))
        return total_stimuli, total_response

    def __len__(self):
        length = int(self.params['total_time']*self.params['samplingrate']/self.params['binsize']) -\
            int(self.params['envduration']) + 1 #time in ms
        # print(length)
        return length

    def __getitem__(self, idx):
        stim_env = self.total_stimuli[:, int(idx):int(idx)+int(self.params['envduration'])]
        res = self.total_response[idx]
        return torch.tensor(stim_env, dtype=torch.float32),\
                torch.tensor([0.0], dtype=torch.float32),\
                torch.tensor([0.0], dtype=torch.float32), res
    
    def _plot_envelope_(self, figloc):
        # plt.pcolormesh(2*self.total_stimuli[:, 0:500]-1, vmin=-1, vmax=1, cmap='seismic')
        plt.pcolormesh(self.total_stimuli[:, 0:500], vmin=-1, vmax=1, cmap='seismic')
        plt.xlabel('time (ms)')
        plt.ylabel('frequency (octaves)')
        plt.colorbar()
        # plt.savefig(figloc)
        # plt.close()
        # plt.plot()
        plt.show()

if __name__ == "__main__":
    gen_dmr = Gen_Data_DynamicMovingRipple()
    gen_dmr._plot_envelope_(None)
