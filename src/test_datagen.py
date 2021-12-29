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

import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge
from sklearn.preprocessing import scale

from dataloader import load_data, get_eventraster, get_stimulifreq_barebone
from dataloader_strf import loaddata_withraster_strf
from utils import raster_fulltoevents, exponentialClass, spectral_resample, numpify
from plotting import plot_autocor, plot_neuronsummary, plot_utauests, plot_histdata,\
    plot_rasterpsth, plot_spectrogram, plot_strf

np.random.seed(100)
torch.manual_seed(100)

class TestDataset(Dataset):
    def __init__(self, params):
        self.device = params['device']
        self.params = params
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

    def create_filter(self, params):
        # ## create a filter 
        # weights = self.sample_filter(sfreq)
        ## https://mne.tools/dev/auto_tutorials/machine-learning/30_strf.html
        sfreq = params['samplerate']
        n_freqs = params['freq_bins']
        tmin, tmax = self.params['time_span']

        # To simulate the data we'll create explicit delays here
        delays_samp = np.arange(np.round(tmin * sfreq),
                                np.round(tmax * sfreq)).astype(int)
        delays_sec = delays_samp / sfreq
        freqs = np.linspace(50, 5000, n_freqs)
        grid = np.array(np.meshgrid(delays_sec, freqs))

        # We need data to be shaped as n_epochs, n_features, n_times, so swap axes here
        grid = grid.swapaxes(0, -1).swapaxes(0, 1)

        # Simulate a temporal receptive field with a Gabor filter
        means_high = [.1, 1500]
        means_low = [.2, 4000]
        cov = [[.001, 0], [0, 500000]]
        gauss_high = multivariate_normal.pdf(grid, means_high, cov)
        gauss_low = -1 * multivariate_normal.pdf(grid, means_low, cov)
        print("min max of gauss high:", np.min(gauss_high), np.max(gauss_high))
        print("min max of gauss low:", np.min(gauss_low), np.max(gauss_low))
        weights = 10*(gauss_high + gauss_low)  # Combine to create the "true" STRF
        print("min max of gauss total:", np.min(weights), np.max(weights))
        # print("weights:", weights.shape)
        # kwargs = dict(vmax=np.abs(weights).max(), vmin=-np.abs(weights).max(),
                      # cmap='RdBu_r', shading='gouraud')
        # fig, ax = plt.subplots()
        # ax.pcolormesh(delays_sec, freqs, weights, **kwargs)
        # ax.set(title='Simulated STRF', xlabel='Time Lags (s)', ylabel='Frequency (Hz)')
        # plt.setp(ax.get_xticklabels(), rotation=45)
        # plt.autoscale(tight=True)
        # plt.show()
        return weights

    def stimuli_nme(self, weights):
        rng = np.random.RandomState(1337)

        ## test_stimuli -- load data from mne dataset
        path_audio = mne.datasets.mtrf.data_path()
        data = loadmat(path_audio + '/speech_data.mat')
        audio = data['spectrogram'].T
        print("OG audio shape:", audio.shape)
        sfreq = float(data['Fs'][0, 0])
        print("sfreq", sfreq)
        n_decim = self.params['n_decim']
        audio = mne.filter.resample(audio, down=n_decim, npad='auto')
        sfreq /= n_decim

        ## create a delay spectrogram
        # Reshape audio to split into epochs, then make epochs the first dimension.
        n_epochs, n_seconds = 1, 5*16
        audio = audio[:, :int(n_seconds * sfreq * n_epochs)]
        print("audio :", audio.shape)
        X = audio.reshape([n_freqs, n_epochs, -1]).swapaxes(0, 1)
        print("X: ", X.shape)
        X = np.expand_dims(scale(X[0]), 0)
        n_times = X.shape[-1]

        # Delay the spectrogram according to delays so it can be combined w/ the STRF
        # Lags will now be in axis 1, then we reshape to vectorize
        delays = np.arange(np.round(tmin * sfreq),
                           np.round(tmax * sfreq) + 1).astype(int)
        print("delays", len(delays), tmin, tmax)

        # Iterate through indices and append
        X_del = np.zeros((len(delays),) + X.shape)
        for ii, ix_delay in enumerate(delays):
            # These arrays will take/put particular indices in the data
            take = [slice(None)] * X.ndim
            put = [slice(None)] * X.ndim
            if ix_delay > 0:
                take[-1] = slice(None, -ix_delay)
                put[-1] = slice(ix_delay, None)
            elif ix_delay < 0:
                take[-1] = slice(-ix_delay, None)
                put[-1] = slice(None, ix_delay)
            X_del[ii][tuple(put)] = X[tuple(take)]

        # Now set the delayed axis to the 2nd dimension
        X_del = np.rollaxis(X_del, 0, 3)
        print("xdel shape", X_del.shape)
        X_del_return = torch.tensor(X_del, dtype=torch.float32, device=self.device)[0,:]
        print("xdel self shape", self.X_del.shape)
        X_del = X_del.reshape([n_epochs, -1, n_times])
        print("xdel shape", X_del.shape)
        n_features = X_del.shape[1]
        weights_sim = weights.ravel()

        # Simulate a neural response to the sound, given this STRF
        y = np.zeros((n_epochs, n_times))
        y_poss = np.zeros((n_epochs, n_times))
        for ii, iep in enumerate(X_del):
            # Simulate this epoch and add random noise
            noise_amp = .002
            y[ii] = np.dot(weights_sim, iep) + noise_amp * rng.randn(n_times)
            # print(y[ii].shape, np.exp(y[ii]).shape) 
            y_poss[ii] = np.random.poisson(lam=np.exp(y[ii]), size=(1, n_times)) 

        print("yii poss", y_poss)
        Y_return = torch.tensor(y_poss, dtype=torch.float32, device=self.device)
        print("non zero y", np.count_nonzero(y_poss[0]))
        usedidxs = torch.tensor([i for i in range(0, y.shape[1]-self.strf_bins)],
                device=self.device)
        return X_del_return, Y_return, usedidxs
    
    def calculate_rank(self, X):
        X_flat = torch.flatten(X, start_dim = 1).T
        X_flat = numpify(X_flat)
        print("x flat shape:", X_flat.shape)
        X_cov = np.cov(X_flat)
        print("x cov shape:", X_cov.shape)
        X_cov_rank = np.linalg.matrix_rank(X_cov)
        print("x cov rank:", X_cov_rank)

    def create_stimuli(self):
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

class strfestimation():
    def __init__(self, params):
        self.params = params
        self.device = params['device']
        self.frequency_bins = self.params['freq_bins']
        self.time_bins = self.params['time_bins']
        self.strf_params = torch.tensor(np.random.normal(size=(self.frequency_bins,
            self.time_bins)), requires_grad=True, device = self.device, dtype=torch.float32)
        # self.strf_params = torch.tensor(np.random.normal(size=(self.frequency_bins,
            # self.time_bins)), device = self.device, dtype=torch.float32)
        # self.hist_bins = int(params['hist_size']/params['strf_timebinsize'])
        self.history_filter = torch.tensor(np.random.normal(size=(1, self.params['hist_len'])),
                requires_grad=True, device=self.device, dtype=torch.float32)
        val = np.random.uniform(-5.0, -1.0, 1) 
        self.bias = torch.tensor(val, requires_grad=True, device=self.device, dtype=torch.float)
        # self.bias = torch.randn(1, device=self.device)
        # self.optimizer = torch.optim.LBFGS([self.strf_params, self.history_filter, self.bias], lr=params['lr'])
        self.optimizer_linreg = torch.optim.SGD([self.strf_params, self.bias], lr=params['lr'])
        # minimizer_args = dict(method='Newton-CG', options={'disp':True, 'maxiter':2})
        minimizer_args = dict(method='TNC', options={'disp':False, 'maxiter':10})
        self.optimizer = MinimizeWrapper([self.strf_params, self.bias], minimizer_args)
        # self.optimizer = MinimizeWrapper([self.strf_params, self.history_filter, self.bias], minimizer_args)

    def linreg_fit(self, dataloader):
        print("fitting weights on linear regression")
        for ibatch, batchsample in tqdm(enumerate(dataloader)):
            Xin, Yhist, eta, Yt = batchsample
            Yt = Yt.type('torch.FloatTensor')
            self.optimizer_linreg.zero_grad()
            linsum = torch.squeeze(torch.nn.functional.conv2d(Xin.unsqueeze(1),
                self.strf_params.unsqueeze(0).unsqueeze(1), bias=None)) + self.bias[None,:] 
            # print("linsum: ", linsum)
            lmbda = torch.exp(linsum)[0,:] ## firing rate
            lmbda = torch.clamp(lmbda, max=1e+3)
            # print("in lin reg fit:", lmbda, Yt)
            loss = torch.nn.functional.mse_loss(lmbda, Yt) 
            # print("loss: ", loss)
            # break
            
            loss += self.params['strf_reg']*torch.norm(self.strf_params, p=2)
            loss.backward()
            self.optimizer_linreg.step()
            if(ibatch%100==0):
                print("loss for lin reg fitting : {}".format(loss))
                # print(self.strf_params, self.bias)


    def run(self, dataloader):
        epochs = params['epochs']

        for e in range(epochs):
            print("Epoch: ", e)
            for ibatch, batchsample in tqdm(enumerate(dataloader)):
                Xin, Yhist, eta, Yt = batchsample
                # print(Xin.shape, Yt.shape, Yhist.shape)

                def closure():
                    self.optimizer.zero_grad()
                    temp = torch.nn.functional.conv2d(Xin.unsqueeze(1),
                            self.strf_params.unsqueeze(0).unsqueeze(1), bias=None) 
                    # temp2 = torch.nn.functional.conv1d(Yhist.unsqueeze(1),\
                            # self.history_filter.unsqueeze(0), bias=None)
                    linsum = torch.squeeze(torch.nn.functional.conv2d(Xin.unsqueeze(1),\
                        self.strf_params.unsqueeze(0).unsqueeze(1), bias=None)) +\
                        self.bias[None, :] # + eta 
                    
                    # print("linsum:", linsum)
                    linsumexp = torch.exp(linsum)
                    # print("is there inf in linsum exp? : ", torch.isinf(linsumexp).any())
                    if(torch.isinf(linsumexp).any()):
                        print("linsum :", linsum, linsum.shape)
                        print("limsumexp: ", linsumexp, linsumexp.shape)
                        # print("temp1: ", temp, temp.shape)
                        # print("temp2: ", temp2, temp2.shape)
                        # print("History: ", Yhist, Yhist.shape)

                    # print("linsumexp", linsumexp)
                    LLh = Yt*linsum - linsumexp #- (Yt+1).lgamma().exp()
                    # print("LLH:", LLh)
                    loss = torch.mean(-1*LLh)
                    loss += self.params['strf_reg']*torch.norm(self.strf_params, p=2)
                    # loss += self.params['strf_reg']*torch.norm(self.bias, p=2)
                    # loss += self.params['history_reg']*torch.norm(self.history_filter, p=2)

                    # print("loss before:", loss)
                    # print("strf weights:", self.strf_params, self.bias)

                    if(ibatch%200==0):
                        print("loss at epoch {}, batch {} = {}".format(e, ibatch, loss))
                    loss.backward()
                    return loss

                loss = self.optimizer.step(closure)
                # loss = closure()
                if(torch.isinf(self.strf_params).any() or torch.isnan(self.strf_params).any()):
                    print("strf weights have a nan or inf")
                # print("loss after:", loss)

            # print("loss at epoch {} = {}; bias = {}".format(e, loss, numpify(self.bias)))

    def plotstrf(self, figloc):
        timebins = [i for i in range(self.time_bins)]
        freqbins = [i for i in range(self.frequency_bins)]
        # print(timebins, freqbins)
        print("bias: ", numpify(self.bias))
        print("strf weights: ", numpify(self.strf_params))
        freqs = np.linspace(50, 5000, self.frequency_bins)
        timebinst = np.array(timebins)*(1000.0/self.params['samplerate'])
        strf_vals = numpify(self.strf_params)
        strf_vals = strf_vals/(np.max(np.absolute(strf_vals)))

        plot_strf(strf_vals, numpify(self.history_filter), timebins, freqbins, timebinst, freqs, figloc)

    def save_weights(self, loc):
        with h5py.File(loc, 'w') as h5f :
           h5f.create_dataset('strf_weights', data = numpify(self.strf_params))
           h5f.create_dataset('hist_weights', data= numpify(self.history_filter))
           h5f.create_dataset('bias', data = numpify(self.bias))

    def load_weights(self, loc):
        with h5py.File(loc, 'r') as f:
            a_group_key = list(f.keys())[0]
            self.strf_params = f['strf_weights']
            self.history_filter = f['hist_weights']
            self.bias = f['bias']

def estimate_strf(params, figloc, saveloc):
    testdata = TestDataset(params)
    testdataloader = DataLoader(testdata, batch_size=params['batchsize'], shuffle=False,
            num_workers=6)
    strfest = strfestimation(params)

    # weights_loc = '../testweights_121621_112447.h5'
    # strfest.load_weights(weights_loc)
    # strfest.plotstrf(figloc)

    strfest.linreg_fit(testdataloader)
    # strfest.run(testdataloader)
    # strfest.plotstrf(figloc)
    # strfest.save_weights(saveloc)

if(__name__=="__main__"):
    # torch params
    device = torch.device('cpu')

    # prestrf dataset
    foldername = "../data/prestrf_data/ACx_data_{}/ACx{}/"
    datafiles = [1,2,3]
    cortexside = ["Calyx", "Thelo"]
    dataset_type = 'prestrf'
    #params
    params = {}

    n_decim = 2
    # sfreq = 128.0/n_decim
    sfreq=100
    params['samplerate'] = int(sfreq) #samples per second
    params['n_decim'] = n_decim
    params['binsize'] = 0.02#s = 20ms
    params['freq_bins'] = 20 # total bins
    params['time_span'] = [0.0, 0.3]
    tmin, tmax = params['time_span']
    params['time_bins'] = len(np.arange(np.round(tmin * sfreq), np.round(tmax * sfreq)).astype(int))
    # print(len(params['time_bins']))
    params['hist_len'] = 15

    params['strf_timebinsize'] = 0.001#s = 1ms
    # params['strf_timerange'] = [0, 0.25] #s - 0 to 250ms
    params['strf_timerange'] = [0, 0.1] #s - 0 to 250ms
    params['delayrange'] = [1, 30]#units
    params['sampletimespan'] = [0, 100] #sec
    params['minduration'] = 1.640
    params['freqrange'] = [50, 5000]
    params['freqbins'] = 20 #hz/bin -- heuristic/random?
    params['hist_size'] = 0.02 #s = 20ms
    params['max_amp'] = 100 #db

    params['lr'] = 0.01
    params['device'] = device
    params['batchsize'] = 128
    params['epochs'] = 2

    #regularization params
    params['history_reg'] = 0.001
    params['strf_reg'] = 1.0

    dtnow = datetime.now()
    dtnow = dtnow.strftime("%m%d%y_%H%M%S")

    figloc = "../outputs/teststrf_{}.pdf".format(dtnow)
    saveloc = "../checkpoints/testweights_{}.h5".format(dtnow)
    estimate_strf(params, figloc, saveloc)
    # true_strf = testdata.sample_filter()
