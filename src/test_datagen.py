import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from tqdm import tqdm
import h5py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_minimize.optim import MinimizeWrapper
from scipy.stats import multivariate_normal, poisson
from scipy.io import loadmat

import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge
from sklearn.preprocessing import scale

from dataloader import load_data, get_eventraster, get_stimulifreq_barebone
from dataloader_strf import loaddata_withraster_strf
from utils import raster_fulltoevents, exponentialClass, spectral_resample, numpify
from plotting import plot_autocor, plot_neuronsummary, plot_utauests, plot_histdata,\
    plot_rasterpsth, plot_spectrogram, plot_strf

class TestDataset(Dataset):
    def __init__(self, params):
        self.device = params['device']
        self.params = params
        rng = np.random.RandomState(1337)
        self.strf_bins = int(self.params['strf_timerange'][1]/self.params['strf_timebinsize'])
        self.hist_len = self.params['hist_len']

        ## test_stimuli -- load data from mne dataset
        path_audio = mne.datasets.mtrf.data_path()
        data = loadmat(path_audio + '/speech_data.mat')
        audio = data['spectrogram'].T
        print(audio.shape)
        sfreq = float(data['Fs'][0, 0])
        print("sfreq", sfreq)
        n_decim = self.params['n_decim']
        audio = mne.filter.resample(audio, down=n_decim, npad='auto')
        sfreq /= n_decim

        # ## create a filter 
        # weights = self.sample_filter(sfreq)

        ## https://mne.tools/dev/auto_tutorials/machine-learning/30_strf.html
        n_freqs = params['freq_bins']
        tmin, tmax = self.params['time_span']

        # To simulate the data we'll create explicit delays here
        delays_samp = np.arange(np.round(tmin * sfreq),
                                np.round(tmax * sfreq) + 1).astype(int)
        delays_sec = delays_samp / sfreq
        freqs = np.linspace(50, 5000, n_freqs)
        grid = np.array(np.meshgrid(delays_sec, freqs))

        # We need data to be shaped as n_epochs, n_features, n_times, so swap axes here
        grid = grid.swapaxes(0, -1).swapaxes(0, 1)

        # Simulate a temporal receptive field with a Gabor filter
        means_high = [.1, 500]
        means_low = [.2, 2500]
        cov = [[.001, 0], [0, 500000]]
        gauss_high = multivariate_normal.pdf(grid, means_high, cov)
        gauss_low = -1 * multivariate_normal.pdf(grid, means_low, cov)
        weights = 1*(gauss_high + gauss_low)  # Combine to create the "true" STRF

        # kwargs = dict(vmax=np.abs(weights).max(), vmin=-np.abs(weights).max(),
                      # cmap='RdBu_r', shading='gouraud')
        # fig, ax = plt.subplots()
        # ax.pcolormesh(delays_sec, freqs, weights, **kwargs)
        # ax.set(title='Simulated STRF', xlabel='Time Lags (s)', ylabel='Frequency (Hz)')
        # plt.setp(ax.get_xticklabels(), rotation=45)
        # plt.autoscale(tight=True)
        # plt.plot()
        # plt.show()

        print("weights:", weights.shape)
        
        ## create a delay spectrogram
        # Reshape audio to split into epochs, then make epochs the first dimension.
        n_epochs, n_seconds = 1, 5*16
        audio = audio[:, :int(n_seconds * sfreq * n_epochs)]
        print("audio :", audio.shape)
        X = audio.reshape([n_freqs, n_epochs, -1]).swapaxes(0, 1)
        print("X: ", X.shape)
        X = np.expand_dims(scale(X[0]), 0)
        # print("X :", X)
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
        self.X_del = torch.tensor(X_del, dtype=torch.float32, device=self.device)[0,:]
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

        # X_plt = scale(np.hstack(X[:2]).T).T
        # y_plt = scale(np.hstack(y[:2]))
        # time = np.arange(X_plt.shape[-1]) / sfreq
        # _, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        # ax1.pcolormesh(time, freqs, X_plt, vmin=0, vmax=4, cmap='Reds',
                       # shading='gouraud')
        # ax1.set_title('Input auditory features')
        # ax1.set(ylim=[freqs.min(), freqs.max()], ylabel='Frequency (Hz)')
        # ax2.plot(time, y_plt)
        # ax2.set(xlim=[time.min(), time.max()], title='Simulated response',
        # xlabel='Time (s)', ylabel='Activity (a.u.)')
        # plt.show()

        # print("yii", y)
        print("yii poss", y_poss)
        self.Y = torch.tensor(y_poss, dtype=torch.float32, device=self.device)
        print("non zero y", np.count_nonzero(y_poss[0]))
        self.usedidxs = torch.tensor([i for i in range(0, y.shape[1]-self.strf_bins)],
                device=self.device)

    def __len__(self):
        # print("useidxs", self.usedidxs.shape[0])
        return self.usedidxs.shape[0]

    def __getitem__(self, idx):
        X = self.X_del[:,:, idx+self.strf_bins]
        # print(X.shape)
        Yhist = self.Y[0, idx+self.strf_bins-self.hist_len-1:idx+self.strf_bins-1]
        # print(Yhist.shape)
        return X, Yhist, torch.tensor([0.0]), self.Y[0, idx+self.strf_bins] 

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
        self.bias = torch.randn(1, requires_grad=True, device=self.device)
        # self.bias = torch.randn(1, device=self.device)
        # self.optimizer = torch.optim.LBFGS([self.strf_params, self.history_filter, self.bias], lr=params['lr'])
        minimizer_args = dict(method='Newton-CG', options={'disp':True, 'maxiter':10})
        # minimizer_args = dict(method='TNC', options={'disp':False, 'maxiter':10})
        self.optimizer = MinimizeWrapper([self.strf_params, self.bias], minimizer_args)
        # self.optimizer = MinimizeWrapper([self.strf_params, self.history_filter, self.bias], minimizer_args)

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
                    # linsum = torch.squeeze(torch.nn.functional.conv2d(Xin.unsqueeze(1),\
                        # self.strf_params.unsqueeze(0).unsqueeze(1), bias=None)) +\
                        # torch.squeeze(torch.nn.functional.conv1d(Yhist.unsqueeze(1), self.history_filter.unsqueeze(0),\
                        # bias=None)) +  self.bias[None, :] # + eta 
                    
                    # print("linsum:", linsum)
                    linsumexp = torch.exp(linsum)
                    # print("is there inf in linsum exp? : ", torch.isinf(linsumexp).any())
                    if(torch.isinf(linsumexp).any()):
                        print("linsum :", linsum, linsum.shape)
                        print("limsumexp: ", linsumexp, linsumexp.shape)
                        print("temp1: ", temp, temp.shape)
                        # print("temp2: ", temp2, temp2.shape)
                        # print("History: ", Yhist, Yhist.shape)

                    # print("linsumexp", linsumexp)
                    LLh = Yt*linsum - linsumexp #- (Yt+1).lgamma().exp()
                    # print("LLH:", LLh)
                    loss = torch.mean(-1*LLh)
                    loss += self.params['strf_reg']*torch.norm(self.strf_params, p=2)
                    loss += self.params['strf_reg']*torch.norm(self.bias, p=2)
                    # loss += self.params['history_reg']*torch.norm(self.history_filter, p=2)

                    # gradient_strf = torch.mul(Yt[:,None,None], Xin) - torch.mul(Xin,\
                        # linsumexp[0][:, None, None]) + (1/(self.params['strf_reg']*\
                        # torch.norm(self.strf_params, p=2)))*self.strf_params
                    # gradient_strf_batch = torch.mean(gradient_strf, 0)

                    # gradient_bias = Yt - linsumexp[0]
                    # gradient_bias_batch = torch.mean(gradient_bias, 0)

                    # hessian_strf = -1 * torch.multiply(torch.square(Xin), linsumexp[0][:,None,None])
                    # hessian_strf_batch = torch.mean(hessian_strf, 0) + 1e-8
                    # hessian_bias = -1 * linsumexp[0]
                    # hessian_bias_batch = torch.mean(hessian_bias, 0) + 1e-8

                    # delta_strf = torch.div(gradient_strf_batch, hessian_strf_batch)
                    # delta_bias = torch.div(gradient_bias_batch, hessian_bias_batch)

                    # self.strf_params -= delta_strf
                    # self.bias -= delta_bias

                    if(torch.isinf(self.strf_params).any() or torch.isnan(self.strf_params).any()):
                        print("strf weights have a nan or inf")
                        print("strf params: ", self.strf_params)
                        print("grad strf: ", gradient_strf)
                        print("hessian strf: ", hessian_strf)

                    # print("loss before:", loss)
                    # print("strf weights:", self.strf_params, self.bias)
                    if(ibatch%100==0):
                        print("loss as epoch {}, batch {} = {}".format(e, ibatch, loss))
                    # loss.backward()
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
        plot_strf(numpify(self.strf_params),
                numpify(self.history_filter), timebins, freqbins, figloc)

    def save_weights(self, loc):
       with h5py.File(loc, 'w') as h5f :
           h5f.create_dataset('strf_weights', data = numpify(self.strf_params))
           h5f.create_dataset('hist_weights', data= numpify(self.history_filter))
           h5f.create_dataset('bias', data = numpify(self.bias))

def estimate_strf(params, figloc, saveloc):
    testdata = TestDataset(params)
    testdataloader = DataLoader(testdata, batch_size=params['batchsize'], shuffle=False,
            num_workers=4)
    strfest = strfestimation(params)
    strfest.run(testdataloader)
    strfest.plotstrf(figloc)
    strfest.save_weights(saveloc)

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
    sfreq = 128.0/n_decim
    params['samplerate'] = int(sfreq) #samples per second
    params['n_decim'] = n_decim
    params['binsize'] = 0.02#s = 20ms
    params['freq_bins'] = 20 # total bins
    params['time_span'] = [0.0, 0.5]
    tmin, tmax = params['time_span']
    params['time_bins'] = len(np.arange(np.round(tmin * sfreq), np.round(tmax * sfreq) +\
        1).astype(int))
    # print(len(params['time_bins']))
    params['hist_len'] = 12

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
    params['epochs'] = 100

    #regularization params
    params['history_reg'] = 0.001
    params['strf_reg'] = 0.0001

    figloc = "../outputs/teststrf.pdf"
    saveloc = "../checkpoints/testweights.h5"
    estimate_strf(params, figloc, saveloc)
    # true_strf = testdata.sample_filter()