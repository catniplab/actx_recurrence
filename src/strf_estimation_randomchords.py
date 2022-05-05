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

# from dataloader import load_data, get_eventraster, get_stimulifreq_barebone
from src.dataloaders.dataloader_strf import loaddata_withraster, STRF_Dataset_randomchords

# importing config
from src.default_cfg import get_cfg_defaults

from src.utils import raster_full_to_events, exponentialClass, spectral_resample, numpify,\
        check_and_create_dirs
from src.plotting import plot_autocor, plot_neuronsummary, plot_utauests, plot_histdata,\
    plot_rasterpsth, plot_spectrogram, plot_strf


class STRF_Estimation_GLM():
    def __init__(self, params):
        self.params = params
        self.device = params['device']
        self.frequency_bins = int((params['freqrange'][1]-params['freqrange'][0])\
                /params['freqbinsize'])
        self.time_bins = int((params['strf_timerange'][1]-params['strf_timerange'][0])\
                /params['strf_timebinsize'])
            
        self.strf_params = torch.tensor(np.random.normal(size=(self.frequency_bins,
            self.time_bins)), requires_grad=True, device = self.device, dtype=torch.float32)
        self.hist_bins = int(params['hist_size']/params['strf_timebinsize'])
        self.history_filter = torch.tensor(np.random.normal(size=(1, self.hist_bins)),
                requires_grad=True, device=self.device, dtype=torch.float32)
        val = np.random.uniform(-5.0, -1.0, 1) 
        self.bias = torch.tensor(val, requires_grad=True, device=self.device, dtype=torch.float)
        # self.optimizer = torch.optim.LBFGS([self.strf_params, self.history_filter, self.bias],\
                # lr=params['lr'])
        # minimizer_args = dict(method='Newton-CG', options={'disp':True, 'maxiter':10})
        minimizer_args = dict(method='TNC', options={'disp':False, 'maxiter':10})
        self.optimizer = MinimizeWrapper([self.strf_params, self.bias], minimizer_args)
        # self.optimizer = MinimizeWrapper([self.strf_params, self.history_filter, self.bias],\
                # minimizer_args)

    def run(self, dataloader):
        epochs = params['epochs']

        for e in range(epochs):
            print("Epoch: ", e)
            for ibatch, batchsample in tqdm(enumerate(dataloader)):
                Xin, Yhist, eta, Yt = batchsample

                def closure():
                    self.optimizer.zero_grad()
                    # temp = torch.nn.functional.conv2d(Xin.unsqueeze(1),
                            # self.strf_params.unsqueeze(0).unsqueeze(1), bias=None) 
                    # temp2 = torch.nn.functional.conv1d(Yhist.unsqueeze(1),\
                            # self.history_filter.unsqueeze(0), bias=None)

                    # linsum = torch.squeeze(torch.nn.functional.conv2d(Xin.unsqueeze(1),\
                        # self.strf_params.unsqueeze(0).unsqueeze(1), bias=None)) +\
                        # torch.squeeze(torch.nn.functional.conv1d(Yhist.unsqueeze(1),\
                        # self.history_filter.unsqueeze(0),bias=None)) +\
                        # self.bias[None, :] # + eta 

                    linsum = torch.squeeze(torch.nn.functional.conv2d(Xin.unsqueeze(1),\
                        self.strf_params.unsqueeze(0).unsqueeze(1), bias=None)) +\
                        self.bias[None, :]
                    
                    linsumexp = torch.exp(linsum)
                    # TODO: add torch clip on the linsumexp; do check how to select the range
                    linsumexp = torch.clamp(linsumexp, min=0.0001, max = 10000)
                    # print("is there inf in linsum exp? : ", torch.isinf(linsumexp).any())
                    if(torch.isinf(linsumexp).any()):
                        print("linsum :", linsum, linsum.shape)
                        print("limsumexp: ", linsumexp, linsumexp.shape)
                        # print("temp1: ", temp, temp.shape)
                        # print("temp2: ", temp2, temp2.shape)
                        # print("History: ", Yhist, Yhist.shape)

                    # print("linsumexp", linsumexp)
                    LLh = Yt*linsum - linsumexp #- (Yt+1).lgamma().exp()
                    # print("llh non reg: ", LLh)
                    # print("LLH:", LLh)
                    loss = torch.mean(-1*LLh)
                    loss += self.params['strf_reg']*torch.norm(self.strf_params, p=2)
                    # loss += self.params['history_reg']*torch.norm(self.history_filter, p=2)
                    # loss += self.params['strf_reg']*torch.norm(self.bias, p=2)
                    # print("loss before:", loss)
                    # print("strf weights:", self.strf_params, self.bias)
                    loss.backward()
                    if(ibatch%params['output_every']==0):
                        print("loss after {} epochs, {} batches = {}".format(e, ibatch, loss))
                    return loss

                loss = self.optimizer.step(closure)
                if(torch.isinf(self.strf_params).any() or torch.isnan(self.strf_params).any()):
                    print("strf weights have a nan or inf")

            # print("loss at epoch {} = {}; bias = {}".format(e, loss, numpify(self.bias)))

    def plotstrf(self, figloc):
        timebins = [i*self.params['strf_timebinsize']*1000 for i in range(self.time_bins)]
        freqbins = [(i/4000)*self.params['freqbinsize'] for i in range(self.frequency_bins)]
        # print(timebins, freqbins)
        print("bias: ", numpify(self.bias))
        plot_strf(numpify(self.strf_params), numpify(self.history_filter), timebins, freqbins, figloc)

    def save_weights(self, loc):
       with h5py.File(loc, 'w') as h5f :
           h5f.create_dataset('strf_weights', data = numpify(self.strf_params))
           h5f.create_dataset('hist_weights', data= numpify(self.history_filter))
           h5f.create_dataset('bias', data = numpify(self.bias))


def estimate_STRF(cfg, params, foldername, figloc, saveloc):
    #params
    binsize = params['binsize']#s = 20ms
    delayrange = params['delayrange']#units
    samplerate = params['samplerate']#samples per second
    sampletimespan = params['sampletimespan']
    minduration = params['minduration']

    stimuli_df, spikes_df, raster, rater_full, fmsdata =\
            loaddata_withraster(foldername, params)#fetch raw data

    strf_dataset = STRF_Dataset_randomchords(params, stimuli_df, spikes_df, fmsdata)
    strf_dataloader = DataLoader(strf_dataset, batch_size=params['batchsize'], shuffle=False,
            num_workers=4)
    strfest = STRF_Estimation_GLM(params)
    strfest.run(strf_dataloader)
    strfest.save_weights(saveloc)
    strfest.plotstrf(figloc)

def set_params():
    params = {}
    params['device'] = device

    params['binsize'] = 0.02#s = 20ms
    params['delayrange'] = [1, 30]#units
    params['samplerate'] = 10000#samples per second
    params['sampletimespan'] = [100, 300]
    params['minduration'] = 1.640
    params['rng'] = [0, 1.640]

    params['strf_timebinsize'] = 0.001 #s = 1ms
    params['strf_timerange'] = [0, 0.1] #s = 0 to 100ms
    params['freqrange'] = [0, 41000] #hz
    params['freqbinsize'] = 100 #hz/bin -- heuristic/random?

    params['hist_size'] = 0.02 #s -- ms
    params['max_amp'] = 100 #dB

    params['lr'] = 0.001
    params['batchsize'] = 32
    params['epochs'] = 1
    params['output_every'] = 50 # batches

    params['history_reg'] = 0.001
    params['strf_reg'] = 0.001
    
    return params

if(__name__=="__main__"):
    # torch params
    device = torch.device('cpu')

    # strf dataset
    foldername = "../../data/strf_data/"
    output_foldername = "../../outputs/strf_rc/"
    check_and_create_dirs(output_foldername)

    #params
    params = set_params()
    
    # configurations values
    cfg = get_cfg_defaults()
    cfg.freeze()

    # single datafile test
    foldername = "../data/strf_data/20210825-xxx999-002-001/"
    figloc = "../outputs/strf_rc/{}.pdf".format("20210825-xxx999-002-001")
    saveloc = output_foldername + "weights_{}.h5".format("20210825-xxx999-002-001")
    dataset_type = 'strf'

    # call the strf estimation function
    estimate_STRF(cfg, params, foldername, figloc, saveloc)

    # plot strf outputs
    # plot_strf_summary()

    # labels = []

    # foldernames = []
    # cortexsides = []
    # filenames = []
    # if(dataset_type=='prestrf'):
        # for dfs in datafiles:
            # for ctxs in cortexside:
                # fname = foldername.format(dfs, ctxs)
                # foldersinfname = os.listdir(fname)
                # for f in foldersinfname:
                    # foldernames.append(fname+f+'/')
                    # cortexsides.append(ctxs)
                    # filenames.append(f)
    # else:
        # foldersinfname = os.listdir(foldername)
        # for f in foldersinfname:
            # foldernames.append(foldername+f+'/')
            # cortexsides.append('NA')
            # filenames.append(f)

    # datafiles = {'folderloc':foldernames, 'label':cortexsides, 'filenames':filenames}
    # # print(datafiles)

    # for count, dfs in enumerate(datafiles['folderloc']):
        # figloc = "../outputs/strf_{}.pdf".format(datafiles['filenames'][count])
        # saveloc = "../checkpoints/weights_{}.h5".format(datafiles['filenames'][count])
        # print("dfs", dfs)
        # labels.append(datafiles['label'][count])
        # # figtitle = foldername
        # estimate_strf(dfs, dataset_type, params, figloc, saveloc)
        # print("label: ", datafiles['label'][count])
    
