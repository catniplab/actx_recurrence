import numpy as np
import os, pickle
import scipy
from scipy.optimize import curve_fit
from scipy.stats import norm, multivariate_normal
from dich_gauss.dichot_gauss import DichotGauss 
from dich_gauss.optim_dichot_gauss import get_bivargauss_cdf, find_root_bisection
import matplotlib.pyplot as plt

def raster_fulltoevents(raster_full, samplerate, sampletimespan):
    raster = []
    for i in range(raster_full.shape[0]):
        rowidx=np.array(np.nonzero(raster_full[i]))
        raster.append((rowidx/raster_full.shape[1])*(sampletimespan[1]-sampletimespan[0])+sampletimespan[0])
    return raster

def calculate_meanfiringrate(raster, sampletime):
    mfs = []
    for i in range(len(raster)):
       mfs.append(len(raster[i])/(sampletime[1]-sampletime[0]))#mfs across time
    return np.mean(mfs)#mean across trials

def calculate_fanofactor(isi, raster, samplerate, binsize):
    ## mean of fano factor on multiple trials
    # fanof = (np.var(data, ddof=1)/np.mean(data))
    fanofs = []
    print("isi len", len(isi))
    for i in range(len(isi)):
        fanofi = np.var(isi[i],ddof=1)/(np.mean(isi[i])+1e-8)
        if(not np.isnan(fanofi)):
            fanofs.append(fanofi)
    # fanof = np.sum(fanofs)/len(isi)
    fanof = np.mean(fanofs)
    # print(fanofs)
    return fanof 

class exponentialClass:
    def __init__(self):
        self.b = 0

    def exponential_func(self, t, tau, a):
        return a * np.exp(-t/tau) + self.b

def measure_isi(raster):
    isi_list = []
    isis_list = []
    for i in range(len(raster)):
        spktimesi = raster[i]
        isii =  np.asarray(spktimesi[1:]) - np.asarray(spktimesi[0:-1])
        isi_list.extend(isii.tolist())
        isis_list.append(isii)
    return isi_list, isis_list

def measure_psth(raster_full, binsizet, period, samplerate):
    binsize = int(samplerate*binsizet)
    totalbins = int((period*samplerate)/binsize)
    print(period, binsize, totalbins, raster_full.shape)
    normspikesperbin = []
    for i in range(totalbins):
        binslice = raster_full[:, i*binsize:(i+1)*binsize]
        # print(binslice.shape)
        spikecount = np.sum(binslice)
        normspikesperbin.append(spikecount/(raster_full.shape[0]*binsize))
    return normspikesperbin


