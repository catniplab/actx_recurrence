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

class exponentialClass:
    def __init__(self):
        self.b = 0

    def exponential_func(self, t, tau, a):
        return a * np.exp(-t/tau) + self.b

def measure_isi(raster):
    isi_list = []
    for i in range(len(raster)):
        spktimesi = raster[i]
        isii =  np.asarray(spktimesi[1:]) - np.asarray(spktimesi[0:-1])
        isi_list.extend(isii.tolist())
    return isi_list

def measure_psth(raster_full, binsizet, period, samplerate):
    binsize = int(samplerate*binsizet)
    totalbins = int((period*samplerate)/binsize)
    normspikesperbin = []
    for i in range(totalbins):
        binslice = raster_full[:, i*binsize:(i+1)*binsize]
        spikecount = np.sum(binslice)
        normspikesperbin.append(spikecount/(raster_full.shape[0]*binsize))
    return normspikesperbin


