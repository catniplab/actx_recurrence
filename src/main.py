import numpy as np
import os, pickle
from dataloader import loaddata_withraster
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class dichotomizedgaussian_surrogate():
    def __init__(self):
        self.a=0

    def dichotomizedgauss(self):
        self.a+=1

    def geenratesurrogate(self):
        return self.a

def autocorrelation(sig, delay):
    autocor = []
    for d in delay:
        #shift the signal
        sig_delayed = np.zeros_like(sig)
        if(d>0):
            sig_delayed[:, d:] = sig[:, 0:-d]
        else:
            sig_delayed = sig
        #calculate the correlation
        # Y_mean = np.mean(sig, 0)
        acr = np.sum(sig * sig_delayed, 0)/(sig.shape[1])
        acr = np.sum(acr, 0)/sig.shape[0]
        autocor.append(acr)
    return autocor

def resample(raster, raster_full, binsize, og_samplerate):
    newbinsize = binsize*og_samplerate#new sample bin size in previous sample rate
    new_raster_full = np.zeros((raster_full.shape[0], raster_full.shape[1]//int(newbinsize))) 
    # newbinsize = og_samplerate//binsize
    new_raster = []
    for i in range(new_raster_full.shape[0]):
        new_raster_tmp = []
        for j in range(len(raster[i])):
            new_raster_tmp.append(raster[i][j])
            new_raster_full[i, ((new_raster_tmp[j]*og_samplerate)/newbinsize).astype(int)]=1
        new_raster.append(new_raster_tmp)
    return new_raster, new_raster_full

def leastsquares_fit(autocor, delay, b):
    xdata = np.array(delay)
    exc_int = exponentialClass()
    exc_int.b = b
    optval, optcov = curve_fit(exc_int.exponential_func, xdata, autocor) 
    return optval

class exponentialClass:
    def __init__(self):
        self.b = 0

    def exponential_func(self, t, tau, a):
        return a * np.exp(-t/tau) + self.b

def plot_autocor(autocor, delay, a, b, tau):
    exc_int = exponentialClass()
    exc_int.b = b
    x_exponen = np.linspace(delay[0], delay[-1], 100)
    y_exponen = exc_int.exponential_func(x_exponen, tau, a)
    plt.plot(x_exponen, y_exponen, color='r')
    plt.plot(delay, autocor, color='b', marker='o')
    plt.xlabel('delay (s)')
    plt.ylabel('autocorrelation')
    plt.title('autocorrelation least squares fit')
    # plt.ylim((0, 1000))
    plt.show()

def calculate_meanfiringrate(raster, sampletime):
    mfs = []
    for i in range(len(raster)):
       mfs.append(len(raster[i])/(sampletime[1]-sampletime[0]))#mfs across time
    return np.mean(mfs)#mean across trials

def estimate_ogtau(foldername):
    #params
    binsize = 0.02#s = 20ms
    delayrange = [1, 20]#units
    samplerate = 10000#samples per second
    sampletimespan = [-0.5, 2]#s
    # sampletimespan *= 10 #100ms time units

    stimuli_df, spike_df, raster, raster_full = loaddata_withraster(foldername)#fetch raw data
    raster, raster_full = resample(raster, raster_full, binsize, 10000)#resize bins
    # delay = np.linspace(delayrange[0], delayrange[1], 20)
    delay = [i for i in range(delayrange[0], delayrange[1])]#range of delays
    mfr = calculate_meanfiringrate(raster, sampletimespan)#mean firing rate
    autocor = autocorrelation(raster_full, delay)#autocorr calculation
    b=(binsize*mfr)**2
    tau, a = leastsquares_fit(np.asarray(autocor), np.asarray(delay)*binsize, b)#least sq fit 
    print("mfr = {}, b = {}, a={}, tau={}".format(mfr, b, a, tau))
    plot_autocor(np.array(autocor), np.asarray(delay)*binsize, a, b, tau)#plotting autocorr
    return [a, b, tau, mfr, binsize, delay, binsize, autocor]

if(__name__=="__main__"):
    foldername = "..//data/ACx_data_{}/ACx{}/"
    datafiles = [1,2,3]
    cortexside = ["Calyx", "Thelo"]

    # foldername = "..//data/ACx_data_1/ACxCalyx/20080930-002/"
    foldername = "..//data/ACx_data_1/ACxCalyx/20170909-010/"
    outputs = estimate_ogtau(foldername)

    # foldernames = []
    # cortexsides = []
    # for dfs in datafiles:
        # for ctxs in cortexside:
            # fname = foldername.format(dfs, ctxs)
            # foldersinfname = os.listdir(fname)
            # for f in foldersinfname:
                # foldernames.append(fname+f+'/')
                # cortexsides.append(ctxs)

    # datafiles = {'folderloc':foldernames, 'label':cortexsides}
    # print(datafiles)

    # for count, dfs in enumerate(datafiles['folderloc']):
        # print("dfs", dfs)
        # estimations = estimate_ogtau(dfs)
        # print("label: ", datafiles['label'][count])


