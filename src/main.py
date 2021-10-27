import numpy as np
import os, pickle
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

from dich_gauss.dichot_gauss import DichotGauss 
from dich_gauss.optim_dichot_gauss import get_bivargauss_cdf, find_root_bisection
from dataloader import loaddata_withraster
from dataloader_strf import loaddata_withraster_strf
from utils import raster_fulltoevents, calculate_meanfiringrate, exponentialClass,\
    measure_isi, measure_psth
from plotting import plot_autocor, plot_neuronsummary, plot_utauests, plot_histdata

class dichotomizedgaussian_surrogate():
    def __init__(self, mfr, autocorr, data, delay):
        self.data = data
        self.gauss_mean = self.calculate_gmean(mfr) #recheck mfr formulation in the code
        self.gauss_cov = self.calculate_gcov(mfr, autocorr, delay)
        # print("gauss mean and cov", self.gauss_mean, self.gauss_corr)

    def calculate_gmean(self, mfr):
        # mfr[mfr==0.0]+=1e-4
        return norm.ppf(mfr)

    def calculate_gcov(self, mfr, autocorr, delay):
        gauss_cov = np.zeros(len(delay)+1)
        for d in range(len(delay)):
            data_cov = np.eye(2)
            data_cov[1,0], data_cov[0,1] = autocorr[d], autocorr[d]
            x = find_root_bisection([mfr, mfr], [self.gauss_mean, self.gauss_mean], data_cov[1,0])
            # gauss_cov[1,0], gauss_cov[0,1] = x, x
            gauss_cov[d+1]=x
        gauss_cov = scipy.linalg.toeplitz(gauss_cov)
        return gauss_cov

    def dichotomized_gauss(self):
        mean = np.repeat(self.gauss_mean, self.gauss_cov.shape[0])
        gen_dichgauss = np.random.multivariate_normal(mean=mean,
                cov=self.gauss_cov, size=self.data.shape)
        gen_data = np.zeros_like(gen_dichgauss)
        gen_data[gen_dichgauss>0]=1
        gen_data[gen_dichgauss<=0]=0
        # print("size of gen data", gen_data.shape)
        self.gen_data = gen_data
        return gen_data

    def dich_autocorrelation(self, data, delay):
        autocor = []
        for i in range(len(delay)):
            acr = np.sum(data[:,:,0]*data[:,:,1+i],0)/(data.shape[1])
            acr = np.sum(acr, 0)/data.shape[0]
            autocor.append(acr)
        return autocor

    def estimate_tau(self, binsize, samplerate, delayrange, sampletimespan):
        raster = raster_fulltoevents(self.gen_data, samplerate, sampletimespan)
        # delay = np.linspace(delayrange[0], delayrange[1], 20)
        delay = [i for i in range(delayrange[0], delayrange[1])]#range of delays
        mfr = calculate_meanfiringrate(raster, sampletimespan)#mean firing rate
        autocor = self.dich_autocorrelation(self.gen_data, delay)#autocorr calculation
        b=(binsize*mfr)**2
        tau, a = leastsquares_fit(np.asarray(autocor), np.asarray(delay)*binsize, b)#least sq fit 
        # plot_autocor(np.array(autocor), np.asarray(delay)*binsize, a, b, tau)#plotting autocorr
        # print("mfr = {}, b = {}, a={}, tau={}".format(mfr, b, a, tau))
        return tau, a

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


def estimate_ogtau(foldername):
    #params
    binsize = 0.02#s = 20ms
    delayrange = [1, 20]#units
    samplerate = 10000#samples per second
    sampletimespan = [-0.5, 2]#s
    # sampletimespan *= 10 #100ms time units

    stimuli_df, spike_df, raster, raster_full = loaddata_withraster_strf(foldername)#fetch raw data

    # measure isi and psth before resampling
    isi_list = measure_isi(raster)
    psth_measure = measure_psth(raster_full, binsize, sampletimespan[1]-sampletimespan[0],
            samplerate)

    # resampling with wider bins and fitting exponential curve to og data
    raster, raster_full = resample(raster, raster_full, binsize, samplerate)#resize bins
    # delay = np.linspace(delayrange[0], delayrange[1], 20)
    delay = [i for i in range(delayrange[0], delayrange[1])]#range of delays
    mfr = calculate_meanfiringrate(raster, sampletimespan)#mean firing rate
    rv_mean = np.mean(raster_full)
    autocor = autocorrelation(raster_full, delay)#autocorr calculation
    b=(binsize*mfr)**2
    tau, a = leastsquares_fit(np.asarray(autocor), np.asarray(delay)*binsize, b)#least sq fit 
    print("mfr = {}, b = {}, a={}, tau={}".format(mfr, b, a, tau))
    ogest = [a, b, tau]

    # plot_autocor(np.array(autocor), np.asarray(delay)*binsize, a, b, tau)#plotting autocorr
    # return [a, b, tau, mfr, binsize, delay, binsize, autocor]

    ## create surrogate and estimate unbiased tau
    surrogate_taus = []
    surrogate_as = []
    surr_iters = 400
    for i in range(surr_iters):
        dgauss_surr = dichotomizedgaussian_surrogate(rv_mean, autocor, raster_full, delay)
        _ = dgauss_surr.dichotomized_gauss()
        tau_est, a_est = dgauss_surr.estimate_tau(binsize, samplerate, delayrange, sampletimespan)
        surrogate_taus.append(tau_est)
        surrogate_as.append(a_est)

    # check surrogate a's distribution by plotting it's hist
    # plot_histdata(surrogate_as)

    surrogate_taus = np.array(surrogate_taus)
    params = scipy.stats.lognorm.fit(surrogate_taus)
    bias = params[0]-np.log(tau)
    std_tau = params[1]
    # print(np.exp(params[0]))
    logunbiasedtau = np.log(tau) - bias
    a_est = np.mean(surrogate_as)

    # intergrate the posteriors
    dichgaussest = [a_est, b, np.exp(logunbiasedtau), std_tau]
    print("dich gaus estimates a={}, b={}, tau={}, std={}, bias={}".format(a_est, b,
        np.exp(logunbiasedtau), std_tau, bias))

    return raster, raster_full, isi_list, psth_measure, np.asarray(delay)*binsize, autocor, mfr, ogest, dichgaussest



if(__name__=="__main__"):
    # prestrf dataset
    foldername = "..//data/prestrf_data/ACx_data_{}/ACx{}/"
    datafiles = [1,2,3]
    cortexside = ["Calyx", "Thelo"]

    #strf dataset
    # foldername = "../data/prestrf_data/ACx_data_{}/ACx{}/"
    # datafiles = [1,2,3]
    # cortexside = ["Calyx", "Thelo"]

    # foldername = "../data/prestrf_data/ACx_data_1/ACxCalyx/20170909-010/"
    foldername = "../data/strf_data/20210825-xxx999-002-001/"
    # figloc = "../outputs/{}.png".format("20170909-010")
    figloc = "../outputs/{}.png".format("20210825-xxx999-002-001")
    raster, raster_full, isi_list, psth_measure, delay, autocor, mfr, ogest, dichgaussest =\
        estimate_ogtau(foldername)
    plot_neuronsummary(autocor, delay, raster, isi_list, psth_measure, ogest, dichgaussest,\
            foldername, figloc)

    dichgaussests = []
    labels = []
    mfrs = []
    dichgaussests.append(dichgaussest)
    labels.append("Calyx")
    mfrs.append(mfr)
    figloc = "../outputs/ests_{}.png".format("20170909-010")
    plot_utauests(dichgaussests, mfrs, labels, figloc)

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
    # # print(datafiles)

    # for count, dfs in enumerate(datafiles['folderloc']):
        # print("dfs", dfs)
        # try:
            # estimations = estimate_ogtau(dfs)
        # except:
            # print("found exception")
        # print("label: ", datafiles['label'][count])


