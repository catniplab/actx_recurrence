import numpy as np
import matplotlib.pyplot as plt
import os, pickle
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import seaborn

from utils import exponentialClass

def plot_strf(strfweights, historyweights, timebinst, freqbins, figloc):
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    print(strfweights.shape)
    # ax[0].imshow(np.flip(strfweights, 1), cmap='hot', interpolation='none')
    ax[0].pcolormesh(timebinst, freqbins, strfweights, cmap='hot')
    ax[0].set_xlabel('lag')
    ax[0].set_ylabel('frequncy bin')

    ## plot history weights
    # print(historyweights.shape)
    # print(np.flip(historyweights, 0))
    ax[1].plot(np.flip(historyweights,0)[0], 'go--')
    ax[1].set_xlabel('lag')
    ax[1].set_ylabel('amplitude')

    plt.savefig(figloc, bbox_inches='tight')
    plt.close()

def plot_spectrogram(spectrogram, figloc):
    plt.pcolormesh(np.transpose(spectrogram))
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.savefig(figloc)
    plt.close()

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

def plot_raster(event_data):
    # print("event data shape:", event_data.shape)
    plt.eventplot(event_data, linelengths=0.9, linewidth = 0.4)
    plt.xlabel("time")
    plt.ylabel("event count")
    plt.show()

def plot_waveform(timestamp, recording):
    plt.plot(timestamp, recording)
    plt.xlabel("timestamp")
    plt.ylabel("voltage")
    plt.show()

def plot_rasterpsth(raster, psth, isi, figloc, binsizet, rng, fanof):
    fig, ax = plt.subplots(1, 3, figsize=(30,10))

    # plot raster
    ax[0].eventplot(raster, linelengths=0.9, linewidth = 0.6)
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("event count")

    # plot PSTH
    psthx = [i*binsizet+rng[0]*1000 for i in range(len(psth))]
    ax[1].bar(psthx, psth, alpha=0.5, align='edge', width=binsizet)
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("count")
    y_ = gaussian_filter1d(psth, 3)
    ax[1].plot(psthx, y_, '-', color='r')

    #plot ISI hist
    ax[2].hist(isi, bins=30)
    ax[2].set_xlabel("spike time diff")
    ax[2].set_ylabel("count")

    fig.suptitle("Fano factor = {}".format(fanof))
    plt.savefig(figloc, bbox_inches='tight')
    plt.close()

def plot_neuronsummary(autocorr, params, raster, isi, psth, og_est, dichgaus_est, title, figloc):
    # plot 4 subfigures - 
        # raster
        # ISI
        # psth
        # autocorr; og_est; dichgauss_est

    #params
    delayrange = [i for i in range(params['delayrange'][0], params['delayrange'][1])]
    delay = np.asarray(delayrange)*params['binsize']
    binsize = params['binsize']
    rng = params['sampletimespan']
    samplerate = params['samplerate']

    # plot raster
    fig, ax = plt.subplots(1, 4, figsize=(40,10))
    ax[0].eventplot(raster, linelengths=0.9, linewidth = 0.6)
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("event count")

    # plot ISI 
    ax[1].hist(isi, bins=1000)
    ax[1].set_xlabel("spike time diff")
    ax[1].set_ylabel("count")

    # plot PSTH
    psthx = [i*binsize for i in range(len(psth))]
    # psthx = [i for i in range(len(psth))]
    ax[2].bar(psthx, psth)
    ax[2].set_xlabel("spike counts")
    ax[2].set_ylabel("count")

    # plot autocorss
    ax[3].plot(delay, autocorr, 'b')
    a,b,tau = og_est
    exc_int = exponentialClass()
    exc_int.b = b
    x_exponen = np.linspace(delayrange[0], delayrange[-1], 100)*binsize
    y_exponen = exc_int.exponential_func(x_exponen, tau, a)
    ax[3].plot(x_exponen, y_exponen, 'r')

    if(dichgaus_est is not None):
        a, b, tau, std = dichgaus_est
        exc_int = exponentialClass()
        exc_int.b = b
        x_exponen = np.linspace(delay[0], delay[-1], 100)
        y_exponen = exc_int.exponential_func(x_exponen, tau, a)
        y_exponenpstd = exc_int.exponential_func(x_exponen, tau+std, a)
        y_exponenmstd = exc_int.exponential_func(x_exponen, tau-std, a)
        ax[3].plot(x_exponen, y_exponen, 'g')
        ax[3].fill_between(x_exponen, y_exponen - y_exponenmstd, y_exponen + y_exponenpstd,
                     color='g', alpha=0.2)

    ax[3].set_ylabel("autocorrelation value")
    ax[3].set_xlabel("delay")
    ax[3].set_ylim(bottom=0.0)

    fig.suptitle(title)
    plt.savefig(figloc, bbox_inches='tight')
    plt.close()

def plot_utauests(estimates, mfrs, labels, figloc):
    fig = plt.figure(figsize=(10,10))
    for i in range(len(estimates)):
        _, _, tau, std = estimates[i]
        color = None
        if labels[i]=="Calyx":
            color = 'g'
        else:
            color = 'r'
        # plt.plot(mfrs[i], tau, color=color, marker = 'o')
        plt.errorbar(mfrs[i], tau, yerr=std, fmt='o', color=color, elinewidth=2, capsize=5,
                ecolor=color)
    plt.xlabel("mean firing rate")
    plt.ylabel("time constant")
    plt.savefig(figloc, bbox_inches='tight')

def plot_histdata(data):
    plt.figure(figsize=(10,10))
    plt.hist(data, bins=50)
    plt.show()

def multichannel_waveform_plot(recordings):
    fig, ax = plt.subplots(5,5)
    toshow = random.sample(range(len(recordings)), 25)
    for i in range(5):
        for j in range(5):
            y = recordings[i*5 + j]
            x = [i for i in range(len(y))]
            ax[i,j].plot(x, y)
            # ax[i,j].xlabel("timestamp")
            # ax[i,j].ylabel("voltage")
    plt.show()
