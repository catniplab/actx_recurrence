import numpy as np
import matplotlib.pyplot as plt
import os, pickle

from utils import exponentialClass

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

def plot_neuronsummary(autocorr, delay, raster, isi, psth, og_est, dichgaus_est, title, figloc):
    # plot 4 subfigures - 
        # raster
        # ISI
        # psth
        # autocorr; og_est; dichgauss_est

    # plot raster
    fig, ax = plt.subplots(1, 4, figsize=(40,10))
    ax[0].eventplot(raster, linelengths=0.9, linewidth = 0.4)
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("event count")

    # plot ISI 
    ax[1].hist(isi, bins=10)
    ax[1].set_xlabel("spike time diff")
    ax[1].set_ylabel("count")

    # plot PSTH
    psthx = [i for i in range(len(psth))]
    ax[2].bar(psthx, psth)
    ax[2].set_xlabel("spike counts")
    ax[2].set_ylabel("count")

    # plot autocorss
    ax[3].plot(delay, autocorr, 'b')
    
    a,b,tau = og_est
    exc_int = exponentialClass()
    exc_int.b = b
    x_exponen = np.linspace(delay[0], delay[-1], 100)
    y_exponen = exc_int.exponential_func(x_exponen, tau, a)
    ax[3].plot(x_exponen, y_exponen, 'r')

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

    ax[3].set_xlabel("autocorrelation value")
    ax[3].set_ylabel("delay")
    ax[3].set_ylim(bottom=0.0)

    fig.suptitle(title)
    plt.savefig(figloc, bbox_inches='tight')

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

