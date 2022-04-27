import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import os, pickle
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
from matplotlib.collections import PolyCollection
from utils import exponentialClass

def plot_psd(yf, xf, path):
    plt.plot(xf, yf)
    plt.grid()
    plt.savefig(path)
    plt.close()

def plot_psds_3d(psds, freqs, labels, params, figloc):
    left_idx = [i for i,x in enumerate(labels) if\
            (x=="Calyx" and not np.isnan(np.array(psds[i])).any())] 
    right_idx = [i for i,x in enumerate(labels) if\
            (x=="Thelo" and not np.isnan(np.array(psds[i])).any())] 

    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    norm = mpl.colors.Normalize(vmin=0, vmax=50)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap.set_array([])
    cmap2 = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.YlOrRd)
    cmap2.set_array([])

    zs = []
    verts=[]
    fcolors = []
    print(len(freqs), freqs[0].shape)
    print(len(psds), psds[0].shape)
    for i in range(len(left_idx)):
        # zs.append(i)
        # verts.append(list(zip(freqs[left_idx[i]], psds[left_idx[i]])))
        # fcolors.append(cmap.to_rgba(80))
        ax.plot(freqs[left_idx[i]], psds[left_idx[i]], zs=i, zdir='z', alpha=0.6,\
                color=cmap.to_rgba(30 + 2*i))
    # poly = PolyCollection(verts, facecolors=fcolors)
    # poly.set_alpha(0.5)
    # ax.add_collection3d(poly, zs=zs)

    ax.set_xlabel('frequencies (Hz)')
    ax.set_ylabel('power spectrum density (dB)')
    ax.set_zlabel('Neuron Number')
    ax.set_zlim3d(0, len(labels)+1)
    ax.set_ylim3d(-20, 0)
    ax.set_xlim3d(0, 30)
    # ax.set_xticks(xticks)
    ax.set_title('left hemisphere')

    plt.show()


def plot_psds(psds, freqs, labels, params, figloc):
    left_idx = [i for i,x in enumerate(labels) if x=="Calyx"] 
    right_idx = [i for i,x in enumerate(labels) if x=="Thelo"] 
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    stagger_ct = [0,0]
    stg_frac = 0.2
    norm = mpl.colors.Normalize(vmin=0, vmax=50)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap.set_array([])
    cmap2 = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.YlOrRd)
    cmap2.set_array([])
    xticks = [i for i in range(0, int(freqs[0][-1]+4), 5)]

    for i in range(len(left_idx)):
        # stg_val_x = np.min(freqs[left_idx[i]]) * stg_frac
        # stg_val_y = np.mean(psds[left_idx[i]]) * stg_frac
        # stagger_ct[0]+=stg_val_x
        # stagger_ct[0]+= 0.25
        # stagger_ct[1]+=stg_val_y
        log_psd = psds[left_idx[i]]+stagger_ct[1]
        ax.plot(freqs[left_idx[i]]+stagger_ct[0], log_psd,\
                color="#4f94c4", alpha=0.6)
        # ax.fill_between(freqs[left_idx[i]]+stagger_ct[0],\
                # stagger_ct[1] * np.ones(freqs[left_idx[i]].size),\
                # log_psd, alpha=0.3, color=cmap.to_rgba(i+1))
    ax.set_xlabel('frequencies (Hz)')
    ax.set_ylabel('power spectrum density (dB)')
    ax.set_xticks(xticks)
    ax.set_title('left hemisphere')

    stagger_ct = [0,0]
    for i in range(len(right_idx)):
        # stg_val_x = np.min(freqs[right_idx[i]]) * stg_frac
        # stg_val_y = np.mean(psds[right_idx[i]]) * stg_frac
        # stagger_ct[0]+=stg_val_x
        # stagger_ct[0]+= 0.25
        # stagger_ct[1]+=stg_val_y
        ax2.plot(freqs[right_idx[i]]+stagger_ct[0], psds[right_idx[i]]+stagger_ct[1],\
                color="#ff851a", alpha=0.6)
        # ax2.fill_between(freqs[right_idx[i]]+stagger_ct[0],\
                # stagger_ct[1] * np.ones(freqs[right_idx[i]].size),\
                # psds[right_idx[i]]+stagger_ct[1], alpha=0.3, color=cmap2.to_rgba(i+1))
    ax2.set_xlabel('frequencies (Hz)')
    ax2.set_ylabel('power spectrum density (dB)')
    ax2.set_xticks(xticks)
    ax2.set_title('right hemisphere')

    # save the plot
    # fig.tight_layout()
    fig.savefig(figloc, dpi=120)

def plot_strf(strfweights, historyweights, timebinst, freqbins, timebin, freqs, figloc):
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    # print(strfweights.shape)
    # print(freqs, timebin)
    # ax[0].imshow(np.flip(strfweights, 1), cmap='hot', interpolation='none')
    # ax[0].pcolormesh(timebinst, freqbins, strfweights, cmap='seismic', shading='gouraud', vmin=-1,
            # vmax=1)
    ax[0].pcolormesh(timebin, freqs, strfweights, cmap='seismic', shading='gouraud', vmin=-1,
            vmax=1)
    ax[0].set_xlabel('lag (ms)')
    ax[0].set_ylabel('frequncy bin')
    # ax[0].set_xticks(timebin)
    # ax[0].set_yticks(freqs)

    ## plot history weights
    # print(historyweights.shape)
    # print(np.flip(historyweights, 0))
    ax[1].plot(np.flip(historyweights,0)[0], 'go--')
    ax[1].set_xlabel('lag')
    ax[1].set_ylabel('amplitude')
    
    plt.autoscale(tight=True)
    plt.savefig(figloc, bbox_inches='tight')
    plt.close()

def plot_spectrogram(spectrogram, figloc):
    # plt.pcolormesh(np.transpose(spectrogram))
    plt.pcolormesh(spectrogram)
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

def plot_filter(weights, delays_sec, freqs, params):
    print("weights:", weights.shape)
    kwargs = dict(vmax=np.abs(weights).max(), vmin=-np.abs(weights).max(),
                  cmap='RdBu_r', shading='gouraud')
    fig, ax = plt.subplots()
    ax.pcolormesh(delays_sec, freqs, weights, **kwargs)
    ax.set(title='Simulated STRF', xlabel='Time Lags (s)', ylabel='Frequency (Hz)')
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.autoscale(tight=True)
    # TODO: add a figure saving code
    plt.show()

