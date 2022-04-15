
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
from scipy.interpolate import interp1d
from sklearn.preprocessing import scale

# from dataloader import load_data, get_eventraster, get_stimulifreq_barebone
# from dataloader_strf import loaddata_withraster_strf
from utils import raster_fulltoevents, exponentialClass, spectral_resample, numpify
from plotting import plot_autocor, plot_neuronsummary, plot_utauests, plot_histdata,\
    plot_rasterpsth, plot_spectrogram, plot_strf

class Data_Generator_Base(Dataset):
    """
        Base class for all data generator classes
        Defines the functions for:
            >> loading the data
            >> creating the stimuli spectrogram from barebone input
            >> creating the stimuli spectroram from given parameters 
            >> creating the spike event raster from raw spoike data or pre-processed data
        Returns all values in pytorch tensors as a data.Dataset class of pytorch for easy loading
    """
    def __init__(self, PARAMS):
        self.PARAMS = PARAMS

    def __load_data(self):
        return None
    
    def __create_custom_filter(self):
        """ 
        Creates a custom filter for all stimuli response functions 
        """
        return None

    def __create_stimuli(self):
        """
        Given stimuli parameters, create the stimuli spectrogram
        """
        return None

    def __create_stimuli_from_barebone(self):
        """
        Given barebone stimuli spectrogram data, create the stimuli spectrogram
        """
        return None

    def __create_spiking_raster(self):
        return None

    def __default_params(self):
        return None

    def __save_dataset(self):
        return None
    
    def __load_dataset(self):
        return None


