import numpy as np
import os, pickle, random, math
import scipy.io, scipy
import pandas as pd

import matplotlib.pyplot as plt
from plotting import plot_raster
import Data_Loading

class Data_Loading_FMSweep(Data_Loading):
    def __init__(self, PARAMS, foldername):
        self.PARAMS = PARAMS


def loaddata_withraster(foldername, PARAMS):
    stimuli_df, spike_df = load_data(foldername, PARAMS)

    # raster, raster_full = get_sorted_event_raster(stimuli_df, spike_df, rng)
    raster, raster_full = get_event_raster(stimuli_df, spike_df, PARAMS)
    return stimuli_df, spike_df, raster, raster_full

if (__name__ == "__main__"):
    PARAMS = {
        rng = [-0.5, 2], 
        sample_rate = 10000, 
        minduration = 1.640 #s
        }
    foldername = "..//data/ACx_data_1/ACxCalyx/20080930-002/"
    stimuli_df, spike_df = load_data(foldername, PARAMS)
    raster, raster_full = sortnplot_eventraster(stimuli_df, spike_df, PARAMS)
    plot_raster(raster)
 
