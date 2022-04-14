import numpy as np
import os, pickle, random, math
import scipy.io, scipy
import pandas as pd

import matplotlib.pyplot as plt
from plotting import plot_raster
from dataloader_base import Data_Loading

class Data_Loading_FMSweep(Data_Loading):
    def __init__(self, PARAMS, foldername):
        super().__init__(PARAMS, foldername)


def loaddata_withraster(foldername, PARAMS):
    fmsdata = Data_Loading_FMSweep(PARAMS, foldername)

    # raster, raster_full = get_sorted_event_raster(stimuli_df, spike_df, rng)
    raster, raster_full = fmsdata.get_event_raster(fmsdata.stimuli_df, fmsdata.spike_data_df, PARAMS)
    return fmsdata.stimuli_df, fmsdata.spike_data_df, raster, raster_full

if (__name__ == "__main__"):
    PARAMS = {
            'rng': [-0.5, 2], 
            'sample_rate': 10000, 
            'minduration': 1.640 #s
        }
    foldername = "../../data/prestrf_data/ACx_data_1/ACxCalyx/20080930-002/"
    stimuli_df, spike_df, raster, raster_full = loaddata_withraster(foldername, PARAMS)
    plot_raster(raster)
 
