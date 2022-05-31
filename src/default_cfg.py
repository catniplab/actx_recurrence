"""
Default config file for all experiments
"""

from yacs.config import CfgNode as CN
import io
import yaml

_C = CN()

_C.DATASET = CN()
_C.DATASET.binsize = 0.02 #s = 20ms
_C.DATASET.foldername = "../data/prestrf_data/ACx_data_{}/ACx{}/"
_C.DATASET.datafiles = [1,2,3]
_C.DATASET.cortexside = ["Calyx", "Thelo"]
_C.DATASET.dataset_type = 'prestrf'

_C.DATASET.delayrange = [0, 40] # bins unit; for autocorrelation check.
_C.DATASET.samplerate = 10000 # samples per second
_C.DATASET.trial_minduration = 1.640
_C.DATASET.window_range = [0.1, 1.640] #s
_C.DATASET.del_time = 0.001 #s

# for a single trial dataset
_C.DATASET.sampletimespan = [0, 1.640]#s
# _C.DATASET.sampletimespan = [0, 150] #sec
_C.DATASET.psth_binsize = 0.02 #s

# for STRF analysis
_C.DATASET.strf_timebinsize = 0.001#s = 1ms
_C.DATASET.strf_timerange = [0, 0.1] #s - 0 to 250ms
_C.DATASET.freqrange = [0, 41000]
_C.DATASET.freqbinsize = 100 #hz/bin -- heuristic/random?
_C.DATASET.hist_size = 0.02 #s = 20ms
_C.DATASET.max_amp = 100 #db


_C.TRAIN = CN()
_C.TRAIN.lr = 0.01
_C.TRAIN.batchsize = 32
_C.TRAIN.epochs = 1
# _C.TRAIN.tau0 = [1e-3, 1e-2, 1e-1, 1, 10, 10e2, 10e3, 10e4, 10e5, 10e6, 10e7]
_C.TRAIN.tau0 = [1e-3, 1e-2, 1e-1, 1, 10, 10e2, 10e3, 10e4]
_C.TRAIN.a0 = [1e-3, 1e-2, 1e-1, 1, 10, 10e2, 10e3, 10e4]

_C.REGULARIZE = CN()
_C.REGULARIZE.history_reg = 0.001
_C.REGULARIZE.strf_reg = 0.001

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()


