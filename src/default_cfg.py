"""
Default config file for all experiments
"""

from yacs.config import CfgNode as CN
import io
import yaml

_C = CN()

_C.DATASET = CN()
_C.DATASET.binsize = 0.02#s = 20ms
_C.DATASET.strf_timebinsize = 0.001#s = 1ms
_C.DATASET.strf_timerange = [0, 0.1] #s - 0 to 250ms
_C.DATASET.delayrange = [1, 30]#units
_C.DATASET.samplerate = 10000#samples per second
_C.DATASET.sampletimespan = [0, 150] #sec
_C.DATASET.minduration = 1.640
_C.DATASET.freqrange = [0, 41000]
_C.DATASET.freqbinsize = 100 #hz/bin -- heuristic/random?
_C.DATASET.hist_size = 0.02 #s = 20ms
_C.DATASET.max_amp = 100 #db

_C.TRAIN = CN()
_C.TRAIN.lr = 0.01
_C.TRAIN.batchsize = 32
_C.TRAIN.epochs = 1

_C.REGULARIZE = CN()
_C.REGULARIZE.history_reg = 0.001
_C.REGULARIZE.strf_reg = 0.001

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()


# #params
# params = {}
# params['binsize'] = 0.02#s = 20ms
# params['delayrange'] = [1, 300]#units
# params['samplerate'] = 10000#samples per second
# # sampletimespan = [0, 1.640]#s
# params['sampletimespan'] = [100, 300]
# params['minduration'] = 1.640
# # sampletimespan *= 10 #100ms time units
