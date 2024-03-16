import sys
import yaml
import copy
from easydict import EasyDict

_C = EasyDict()

_C.TAG = 'default'

_C.TRAINING = EasyDict()
_C.TRAINING.EPOCHS = 20
_C.TRAINING.BATCHSIZE = 4
_C.TRAINING.N_WORKERS = 8
_C.TRAINING.PRECISION = 16
_C.TRAINING.DT = 0.1        # time step in seconds
_C.TRAINING.HORIZON = 500   # horizon in number of points
_C.TRAINING.GAMMA = 0.998
_C.TRAINING.VIS_INTERVAL = 50
_C.TRAINING.VERBOSE = True

# model parameters
_C.MODEL = EasyDict()
_C.MODEL.ENCODER = 'resnet50'
_C.MODEL.LOAD_NETWORK = None
_C.MODEL.DOWNSAMPLE = 8
_C.MODEL.LATENT_DIM = 64
_C.MODEL.TIME_LENGTH = 3
_C.MODEL.PREDICT_DEPTH = True
_C.MODEL.TRAIN_DEPTH = True
_C.MODEL.FUSE_PCLOUD = True
_C.MODEL.ENC_3_LAYERS = True
_C.MODEL.INPUT_SIZE = (320, 180)
_C.MODEL.GRID_BOUNDS = {
    'xbound': [-2.0, 8.0, 0.1],
    'ybound': [-5.0, 5.0, 0.1],
    'zbound': [-2.0, 2.0, 0.1],
    'dbound': [ 0.3, 8.0, 0.2]}

# training parameters
_C.OPTIMIZER = EasyDict()
_C.OPTIMIZER.LR = 3e-4
_C.OPTIMIZER.WEIGHT_DECAY = 1e-7

_C.DATASET = EasyDict()
_C.DATASET.TRAIN_DATA = ['../dataset/zed2/data_train', '../dataset/realsense/data_train']
_C.DATASET.VALID_DATA = ['../dataset/zed2/data_valid', '../dataset/realsense/data_valid']
_C.DATASET.CSV_FILE = 'rosbags.csv'

_C.AUGMENTATIONS = EasyDict()
_C.AUGMENTATIONS.HORIZ_FLIP = 0.5         # horizontal flip augmentation
_C.AUGMENTATIONS.PCLOUD_DROPOUT = 0.3     # probability to drop the pointcloud input

_C.CONTROL = EasyDict()
# _C.CONTROL.precision = 16
_C.CONTROL.dt = 0.5
_C.CONTROL.real_dt = 0.05
_C.CONTROL.samples = 2048   # N_SAMPLES
_C.CONTROL.horizon = 16     # TIMESTEPS
_C.CONTROL.u_min = [-0.7, -3]
_C.CONTROL.u_max = [0.7, 3]
_C.CONTROL.noise_mu = [0.0, 0.0]
_C.CONTROL.noise_sigma = [0.2, 1.0]
_C.CONTROL.action_dim = 2
_C.CONTROL.gamma = 50.0
_C.CONTROL.Q = [5, 5, 1]
_C.CONTROL.Qf = [10, 10, 2]
_C.CONTROL.R = [500, 0.1]
_C.CONTROL.W = [0, 0]

# Set randomization seed
_C.SEED = 42

def merge_cfgs(base_cfg, new_cfg):
    config = copy.deepcopy(base_cfg)
    for key, val in new_cfg.items():
        if key in config:
            if type(config[key]) is EasyDict:
                config[key] = merge_cfgs(config[key], val)
            else:
                config[key] = val
        else:
            sys.exit("key {} doesn't exist in the default configs".format(key))

    return config

def get_cfg(cfg_file):
    cfg = copy.deepcopy(_C)

    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

    cfg = merge_cfgs(cfg, new_config)

    return cfg