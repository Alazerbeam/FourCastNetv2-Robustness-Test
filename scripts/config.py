from utils import randomize_seeds
from track_hurricane import lat_to_index, lon_to_index
import os

# TODO: choose which steps to take
RETRIEVE_DATA = True         # noise mode only
PREDICT = True
VISUALIZE_LOCAL = True        # noise mode only
VISUALIZE_GLOBAL = True
PLOT_ERROR_LOCAL = True      # noise mode only
PLOT_ERROR_GLOBAL = True
CLEAN_UP = False

# TODO: choose number of timesteps, directories, channels to visualize
TIMESTEPS = 15                  # number of timesteps to run
HOME_PATH = "/your/path/here"   # home directory to contain data and plots
MODEL_DIR = "/your/path/here"   # directory which contains model information: weights.tar, global_means.npy, global_stds.npy, metadata.json
ERA5_STATS_DIR = "/your/path/here"  # directory which contains the original global_means.npy and global_stds.npy
CHANNELS = ["msl", "u10m", "v10m"]  # list of channels which you want to visualize

# NOISE MODE ONLY
# TODO: choose which noise levels to test, how many experiments per noise level, and how many timesteps
NOISE_PCTS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]  # list of noise percents to test
NUM_EXPERIMENTS = 30    # number of trials to run per noise percent

# set limits on area (lats/lons) of local data
# for latitude, degrees north is positive and south is negative
# for longitude, degrees east is positive and west is negative
# example: hurricane florence is in area (40N, 30N, 90W, 70W) ==> (40, 30, -90, -70)
NORTHMOST_LAT = 40
SOUTHMOST_LAT = 30
WESTMOST_LON = -90
EASTMOST_LON = -70

# RANDOM MODE ONLY
# choose which distributions to generate initial conditions from
DISTRIBUTIONS = ["normal", "chi-sq", "lognormal", "uniform"]

# =================================================================================================

# convert lat/lon to indices to use later
x_min = lon_to_index(WESTMOST_LON)
x_max = lon_to_index(EASTMOST_LON)
y_min = lat_to_index(NORTHMOST_LAT)
y_max = lat_to_index(SOUTHMOST_LAT)

# predetermine random list of seeds for 
SEEDS = randomize_seeds(len(NOISE_PCTS) * NUM_EXPERIMENTS)

# set up folder architecture
DATA_PATH = os.path.join(HOME_PATH, "data")
PRED_PATH = os.path.join(DATA_PATH, "hurricane_run.nc")
TRUE_PATH = os.path.join(DATA_PATH, "fcnv2_input.nc")
ERROR_LOG_PATH = os.path.join(DATA_PATH, "error_log.json")
ERROR_DATAPATH = os.path.join(DATA_PATH, "error_ds.nc")

PLOT_DIR = os.path.join(HOME_PATH, "plots")
PRED_PLOT_DIR = os.path.join(PLOT_DIR, "predictions")
ERR_PLOR_DIR = os.path.join(PLOT_DIR, "errors")
LOCAL_PRED_PLOT_DIR = os.path.join(PRED_PLOT_DIR, "local")
GLOBAL_PRED_PLOT_DIR = os.path.join(PRED_PLOT_DIR, "global")
LOCAL_ERR_PLOT_DIR = os.path.join(ERR_PLOR_DIR, "local")
GLOBAL_ERR_PLOT_DIR = os.path.join(ERR_PLOR_DIR, "global")