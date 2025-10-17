import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import xarray as xr
from tqdm import tqdm
import os
from torch.distributions import Chi2, LogNormal
from earth2mip.networks import get_model
from load_data import load_era5_stats, load_dataset, get_all_data_values
from utils import set_seed, create_synthetic_stats, restore_channel_stats, calc_mean_and_std_from_distr, rescale_fcnv2_output
from config import *

# add noise to real input
def get_noisy_input(noise_prop: float):
    size = (1, 1, 73, 721, 1440) # batch size, timesteps, channels, latitudes, longitudes
    data = get_all_data_values(TRUE_PATH)
    init_conds = data[0].reshape(size)
    init_conds = torch.from_numpy(init_conds).float()
    
    _, era5_stds = load_era5_stats()
    stds = era5_stds[0, :, 0, 0].reshape(1, 1, 73, 1, 1)
    base_noise = torch.randn_like(init_conds) # N(0,1)
    return init_conds + base_noise * noise_prop * stds # init_conds + N(0,noise_prop*stds)

# generate fully random data from given distribution (normal, chi-sq, lognormal, uniform) with given parameters
def get_random_input(distribution: str = "normal", mean: float = 0, std: float = 1, df: float = 1, a: float = 0, b: float = 1):
    size = (1, 1, 73, 721, 1440) # batch size, timesteps, channels, latitudes, longitudes
    if distribution.lower()[0] == 'n': # normal
        return torch.normal(mean=mean, std=std, size=size)
    elif distribution.lower()[0] == 'c': # chi-sq
        return Chi2(df).sample(size)
    elif distribution.lower()[0] == 'u': # uniform
        return a + (b - a) * torch.rand(*size)
    elif distribution.lower()[0] == 'l': # lognormal
        return LogNormal(mean, std).sample(size)
    else: # otherwise use default normal distribution
        print('Unknown distribution. Defaulting to N(0,1).')
        return torch.randn(*size)

# load model, run inference, save forecast to NetCDF file
def run_inference(
    mode: str = "noise", noise_prop: float = 0.0, distribution: str = "normal", 
    mean: float = 0, std: float = 1, df: float = 1, a: float = 0, b: float = 1, 
    seed: float = 42, verbose: bool = False
    ):
    # set seed for reproducibility
    set_seed(seed)

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # only used for random input
    distr_mean = 0
    distr_std = 1
    
    # get the initial conditions
    if mode.lower()[0] == "n": # if noise mode, add Gaussian noise to real input
        init_cond = get_noisy_input(noise_prop).to(device)
        if verbose: print("Successfully generated noisy initial condition!")
    else: # if random mode, generate random initial condition and update channel means and stds
        init_cond = get_random_input(distribution, mean, std, df, a, b).to(device)
        distr_mean, distr_std = calc_mean_and_std_from_distr(distribution, mean, std, df, a, b)
        create_synthetic_stats(MODEL_DIR, distr_mean, distr_std)
        if verbose: print("Successfully generated random initial condition and updated channel stats!")
    
    # save path to modified files
    pkg_dir = Path(MODEL_DIR).resolve()
    
    # load the model
    model = get_model(f"file://{pkg_dir}")
    model = model.to(device)  # move to gpu if available
    
    if verbose:
        print("Successfully loaded model!")
    
    # run inference
    time = datetime(2018, 9, 13, 0, 0)
    predictions = []
    times = []
    
    iterator = model(time, init_cond)
    for _ in tqdm(range(TIMESTEPS), desc="Generating forecast"):
        temp_time, temp_output, _ = next(iterator)
        predictions.append(temp_output[0].cpu().numpy())
        times.append(temp_time)

    if verbose:
        print("Successfully ran inference!")
    
    predictions = np.stack(predictions)     # shape: (timesteps, channels, lat, lon)

    # if using random input, rescale output to match era5 distribution and restore channel stats
    if mode.lower()[0] != "n":
        if verbose: print("rescaling predictions and restoring channel stats")
        predictions = rescale_fcnv2_output(predictions, ERA5_STATS_DIR, distr_mean, distr_std)
        restore_channel_stats(MODEL_DIR, ERA5_STATS_DIR)

    # convert to xarray dataset
    data_arr = xr.DataArray(
        predictions,
        dims = ["time", "channel", "lat", "lon"],
        coords = {
            "time": times,
            "channel": model.out_channel_names,
            "lat": model.grid.lat,
            "lon": model.grid.lon
        },
        name = "forecast"
    )
    ds = data_arr.to_dataset()
    
    if verbose:
        print("Successfully converted to xarray dataset!")
    
    # save
    ds.to_netcdf(PRED_PATH, mode='w')
    ds.close()

    if verbose:
        print(f"Saved forecast to {PRED_PATH}!")