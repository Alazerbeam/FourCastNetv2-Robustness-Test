import numpy as np
import torch
import random
import os

# create a random list of seeds to use for each experiment
def randomize_seeds(num_seeds):
    seed_list = np.arange(num_seeds)
    np.random.shuffle(seed_list)
    seed_list = [int(seed) for seed in seed_list]
    return seed_list

# set all random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def units(channel_name):
    if channel_name in ["sp", "msl"]: return "Pa"
    elif channel_name == "tcwv": return "$kg/m^2$"
    elif channel_name[0] in ["u", "v"]: return "m/s"
    elif channel_name[0] == "z": return "$m^2/s^2$"
    elif channel_name[0] == "t": return "K"
    elif channel_name[0] == "r": return "%"
    elif channel_name[0] == "q": return "kg/kg"
    else: return "Unknown units."

# return full name of channel including units to display on plot
def full_name(channel_name):
    if channel_name == "sp": return f"Surface Pressure ({units(channel_name)})"
    elif channel_name == "msl": return f"Mean Sea Level Pressure ({units(channel_name)})"
    elif channel_name == "tcwv": return f"Total Column Water Vapor ({units(channel_name)})"
    elif channel_name[-1] == "m":  # if last character is m, represents meters above surface
        if channel_name[0] == "u": return f"East/West Wind {channel_name[1:-1]}m Above Surface ({units(channel_name)})"
        elif channel_name[0] == "v": return f"North/South Wind {channel_name[1:-1]}m Above Surface ({units(channel_name)})"
        elif channel_name[0] == "t": return f"Temperature {channel_name[1:-1]}m Above Surface ({units(channel_name)})"
        else: return "Unknown channel."
    elif channel_name[0] == "u": return f"East/West Wind at {channel_name[1:]} hPa ({units(channel_name)})"
    elif channel_name[0] == "v": return f"North/South Wind at {channel_name[1:]} hPa ({units(channel_name)})"
    elif channel_name[0] == "z": return f"Geopotential at {channel_name[1:]} hPa ({units(channel_name)})"
    elif channel_name[0] == "t": return f"Temperature at {channel_name[1:]} hPa ({units(channel_name)})"
    elif channel_name[0] == "r": return f"Relative Humidity at {channel_name[1:]} hPa ({units(channel_name)})"
    elif channel_name[0] == "q": return f"Specific Humidity at {channel_name[1:]} hPa ({units(channel_name)})"
    else: return "Unknown channel."

# Set means and stds in MODEL_DIR to match your synthetic distributions
def create_synthetic_stats(model_dir, mean=0, std=1):
    global_means = np.ones((1, 73, 1, 1)) * mean
    global_stds = np.ones((1, 73, 1, 1)) * std
    np.save(os.path.join(model_dir, "global_means.npy"), global_means)
    np.save(os.path.join(model_dir, "global_stds.npy"), global_stds)

# Replace means and stds in MODEL_DIR with the correct data from ERA5_STATS_DIR
def restore_channel_stats(model_dir, era5_stats_dir):
    era5_means = np.load(os.path.join(era5_stats_dir, "global_means.npy"))
    era5_stds = np.load(os.path.join(era5_stats_dir, "global_stds.npy"))
    np.save(os.path.join(model_dir, "global_means.npy"), era5_means)
    np.save(os.path.join(model_dir, "global_stds.npy"), era5_stds)

# Calculate the mean and std from given distribution with given parameters
def calc_mean_and_std_from_distr(distribution='normal', mean=0, std=1, df=1, a=0, b=1):
    if distribution.lower()[0] == 'n': # normal
        return mean, std
    elif distribution.lower()[0] == 'c': # chi-sq
        return df, np.sqrt(2*df)
    elif distribution.lower()[0] == 'u': # uniform
        return (a+b)/2., np.sqrt((b-a)**2./12.)
    elif distribution.lower()[0] == 'l': # lognormal
        return np.exp(mean + std**2. / 2.), np.sqrt((np.exp(std**2.) - 1.) * np.exp(2. * mean + std**2.))
    else:
        print("Unknown distribution. Defaulting to N(0,1).")
        return 0., 1.

# rescale the output of FCNv2 to match the era5 distribution for each channel
def rescale_fcnv2_output(output_array, era5_stats_dir, distr_mean, distr_std):
    era5_means = np.load(os.path.join(era5_stats_dir, "global_means.npy"))
    era5_stds = np.load(os.path.join(era5_stats_dir, "global_stds.npy"))
    
    distr_means = np.ones((1, 73, 1, 1)) * distr_mean
    distr_stds = np.ones((1, 73, 1, 1)) * distr_std
    
    normalized = (output_array - distr_means) / distr_stds
    rescaled = normalized * era5_stds + era5_means
    
    return rescaled.astype(np.float32)