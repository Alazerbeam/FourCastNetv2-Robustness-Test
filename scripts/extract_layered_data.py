import numpy as np
from noisy_hurricane import run_inference
from load_data import *
from config import *
from track_hurricane import index_to_lat, index_to_lon

# Convert forecast data to JSON file with architecture:
# noise level (%) -> variable (u, v, wind speed) -> pressure level (50-1000) -> timestep (0-14) -> lat -> lon -> raw data (km/hr)

json_datapath = "hurricane_predictions/data/layered_data.json"
layered_data = {}

def array_to_time_lat_lon_dict(arr):
    """
    Convert (time, lat, lon) NumPy array to nested dict:
    time_index -> lat_value -> lon_value -> float
    """
    out = {}
    timesteps, n_lat, n_lon = arr.shape

    for t in range(timesteps):
        lat_dict = {}
        for i_lat in range(n_lat):
            lat_val = index_to_lat(i_lat + y_min)
            lon_dict = {}
            for i_lon in range(n_lon):
                lon_val = index_to_lon(i_lon + x_min)
                lon_dict[lon_val] = float(arr[t, i_lat, i_lon])
            lat_dict[lat_val] = lon_dict
        out[t] = lat_dict

    return out

error_log = load_json(ERROR_LOG_PATH)
best_seeds = {}
median_seeds = {}
worst_seeds = {}
for noise_pct, seed2error in error_log.items():
    total_errors = [(seed, errors[-1]) for seed, errors in seed2error.items()]
    total_errors.sort(key = lambda x: x[1])
    best_seeds[noise_pct] = total_errors[0][0]
    median_seeds[noise_pct] = total_errors[len(total_errors) // 2][0]
    worst_seeds[noise_pct] = total_errors[-1][0]

noise_levels = ['0.0', '0.05', '0.2', '0.5', 'real']
for noise_pct in noise_levels:
    if noise_pct not in layered_data:
        layered_data[noise_pct] = {}
        layered_data[noise_pct]['u'] = {}
        layered_data[noise_pct]['v'] = {}
        layered_data[noise_pct]['wind speed'] = {}

    if noise_pct != 'real':
        if noise_pct in ['0.0', '0.05']:
            seed = best_seeds[float(noise_pct)]
        elif noise_pct == '0.2':
            seed = median_seeds[float(noise_pct)]
        else:
            seed = worst_seeds[float(noise_pct)]
        run_inference(MODEL_DIR, TRUE_PATH, PRED_PATH, ERA5_STATS_DIR, TIMESTEPS, float(noise_pct), seed, verbose=False)
        ds = load_dataset(PRED_PATH)
    else:
        ds = load_dataset(TRUE_PATH)
    
    for pressure_level in ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000']:
        zonal_wind = limit_data(get_channel_data(ds, 'u'+pressure_level), TIMESTEPS, x_min, x_max, y_min, y_max)
        meridional_wind = limit_data(get_channel_data(ds, 'v'+pressure_level), TIMESTEPS, x_min, x_max, y_min, y_max)
        wind_speeds = np.sqrt(zonal_wind**2. + meridional_wind**2.)

        layered_data[noise_pct]['u'][pressure_level] = array_to_time_lat_lon_dict(zonal_wind)
        layered_data[noise_pct]['v'][pressure_level] = array_to_time_lat_lon_dict(meridional_wind)
        layered_data[noise_pct]['wind speed'][pressure_level] = array_to_time_lat_lon_dict(wind_speeds)
    ds.close()

update_json(json_datapath, layered_data)
        
        
        

