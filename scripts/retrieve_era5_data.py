import cdsapi
import os
import xarray as xr
from config import DATA_PATH, TRUE_PATH
from load_data import load_dataset

def retrieve_era5_data():
    c = cdsapi.Client()
    
    # retrieve pressure level data
    c.retrieve(
        'reanalysis-era5-complete',
        {
            'class': 'ea',
            'date': '2018-09-13/to/2018-09-16',
            'expver': '1',
            'levelist': '1000/925/850/700/600/500/400/300/250/200/150/100/50',  # Pressure levels in hPa
            'levtype': 'pl',  # pressure levels
            'param': '129/130/131/132/157',  # u, v, z, t, r 
            'stream': 'oper',
            'type': 'an',  # analysis
            'grid': '0.25/0.25',  # 0.25-degree resolution
            'time': [
                '00:00', '06:00', '12:00', '18:00',
            ],
            'area': [90, 0, -90, 359.75], # global data
            'format': 'netcdf',
        },
        os.path.join(DATA_PATH, 'era5_pl.nc')
    )
    
    # retrieve surface level data
    c.retrieve(
        'reanalysis-era5-complete',
        {
            'class': 'ea',
            'date': '2018-09-13/to/2018-09-16',
            'expver': '1',
            'levtype': 'sfc',  # surface variables
            'param': '134/137/151/165/166/167/228246/228247',  # u10m, v10m, u100m, v100m, t2m, sp, msl, tcwv
            'stream': 'oper',
            'type': 'an',  # analysis
            'grid': '0.25/0.25',  # 0.25-degree resolution
            'time': [
                '00:00', '06:00', '12:00', '18:00',
            ],
            'area': [90, 0, -90, 359.75], # global data
            'format': 'netcdf',
        },
        os.path.join(DATA_PATH, 'era5_sfc.nc')
    )

    surface_ds = load_dataset(os.path.join(DATA_PATH, "era5_sfc.nc"))
    pressure_ds = load_dataset(os.path.join(DATA_PATH, "era5_pl.nc"))
    
    # rename time dimension to match FCNv2
    surface_ds = surface_ds.rename({'valid_time': 'time'})
    pressure_ds = pressure_ds.rename({'valid_time': 'time'})
    
    # rearrange and merge datasets into a single dataset matching the configuration of FCNv2 input/output
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    
    # each variable in order, starting with all surface vars. note that variable names will be adjusted to match FCNv2 channels
    ordered_vars = [surface_ds[var] for var in ['u10', 'v10', 'u100', 'v100', 't2m', 'sp', 'msl', 'tcwv']]
    
    for var in ['u', 'v', 'z', 't', 'r']:
        arr = pressure_ds[var].sel(pressure_level=levels)
        for lev in levels:
            single_level = arr.sel(pressure_level=lev).assign_coords(pressure_level=None).expand_dims("channel")
            ordered_vars.append(single_level)
    
    data_array = xr.concat(ordered_vars, dim='channel')
    data_array = data_array.transpose('time', 'channel', 'latitude', 'longitude')  # ensure correct order (time, channel, lat, lon)
    tensor = data_array.values
    print("tensor shape:", tensor.shape)
    
    # adjust names to match FCNv2 channels
    channel_names = [
        'u10m', 'v10m', 'u100m', 'v100m', 't2m', 'sp', 'msl', 'tcwv'
    ] + [f"{var}{lev}" for var in ['u', 'v', 'z', 't', 'r'] for lev in levels]
    
    # reconstruct data array with correct names
    data_xr = xr.DataArray(
        tensor,
        dims=["time", "channel", "latitude", "longitude"],
        coords={
            "time": surface_ds.time,
            "channel": channel_names,
            "latitude": surface_ds.latitude,
            "longitude": surface_ds.longitude,
        },
        name="forecast"
    )
    
    ds = data_xr.to_dataset()
    print("Available channels:", ds.channel.values)
    
    ds.to_netcdf(TRUE_PATH)