"""
Module for processing climate data as provided by Frieler et al., 2024. TODO: ref binary format
"""

import sys
import warnings
import numpy as np
import pandas
import xarray as xr
from sklearn.linear_model import LinearRegression
from data.nikbins import ClimateBinReader
from tqdm import tqdm
import random
from models import unepic
from pathlib import Path
import calendar
import rioxarray
import argparse


url_params = ['mdayfile', 'pdayfile', 'climdir', 'locfile', 'co2file', 'out']
parser = argparse.ArgumentParser()
parser.add_argument('--year_from', default=1971, required=True, type=int)
parser.add_argument('--year_to', default=2015, required=True, type=int)
parser.add_argument('--clim_year_from', default=1901, required=True, type=int)
parser.add_argument('--clim_year_to', default=2016, required=True, type=int)
parser.add_argument('--shift', default=0, required=True, type=int)

for param in url_params:
    parser.add_argument(f'--{param}', default=None, required=False, type=Path)

args = parser.parse_args()
year_from = args.year_from
year_to = args.year_to
shift = args.shift
clim_year_from = args.clim_year_from
clim_year_to = args.clim_year_to

paths = {}
for param in url_params:
    paths[param] = args.__dict__.get(param)
    if paths[param] is not None:
        paths[param] = paths[param].resolve()
    if paths[param] is None or not paths[param].exists():
        raise FileNotFoundError(f"File not found: {paths[param]}")

np.random.seed(42)
random.seed(42)

# If mdayfile and pdayfile are passed as arguments, read crop calendar from these. 
# Otherwise, user growth model to estimate GS length. 
calendar_mode = paths['mdayfile'] is not None and paths['pdayfile'] is not None
path_target = paths['out']

if not calendar_mode:
    print('Estimating growing season from reference crop.')
else:
    print('Reading growing season from pdayfile & mdayfile.')
    ggcm_ncs = [paths['pdayfile'], paths['mdayfile']]

print(f'Processing')

# Reference crop for estimating GS in non-calendar mode.
corn = unepic.Crop(
    wa=40., tbsc=8., dlai=0.8, rlad=1., dmla=6., 
    dlap1=15.05, dlap2=50.95, top=25.0, rdmx=2.0, hi=0.5, 
    hmx=2.0, vpth=0.5, vpd2=0.071, gsi=0.007, gmhu=100.
)

# Replace S-curve values EPIC style
corn.dlap1, corn.dlap2 = unepic.ssolve(
    int(corn.dlap1) * .01, (corn.dlap1 - int(corn.dlap1)), 
    int(corn.dlap2) * .01, (corn.dlap2 - int(corn.dlap2))
)

def read_co2(path: str, year_from: int) -> dict:
    data = pandas.read_fwf(path)
    data.columns = ['YR', 'CO2']
    data.set_index('YR', inplace=True)
    data = data.loc[year_from:]
    if not 2015 in data.index:  # Project to 2015
        data.loc[2015] = LinearRegression(fit_intercept=True) \
        .fit(np.array(data.iloc[-5:].index).reshape(-1, 1), data.iloc[-5:].values) \
        .predict(np.array([[2015]])).item()
    return data['CO2'].to_dict()

def count_consecutive(arr: np.ndarray) -> int:
    """
    Returns length of the longest consecutive True sequence in an array.
    """
    return np.diff(np.where(np.concatenate(([arr[0]], arr[:-1] != arr[1:], [True])))[0])[::2]

def process_year(
        data_orig: np.ndarray, year: int, pd: int, hd: int, lat: float, elev: int, 
        phu: float, prmt_74: float, compute_gs: bool = True) -> np.array:

    data = data_orig.copy()

    data[data[:, 4] == 0] = np.nan
    data[data[:, 5] == 0] = np.nan
    
    # Convert units
    data[:, 0] = data[:, 0] / 100  # % -> ratio
    data[:, 1] = data[:, 1] * 24 * 60 * 60  # kg/m²/s -> mm/day; 1 kg of rain water spread over 1 square meter of surface is 1 mm in thickness
    data[:, 2] = data[:, 2] * 60 * 60 * 24 / 1e6  # W/m² -> MJ/m²/day
    data[:, 4] = data[:, 4] - 273.15  # K -> °C
    data[:, 5] = data[:, 5] - 273.15  # K -> °C

    # The growing season starts one day after planting (index: pd - 1 + 1 = pd)
    # HUI accumulation starts one day after GSS >= GMHU!
    gdd = np.maximum((data[pd:, 4] + data[pd:, 5]) / 2 - corn.tbsc, 0.)
    hui_start = np.where(np.cumsum(gdd) >= corn.gmhu)[0]
    hui_start = hui_start[0] + 1 if len(hui_start) > 0 else np.nan

    if np.isnan(hui_start):
        return None

    hui = np.hstack([np.zeros(hui_start), np.cumsum(gdd[hui_start:]) / phu])

    # Day of harvest withing GS
    if compute_gs:
        doh = np.where(hui >= 1.)[0]
        doh = doh[0] + hui_start if len(doh) > 0 else np.inf
        doh = min(doh, hd - pd + 21)
    else:
        doh = hd - pd

    # Select germination period
    data_germ = data[pd:(pd + hui_start)]

    # Cut HUI off at harvest date, so its length is the GS length.
    hui = hui[:doh]
    lai, cht = unepic.lai(hui=hui, dlap1=corn.dlap1, dlap2=corn.dlap2, dlai=corn.dlai, rlad=corn.rlad, dmla=corn.dmla, hmx=corn.hmx)

    data_gs = data[pd:(pd + hui.shape[0])]
    tav = (data_gs[:, 4] + data_gs[:, 5]) / 2

    pet = unepic.pet_pm(
        tav=tav, rad=data_gs[:, 2], ws=data_gs[:, 3], hur=data_gs[:, 0], lai=lai, cht=cht, 
        doy=np.arange(pd, pd + data_gs.shape[0]),
        elev=elev, salb=0.15, lat=lat, 
        co2=co2_data[year], vpth=corn.vpth, gsi=corn.gsi, vpd2=corn.gsi, prmt_74=prmt_74)

    cmd = pet - data_gs[:, 1]
    wd = data_gs[:, 1] > 1.0  # pr > 1mm (McErlich)
    dd = data_gs[:, 1] <= 1.0
    cmd_sum_gs = np.nansum(cmd)
    wet_sum_gs = np.nansum(wd)
    dry_sum_gs = np.nansum(dd)
    cmd_lt0_sum_gs = np.nansum(cmd < 0)
    hdd_sum_gs = np.nansum(data_gs[:, 4] >= 30)  # Schauberger et al. 2017
    kdd_sum_gs = np.nansum(data_gs[:, 4] >= 39)  # Schauberger et al. 2017
    frt_sum_gs = np.nansum(data_gs[:, 5] <= 0)  # tasmin
    ice_sum_gs = np.nansum(data_gs[:, 4] <= 0)  # tasmax
    r10_sum_gs = np.nansum(data_gs[:, 1] >= 10.0)  # pr
    r20_sum_gs = np.nansum(data_gs[:, 1] >= 20.0)
    cwd_sum_gs = np.max(count_consecutive(wd)) if wet_sum_gs > 0 else 0
    cdd_sum_gs = np.max(count_consecutive(dd)) if dry_sum_gs > 0 else 0

    frt_sum_germ = np.nansum(data_germ[:, 5] <= 0)
    ice_sum_germ = np.nansum(data_germ[:, 4] <= 0)
    r10_sum_germ = np.nansum(data_germ[:, 1] >= 10.0)
    r20_sum_germ = np.nansum(data_germ[:, 1] >= 20.0)
    gs_length = data_gs.shape[0]

    derived = np.hstack([
        cmd_sum_gs, wet_sum_gs, dry_sum_gs, cmd_lt0_sum_gs, hdd_sum_gs, kdd_sum_gs, 
        frt_sum_gs, ice_sum_gs, r10_sum_gs, r20_sum_gs, cwd_sum_gs, cdd_sum_gs,
        frt_sum_germ, ice_sum_germ, r10_sum_germ, r20_sum_germ
    ], dtype=np.float32)

    aggregated = np.hstack([
        np.nanmean(data_gs[:, 0]), 
        np.nansum(data_gs[:, 1]), 
        np.nansum(data_gs[:, 2]), 
        np.nanmean(data_gs[:, 3]), 
        np.nanmean(data_gs[:, 4]), 
        np.nanmean(data_gs[:, 5]), 
        np.nansum(pet),
        np.nansum(tav),
    ], dtype=np.float32)

    others = [gs_length]

    return np.hstack([aggregated, derived, others])

def flush_to_disk(data: dict):
    df = pandas.DataFrame(data).T
    df.columns = (
        'hurs_av_gs', 'pr_sum_gs', 'rsds_sum_gs', 'sfcwind_av_gs', 'tasmax_av_gs', 'tasmin_av_gs', 'pet_sum_gs', 'tav_sum_gs', 
        'cmd_sum_gs', 'wet_sum_gs', 'dry_sum_gs', 'cmd_lt0_sum_gs', 'hdd_sum_gs', 'kdd_sum_gs', 
        'frt_sum_gs', 'ice_sum_gs', 'r10_sum_gs', 'r20_sum_gs', 'cwd_sum_gs', 'cdd_sum_gs',
        'frt_sum_germ', 'ice_sum_germ', 'r10_sum_germ', 'r20_sum_germ',
        'gs_length', 
        'interannual'
    )
    df.index.set_names(('lat', 'lon', 'year'), inplace=True)
    df.to_hdf(path_target, key='lat', mode='w', complevel=5)

climate_vars = [
    'hurs',  # Near-Surface Relative Humidity
    'pr',  # Precipitation
    'rsds',  # Surface Downwelling Shortwave Radiation
    'sfcwind',  # Near-Surface Wind Speed
    'tasmax',  # Daily Maximum Near-Surface Air Temperature
    'tasmin',  # Daily Minimum Near-Surface Air Temperature
]

climate = ClimateBinReader(
    data_dir=paths['climdir'], 
    landmap_path=paths['climdir'] / 'land-map.bin', 
    climate_vars=climate_vars)

loc_data = pandas.read_csv(paths['locfile'])
loc_data.set_index(['YLAT', 'XLON'], inplace=True)

co2_data = read_co2(paths['co2file'], year_from)
data = {}

if calendar_mode:
    ds_gs = xr.open_mfdataset(ggcm_ncs, decode_times=False)
    ds_gs = ds_gs.assign_coords(time=np.arange(clim_year_from, clim_year_to + 1))
    ds_gs = ds_gs.to_dataframe().dropna().rename(columns={'plantday-{crop}-noirr': 'pd', 'matyday-{crop}-noirr': 'len'})

    if ds_gs.index.names != ['lon', 'lat', 'time']:
        warnings.warn('Swapping GS information coordinates')
        ds_gs.index = ds_gs.index.reorder_levels(['lon', 'lat', 'time'])

land_pxls = set(climate.land_pixels())
if calendar_mode:
    gs_pxls = set(zip(ds_gs.index.get_level_values('lat'), ds_gs.index.get_level_values('lon')))
    land_pxls = land_pxls & gs_pxls

# Process ts data per pixel
for i, (lat, lon) in enumerate(tqdm(sorted(land_pxls))):

    # Read soil data for pixel
    soil_loc = loc_data.loc[(lat, lon)]
    if 'PLDOY' in soil_loc and 'HRDOY' in soil_loc:
        pd = int(soil_loc['PLDOY'])  # planting date for biphysical estimation
        hd = int(soil_loc['HRDOY'])
    elif not calendar_mode:
        raise ValueError('Please provide planting (PLDOY) and harvest (HRDOY) date in location data')
    
    elev = soil_loc['ELEV']
    phu = soil_loc['PHU']
    prmt_74 = soil_loc['PRMT74']

    climate_data = climate.read(query={(lat, lon): np.arange(year_from, year_to + 1)})

    for year in np.sort(np.arange(year_from + 1, year_to + 1)):
        # Overwrite with GGCM-specific pd and hd
        if calendar_mode:
            if not (lon, lat, year) in ds_gs.index:
                # warnings.warn(f'{lon}, {lat}, {year} not found')
                continue
            gs_len, pd = ds_gs.loc[lon, lat, year].values.astype(int)

            days_in_year = 365 + calendar.isleap(year)
            hd = (pd + gs_len) % days_in_year

            # Correct yearly shifts
            if pd > hd and shift != 0:
                if (year + shift) in ds_gs.loc[lon, lat].index:
                    gs_len, pd = ds_gs.loc[lon, lat, year + shift].values.astype(int)
                    hd = (pd + gs_len) % days_in_year
                else:
                    continue

        if pd > hd:
            data_year = np.vstack([
                climate_data[((lat, lon), year - 1)], 
                climate_data[((lat, lon), year)]
            ])
            data_processed = process_year(data_year, year, pd, 366 + hd, lat, elev, phu, prmt_74, not calendar_mode)
            if data_processed is None:
                continue
        
        else:
            data_year = climate_data[((lat, lon), year)]
            data_processed = process_year(data_year, year, pd, hd, lat, elev, phu, prmt_74, not calendar_mode)
            if data_processed is None:
                continue

        data[(lat, lon, year)] = np.hstack([data_processed, pd > hd])
    
if len(data) > 0:
    flush_to_disk(data)

print('done')
