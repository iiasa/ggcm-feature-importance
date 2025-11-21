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
from enum import Enum
import matplotlib.pyplot as plt


class VarIdx:
    HURS = 0  # Near-Surface Relative Humidity
    PRCP = 1  # Precipitation
    RSDS = 2  # Surface Downwelling Shortwave Radiation
    WIND = 3  # Near-Surface Wind Speed
    TMAX = 4  # Daily Maximum Near-Surface Air Temperature
    TMIN = 5  # Daily Minimum Near-Surface Air Temperature

# url_params = ['mdayfile', 'pdayfile', 'climdir', 'locfile', 'co2file', 'out']
# parser = argparse.ArgumentParser()
# parser.add_argument('--year_from', default=1971, required=True, type=int)
# parser.add_argument('--year_to', default=2015, required=True, type=int)
# parser.add_argument('--clim_year_from', default=1901, required=True, type=int)
# parser.add_argument('--clim_year_to', default=2016, required=True, type=int)
# parser.add_argument('--shift', default=0, required=True, type=int)

# for param in url_params:
#     parser.add_argument(f'--{param}', default=None, required=False, type=Path)

# args = parser.parse_args()
# year_from = args.year_from
# year_to = args.year_to
# shift = args.shift
# clim_year_from = args.clim_year_from
# clim_year_to = args.clim_year_to

# paths = {}
# for param in url_params:
#     paths[param] = args.__dict__.get(param)
#     if paths[param] is not None:
#         paths[param] = paths[param].resolve()
#     if paths[param] is None or not paths[param].exists():
#         raise FileNotFoundError(f"File not found: {paths[param]}")


paths = {
    'mdayfile': r'P:\esmscratch\ggcmi_sync\p\projects\macmit\data\GGCMI\AgMIP.output\EPIC-IIASA\phase3a\gswp3-w5e5\obsclim\mai\epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_matyday-mai-firr_global_annual_1901_2016.nc', 
    'pdayfile': r'P:\esmscratch\ggcmi_sync\p\projects\macmit\data\GGCMI\AgMIP.output\EPIC-IIASA\phase3a\gswp3-w5e5\obsclim\mai\epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_plantday-mai-firr_global_annual_1901_2016.nc',
    'climdir': r'C:\Users\oberleitner\projects\anfos\anfos_repo\data\input\climate',
    'locfile': r'C:\Users\oberleitner\projects\anfos\ggcm-feature-importance\data\input\CLIMATE_LAT_PD_HD_PHU_ELEV_PRMT74_mai_noirr_fH_v3c.csv',
    'co2file': r'C:\Users\oberleitner\projects\anfos\ggcm-feature-importance\data\input\co2_historical_annual_1765_2014.txt',
    'out': r'C:\Users\oberleitner\projects\anfos\ggcm-feature-importance\data\output\generated\climate_EPIC-IIASA_maize_rf.h5',
}
for p in paths:
    paths[p] = Path(paths[p])
year_from = 1971
year_to = 2015
shift = 0
clim_year_from = 1901
clim_year_to = 2016

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
    """
    :param data_orig: 2d array containing 1 or 2yr data, with days in rows and features in columns. 
    """

    data = data_orig.copy()

    # ?????
    # data[data[:, 4] == 0] = np.nan
    # data[data[:, 5] == 0] = np.nan
    
    # Convert units
    data[:, 0] = data[:, 0] / 100  # % -> ratio
    data[:, 1] = data[:, 1] * 24 * 60 * 60  # kg/m²/s -> mm/day; 1 kg of rain water spread over 1 square meter of surface is 1 mm in thickness
    data[:, 2] = data[:, 2] * 60 * 60 * 24 / 1e6  # W/m² -> MJ/m²/day
    data[:, 4] = data[:, 4] - 273.15  # K -> °C
    data[:, 5] = data[:, 5] - 273.15  # K -> °C

    # if year == 1992:
    #     de = 'asd'

    season, hui, gdd = unepic.compute_season(data[:, VarIdx.TMIN], data[:, VarIdx.TMAX], pd, hd, corn.tbsc, corn.gmhu, phu)
    if season is None:
        return None  # no germination

    lai, cht = unepic.lai(hui=hui, dlap1=corn.dlap1, dlap2=corn.dlap2, dlai=corn.dlai, rlad=corn.rlad, dmla=corn.dmla, hmx=corn.hmx)

    # plt.figure()
    # plt.plot(hui, label='HUI')
    # plt.plot(lai, label='LAI')
    # plt.legend()
    # plt.show()
    
    # todo: mind non germinated years

    aggs = {}

    for period, idx in season.items():
        d = data[idx]
        plen = d.shape[0]
        tav = (d[:, VarIdx.TMAX] + d[:, VarIdx.TMIN]) / 2
        pet = unepic.pet_pm(
            tav=tav, rad=d[:, VarIdx.RSDS], ws=d[:, VarIdx.WIND], hur=d[:, VarIdx.HURS], lai=lai[idx], cht=cht[idx], 
            doy=(np.arange(idx.start, idx.start + d.shape[0]) % 366) + 1,
            elev=elev, salb=0.15, lat=lat, 
            co2=co2_data[year], vpth=corn.vpth, gsi=corn.gsi, vpd2=corn.gsi, prmt_74=prmt_74)
        
        cmd = pet - d[:, VarIdx.PRCP]
        wd = d[:, VarIdx.PRCP] > 1.0  # pr > 1mm (McErlich)
        dd = d[:, VarIdx.PRCP] <= 1.0

        wet_sum_gs = np.nansum(wd)
        dry_sum_gs = np.nansum(dd)
        cwd_sum_gs = np.max(count_consecutive(wd)) if wet_sum_gs > 0 else 0
        cdd_sum_gs = np.max(count_consecutive(dd)) if dry_sum_gs > 0 else 0

        aggregates = {
            'TMXav': np.nanmean(d[:, VarIdx.TMAX]),
            'TMNav': np.nanmean(d[:, VarIdx.TMIN]),
            'TAVav': np.nanmean(tav),
            'PRCPsum': np.nansum(d[:, VarIdx.PRCP]),
            'RADsum': np.nansum(d[:, VarIdx.RSDS]),
            'WSDav': np.nanmean(d[:, VarIdx.WIND]),
            'HURav': np.nanmean(d[:, VarIdx.HURS]),
            'HURav': np.nanmean(d[:, VarIdx.HURS]),
            'PETsum': np.nansum(pet),
            'GDDsum': np.nansum(gdd[idx]),
            'CMDsum': np.nansum(cmd),
            'LEN': plen,
            'HUIeop': hui[idx][-1]
        }

        fracs = {
            'HDD': np.nansum(d[:, VarIdx.TMAX] >= 30),  # Schauberger et al. 2017
            'KDD': np.nansum(d[:, VarIdx.TMAX] >= 39),  # Schauberger et al. 2017
            'FRT': np.nansum(d[:, VarIdx.TMIN] <= 0), 
            'ICE':np.nansum(d[:, VarIdx.TMAX] <= 0), 
            'R10': np.nansum(d[:, VarIdx.PRCP] >= 10.0),
            'R20': np.nansum(d[:, VarIdx.PRCP] >= 20.0),
            'WET': wet_sum_gs,
            'DRY': dry_sum_gs,
            'CWD': cwd_sum_gs,
            'CDD': cdd_sum_gs,
            'CMDlt0': np.nansum(cmd < 0)
        }

        aggs = aggs | {k + period: v for k, v in aggregates.items()} | {k + 'frac' + period: v / plen for k, v in fracs.items()}

    return aggs


def get_int_dtype(range: int, signed: bool):
    if not signed:
        if range <= 255:
            return np.uint8
        elif range <= 65535:
            return np.uint16
        else:
            return np.uint32
    else:
        halfrange = range / 2
        if halfrange <= 127:
            return np.int8
        elif halfrange <= 32767:
            return np.int16
        else:
            return np.int32
        

def flush_to_disk(data: dict):

    df = pandas.DataFrame(data).T
    df.index.set_names(('lat', 'lon', 'year'), inplace=True)

    dtypes = {}
    for var in df.columns:
        maxint = df[var].max(axis=0) * 100
        minint = df[var].min(axis=0) * 100
        dtypes[var] = get_int_dtype(maxint - minint, minint < 0)



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

# land_pxls = [(48.25, 16.75)]  # Marchfeld
# land_pxls = [(-55.25, -68.25)]

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
            data_processed = data_processed | {'IA': True}
        
        else:
            data_year = climate_data[((lat, lon), year)]
            data_processed = process_year(data_year, year, pd, hd, lat, elev, phu, prmt_74, not calendar_mode)
            if data_processed is None:
                continue
            data_processed = data_processed | {'IA': False}

        data[(lat, lon, year)] = data_processed
    
if len(data) > 0:
    flush_to_disk(data)

print('done')
