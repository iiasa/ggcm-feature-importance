"""
Module for processing yields, simulated by models of the GGCMI ensemble (Jägermeyr et al., 2021).
"""

import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.nonparametric.smoothers_lowess import lowess
from utils import Timer
from pathlib import Path
import calendar
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--yieldfile', default=None, required=True, type=Path)
parser.add_argument('--mdayfile', default=None, required=False, type=Path)
parser.add_argument('--pdayfile', default=None, required=False, type=Path)
parser.add_argument('--out', default=None, required=True, type=Path, help='Path to output HDF file')
parser.add_argument('--year_from', default=1971, required=False, type=int)
parser.add_argument('--year_to', default=2015, required=False, type=int)
parser.add_argument('--shift', default=0, required=False, type=int)
args = parser.parse_args()

path_yield = args.yieldfile
path_mdayfile = args.mdayfile
path_pdayfile = args.pdayfile
path_out = args.out
year_from = args.year_from
year_to = args.year_to
shift = args.shift

if shift != 0 and (not path_mdayfile or not path_pdayfile):
    raise ValueError('For shifts, pass mdayfile and pdayfile')

YIELD_NC_OFFSET = 1901  # First year in NetCDF

# Prepare shift-mask for GGCMs with shifted data (when pd > hd)
if shift != 0:
    ds_cal = xr.open_mfdataset([path_pdayfile, path_mdayfile], decode_times=False)
    ds_cal = ds_cal.assign_coords(time=ds_cal.time + year_from)
    
    ds_cal = ds_cal.to_dataframe().dropna()
    rename_cols = {
        [x for x in ds_cal.columns if 'matyday' in x][0]: 'len',  # expect columns named "matyday.." and "plantday.."
        [x for x in ds_cal.columns if 'plantday' in x][0]: 'pd'
    }
    ds_cal.rename(columns=rename_cols, inplace=True)
    ds_cal['leap'] = [calendar.isleap(x) for x in ds_cal.index.get_level_values('time')]
    ds_cal['days_in_year'] = 365 + ds_cal['leap']
    ds_cal['hd'] = (ds_cal['pd'] + ds_cal['len']) % ds_cal['days_in_year']
    shiftmask = ds_cal[ds_cal['pd'] > ds_cal['hd']]
    shiftmask.index = shiftmask.index.rename('year', level=2)
    shiftmask.index.reorder_levels(order=['lat', 'lon', 'year'])

else:
    shiftmask = None

with xr.open_dataset(path_yield, decode_times=False) as yield_ds:
    field_name = [x for x in yield_ds.data_vars][0]
    yields = yield_ds.isel(time=np.arange(year_from - YIELD_NC_OFFSET, year_to - YIELD_NC_OFFSET + 1))[field_name]
    min_year = yields.time.min().item()
    yields = yields.to_dataframe()
    yields.index.names = ['year', 'lat', 'lon']
    yields.index = yields.index.reorder_levels(order=['lat', 'lon', 'year'])
    yields.index = yields.index.set_levels((yields.index.levels[2] - min_year + year_from).astype(int), level='year')
    yields.sort_index(inplace=True)
    yields.rename({field_name: 'yield'}, axis=1, inplace=True)

    if shiftmask is not None:
        shift_idx = yields.join(shiftmask, how='inner').index
        yields.loc[shift_idx] = yields.loc[shift_idx].groupby(['lat', 'lon']).shift(shift)

    yields.dropna(inplace=True)

    # Only use pixels with >50% nonzero yields in timeseries
    yields = yields.groupby(['lat', 'lon']).filter(lambda x: (x['yield'] > 0).sum() / x.shape[0] > 0.5)

    # Apply LOESS
    with Timer('Detrending'):
        yhat = yields['yield'].groupby(['lat', 'lon']).transform(
            lambda x: lowess(x, x.index.get_level_values('year'), frac=0.5, it=0)[:, 1])
        yhat.name = 'yhat'
        ryield = (yields['yield'] - yhat) / yhat
        ryield.name = 'rel_yield'
        yields = pd.concat([yields, yhat, ryield], axis=1)

    with Timer('Storing result'):
        yields.dropna(inplace=True)
        yields.to_hdf(path_out, key='lat', mode='w', complevel=5)

print('done')
