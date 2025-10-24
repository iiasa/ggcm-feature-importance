"""
Module for processing yields from the GDHY dataset (Iizumi and Sakai, 2020), which is structured 
differently than the simulated yields. It contains no time dimension; instead, NC files are 
provided by year.
"""

import pandas as pd
from tqdm import tqdm
import xarray as xr
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess
from utils import Timer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dir', default=None, required=True, type=Path)
parser.add_argument('--out', default=None, required=True, type=Path, help='Path to output HDF file')
parser.add_argument('--year_from', default=1981, required=False, type=int)
parser.add_argument('--year_to', default=2016, required=False, type=int)
args = parser.parse_args()

path_dir = args.dir
path_out = args.out
year_from = args.year_from
year_to = args.year_to

years = []
for year in tqdm(range(1981, 2016 + 1), desc='Iterating years NCs'):
    with xr.open_dataset(path_dir / f'yield_{year}.nc4', decode_times=False) as yield_ds:
        yields = yield_ds['var'].to_dataframe()
        yields.dropna(inplace=True)
        yields.rename({'var': 'yield'}, axis=1, inplace=True)
        yields['year'] = year
        yields.set_index(['year'], append=True, inplace=True)
        years.append(yields)

yields = pd.concat(years, axis=0)
yields.sort_index(inplace=True)

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
