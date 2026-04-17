"""
Module for processing climate data as provided by Frieler et al., 2024. TODO: ref binary format
"""

import sys
import warnings
import numpy as np
import pandas
import polars as pl

import xarray as xr
from sklearn.linear_model import LinearRegression
from data.features import add_gdd, add_hd, add_hui, add_lai_chd, add_pet, add_streak_ids, add_subgs, clip_gs, init_gs, longest_streak
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
data_in = []

if calendar_mode:
    ds_gs = xr.open_mfdataset(ggcm_ncs, decode_times=False)
    ds_gs = ds_gs.assign_coords(time=np.arange(clim_year_from, clim_year_to + 1))
    ds_gs = ds_gs.to_dataframe().dropna().rename(columns={'plantday-{crop}-noirr': 'pd', 'matyday-{crop}-noirr': 'len'})

    if ds_gs.index.names != ['lon', 'lat', 'time']:
        warnings.warn('Swapping GS information coordinates')
        ds_gs.index = ds_gs.index.reorder_levels(['lon', 'lat', 'time']) # type: ignore

land_pxls = set(climate.land_pixels())
if calendar_mode:
    gs_pxls = set(zip(ds_gs.index.get_level_values('lat'), ds_gs.index.get_level_values('lon')))
    land_pxls = land_pxls & gs_pxls
land_pxls = sorted(land_pxls)

# land_pxls = [(48.25, 16.75)]  # Marchfeld
# land_pxls = [(-55.25, -68.25)]

n_pixels = 100
pixels = list(sorted(land_pxls))[:n_pixels]
seq_years = np.arange(year_from, year_to + 1)
px_locdat = (
    loc_data.loc[pixels]
    .reset_index()
    .rename({'YLAT': 'LAT', 'XLON': 'LON', 'PLDOY': 'PD', 'HRDOY': 'HD'}, axis=1)
)
px_co2dat = pl.DataFrame({"year": co2_data.keys(), "CO2": co2_data.values()})

clim_results = climate.read(query={(x[0], x[1]): seq_years for x in pixels})
px_ts = [
    np.stack([clim_results[((px[0], px[1]), year)] for year in seq_years], axis=0) 
    for px in pixels
]
px_climate = np.stack(px_ts, axis=0)  # n_pixels x n_years x 366 (days) x n_features



n_pixel, n_year, n_day, n_feature = px_climate.shape

df = pl.DataFrame(
    {
        "pixel": np.repeat(np.arange(n_pixel), n_year * n_day),
        # "year": np.tile(np.repeat(np.arange(n_year), n_day), n_pixel),
        "year": np.tile(np.repeat(seq_years, n_day), n_pixel),
        "day": np.tile(np.arange(1, n_day + 1), n_pixel * n_year),
        **{
            climate_vars[i]: px_climate[..., i].reshape(-1)
            for i in range(n_feature)
        },
    }
)

# Convert units
df = df.with_columns(
    pl.col("hurs") / 100,  # % -> ratio
    pl.col("pr") * 24 * 60 * 60,  # kg/m²/s -> mm/day; 1 kg of rain water spread over 1 square meter of surface is 1 mm in thickness
    pl.col("rsds") * 60 * 60 * 24 / 1e6,  # W/m² -> MJ/m²/day
    pl.col("tasmax") - 273.15,  # K -> °C
    pl.col("tasmin") - 273.15,  # K -> °C
).with_columns(
    tav = (pl.col("tasmax") + pl.col("tasmin")) / 2
)


df1 = pl.DataFrame(px_locdat[['PD', 'HD', 'PHU', 'ELEV', 'PRMT74', 'LAT']]).with_columns(
    pl.Series(np.arange(n_pixel)).alias('pixel')
)

df = df.join(df1, on='pixel').join(px_co2dat, on='year')


df = df.sort(["pixel", "year", "day"])

# df_gs = df.with_columns(
#     season_start=(pl.col("day") == pl.col("PLDOY")).cast(pl.Int32),
#     in_gs=pl.when(pl.col("PLDOY") <= pl.col("HRDOY"))
#     .then(
#         pl.col("day").is_between(pl.col("PLDOY"), pl.col("HRDOY"), closed="both")
#     )
#     .otherwise(
#         (pl.col("day") >= pl.col("PLDOY")) | (pl.col("day") <= pl.col("HRDOY"))
#     ),
# ).with_columns(
#     gs_id_raw=pl.col("season_start").cum_sum().over("pixel"),
# ).with_columns(
#     GS=pl.when(pl.col("in_gs"))
#     .then(pl.col("gs_id_raw"))
#     .otherwise(0)
#     .cast(pl.Int32)
# ).drop(["season_start", "in_gs", "gs_id_raw"])


compute_hd = True

df = df.filter(pl.col("pixel") == 30)

df = init_gs(df)
df = add_gdd(df, corn.tbsc)
df = add_hui(df, corn.gmhu)
if compute_hd:
    df = add_hd(df)
df = clip_gs(df)
df = add_lai_chd(df, corn.dlap1, corn.dlap2, corn.hmx, corn.dlai, corn.dmla, corn.rlad)
df = add_pet(df, corn.vpth, corn.gsi, corn.vpd2)
# Tier 1 subseasons: vegetative and reproductive
df = add_subgs(
    df, 
    "GSsub1", 
    {
        'GSv': (0, 0.5),
        'GSr': (0.5, 1),
    }
)
# Tier 2 subseasons: emergence, development, flowering, grain filling
df = add_subgs(
    df, 
    "GSsub2", 
    {
        'GSe': (0, 0.25),
        'GSd': (0.25, 0.5),
        'GSf': (0.5, 0.75),
        'GSg': (0.75, 1), 
    }
)

df = df.with_columns(CMD = pl.col("PET") - pl.col("pr"))

df = add_streak_ids(df)


def aggregates():
    return [
        pl.col("tasmax").mean().alias("TMXav"),
        pl.col("tasmin").mean().alias("TMNav"),
        pl.col("tav").mean().alias("TAVav"),

        pl.col("pr").sum().alias("PRCPsum"),
        pl.col("rsds").sum().alias("RADsum"),

        pl.col("sfcwind").mean().alias("WSDav"),
        pl.col("hurs").mean().alias("HURav"),

        pl.col("PET").sum().alias("PETsum"),
        pl.col("GDD").sum().alias("GDDsum"),
        pl.col("CMD").sum().alias("CMDsum"),

        pl.len().alias("LEN"),
        pl.col("HUI").last().alias("HUIeop"),
    ]

def counts():
    return [
        # temperature extremes
        (pl.col("tasmax") >= 30).sum().alias("HDD"),
        (pl.col("tasmax") >= 39).sum().alias("KDD"),
        (pl.col("tasmin") <= 0).sum().alias("FRT"),
        (pl.col("tasmax") <= 0).sum().alias("ICE"),

        # precipitation thresholds
        (pl.col("pr") >= 10.0).sum().alias("R10"),
        (pl.col("pr") >= 20.0).sum().alias("R20"),

        # wet / dry days
        (pl.col("pr") > 1.0).sum().alias("WET"),
        (pl.col("pr") <= 1.0).sum().alias("DRY"),

        # CMD < 0
        ((pl.col("CMD")) < 0).sum().alias("CMDlt0"),
    ]

final = []
for lvl, agg in enumerate([["pixel", "GS"], ["pixel", "GS", "GSsub1"], ["pixel", "GS", "GSsub2"]]):
    df_agg = df.group_by(agg).agg(aggregates())
    cwd = longest_streak(df, agg, "wd", "wd_grp", "CWD")
    cdd = longest_streak(df, agg, "dd", "dd_grp", "CDD")
    df_agg = (
        df_agg
        .join(cwd, on=agg, how="left")
        .join(cdd, on=agg, how="left")
    )
    if lvl == 0:
        df_agg = df_agg.with_columns(PERIOD = pl.lit('GS')).drop('GS')
    elif lvl == 1:
        df_agg = df_agg.with_columns(PERIOD = pl.col("GSsub1")).drop(['GS', 'GSsub1'])
    elif lvl == 2:
        df_agg = df_agg.with_columns(PERIOD = pl.col("GSsub2")).drop(['GS', 'GSsub2'])
    final.append(df_agg)


    

# continue: consecutive?
df_agg1 = df.group_by(["pixel", "GS"]).agg(aggregates())
cwd = longest_streak(df, ["pixel", "GS"], "wd", "wd_grp", "CWD")


df_agg1.join(cwd, on=["pixel", "GS"], how="left")


df_agg2 = df.group_by(["pixel", "GS", "GSsub1"]).agg(aggregates())
df_agg3 = df.group_by(["pixel", "GS", "GSsub2"]).agg(aggregates())


asd = 'asd'

