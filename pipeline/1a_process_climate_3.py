"""
Module for processing climate data as provided by Frieler et al., 2024. TODO: ref binary format
"""

from typing import Literal
import numpy as np
import polars as pl

import xarray as xr
from data.features import cal_fix_shift, cal_len_to_hd, process_climate_features
from data.nikbins import ClimateBinReader
from tqdm import tqdm
import random
from data.pipeline import ClimateBinIterator, read_co2
from models import unepic
from pathlib import Path


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


path_pd = Path(r"P:\esmscratch\ggcmi_sync\p\projects\macmit\data\GGCMI\AgMIP.output\EPIC-IIASA\phase3a\gswp3-w5e5\obsclim\mai\epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_plantday-mai-firr_global_annual_1901_2016.nc")
path_hd = Path(r"P:\esmscratch\ggcmi_sync\p\projects\macmit\data\GGCMI\AgMIP.output\EPIC-IIASA\phase3a\gswp3-w5e5\obsclim\mai\epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_matyday-mai-firr_global_annual_1901_2016.nc")
path_climate = Path(r"C:\Users\oberleitner\projects\anfos\anfos_repo\data\input\climate")
path_co2 = Path(r"C:\Users\oberleitner\projects\anfos\ggcm-feature-importance\data\input\co2_historical_annual_1765_2014.txt")
path_location = Path(r"C:\Users\oberleitner\projects\anfos\ggcm-feature-importance\data\input\CLIMATE_LAT_PD_HD_PHU_ELEV_PRMT74_mai_noirr_fH_v3c.csv")
path_out = Path(r"C:\Users\oberleitner\projects\anfos\ggcm-feature-importance\data\output\generated\climate_EPIC-IIASA_maize_rf.parquet")

year_from = 1971
year_to = 2015
cal_start_year = 1901
shift = 0
hd_mode: Literal["strict", "semi_dynamic", "dynamic"] = "strict"

climate_vars = [
    'hurs',  # Near-Surface Relative Humidity
    'pr',  # Precipitation
    'rsds',  # Surface Downwelling Shortwave Radiation
    'sfcwind',  # Near-Surface Wind Speed
    'tasmax',  # Daily Maximum Near-Surface Air Temperature
    'tasmin',  # Daily Minimum Near-Surface Air Temperature
]

np.random.seed(42)
random.seed(42)

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

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# Prepare location/master df
# (This df determines the pixel over which the pipeline is applied)
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

# In this case, loc_data is already masked by the land pixels.
loc_data = pl.read_csv(
    path_location, 
    schema_overrides = {
        "YLAT": pl.Float32, 
        "XLON": pl.Float32, 
        "PLDOY": pl.Int16, 
        "HRDOY": pl.Int16, 
        "PHU": pl.Float32, 
        "ELEV": pl.Int16, 
        "PRMT74": pl.Float32
    }).rename({
        "YLAT": "lat",
        "XLON": "lon",
        "PLDOY": "pd",
        "HRDOY": "hd",
        "PHU": "phu",
        "ELEV": "elev",
        "PRMT74": "prmt74"
    }).drop(
        "CLIMATEID"
    )

# * * * * * * * * * * * * * * * * * * * * * * * *
# Prepare CO2 data
# * * * * * * * * * * * * * * * * * * * * * * * *

co2 = read_co2(path_co2, year_from)

# * * * * * * * * * * * * * * * * * * * * * * * *
# Prepare calendar
# * * * * * * * * * * * * * * * * * * * * * * * *

# Open PD (and HD) NCs and convert to polars df
if hd_mode != "dynamic":
    cal_xr = xr.open_mfdataset(
        [path_pd, path_hd], 
        combine="by_coords",
        chunks=None, 
        decode_times=False
    ).rename({"plantday-mai-firr": "pd", "matyday-mai-firr": "hd", "time": "yr"})
else:
    cal_xr = xr.open_dataset(
        path_pd, 
        decode_times=False
    ).rename({"plantday-mai-firr": "pd", "time": "yr"})
    cal_xr = cal_xr.assign(hd=None)

cal_xr = cal_xr.assign_coords(yr=np.arange(cal_start_year, cal_start_year + len(cal_xr.yr)))
cal_xr = cal_xr.sel(yr=slice(year_from, year_to))

cal = pl.DataFrame(
    cal_xr.to_dataframe().dropna().reset_index(),
    schema={"lat": pl.Float16, "lon": pl.Float16, "yr": pl.Int16, "pd": pl.Int16, "hd": pl.Int16}
)
cal_xr.close()
del cal_xr

# HD related transformations
if hd_mode != "dynamic":

    # Fix shift
    if shift != 0:
        cal = cal_fix_shift(cal, shift)

    # HD actually contains the GS length; convert to true HD.
    cal = cal_len_to_hd(cal)

# Set up climate streamer
bin_reader = ClimateBinReader(
    data_dir=path_climate, 
    landmap_path=path_climate / 'land-map.bin', 
    climate_vars=climate_vars
)

pixels = set(bin_reader.land_pixels())
pixels = sorted(pixels)

# Map lat/lon to pixel integer for faster processing downstream
pxmap = pl.DataFrame({
    "pixel": range(len(pixels)), 
    "lat": [x[0] for x in pixels], 
    "lon": [x[1] for x in pixels]
})

loc_data = loc_data.join(pxmap, on=["lat", "lon"])

px_iterator = ClimateBinIterator(pxmap, bin_reader, year_from, year_to, block_size=1000)
blocks = []
for climate in tqdm(px_iterator):

    block = climate.join(loc_data, on="pixel").join(co2, on="year")

    # Convert units
    block = block.with_columns(
        pl.col("hurs") / 100,  # % -> ratio
        pl.col("pr") * 24 * 60 * 60,  # kg/m²/s -> mm/day; 1 kg of rain water spread over 1 square meter of surface is 1 mm in thickness
        pl.col("rsds") * 60 * 60 * 24 / 1e6,  # W/m² -> MJ/m²/day
        pl.col("tasmax") - 273.15,  # K -> °C
        pl.col("tasmin") - 273.15,  # K -> °C
    ).with_columns(
        tav = (pl.col("tasmax") + pl.col("tasmin")) / 2
    )

    block = process_climate_features(block, crop=corn, hd_mode=hd_mode)
    blocks.append(block)

# Finalize df
df = (
    pl.concat(blocks, how="vertical")
    .join(pxmap, on="pixel")
    .drop("pixel", "gs")
    .rename({
        "cwd": "CWD",
        "cdd": "CDD",
        "period": "PERIOD",
        "lat": "LAT",
        "lon": "LON"
    })
)

df.write_parquet(path_out)

print("done")
