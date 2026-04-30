"""
Components of climate pipeline.
Required:
    * Reader (ClimateBinReader or ClimateNCReader); a class that provides functionlity
      to stream climate features data per N pixels blockwise. Use ClimateBinReader for 
      performance and ClimateNCReader for convenience.
    * Path to a planting date file
    * If hd_mode != "dynamic", path to a harvest date file. This mday file usually 
      contains the GS length, not the actual HD, which is what is assumed here.
    * An auxilary data file containing PHU per location, among other location (but not time)
      -specific quantities.
    * Path to a global CO2 timeseries.
HD modes:
    * strict: Defines the GS as the period between calendar PD and HD.
    * semi_dynamic: Defines the GS as the period between PD and the HD, calculated by the 
      GDD approach *or* max. 21 days after the calendar HD.
    * dynamic: Defines the GS as the period between PD and GDD-calculated HD.
General:
    * The master file is a polars DataFrame containing LAT, LON of the pixels to be processed, 
      the crop calendar PD and [HD], PHU for HUI calculation, ELEV and PRMT74 for PET.

"""
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression

from data.nikbins import ClimateBinReader

class ClimateBinIterator:

    def __init__(
        self, 
        pxmap: pl.DataFrame,
        reader: ClimateBinReader,
        year_from: int,
        year_to: int,
        block_size: int = 1000,
    ):
        self.pxmap = pxmap
        self.reader = reader
        self.year_from = year_from
        self.year_to = year_to
        self.block_size = block_size
        self.years = np.arange(year_from, year_to + 1)
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self
    
    def __len__(self) -> int:
        return int(np.ceil(self.pxmap.height / self.block_size))

    def __next__(self):
        px_from = int(self.block_size * self.i)
        px_to = min(
            int(self.block_size * (self.i + 1)), 
            self.pxmap.height
        )
        
        if px_from >= self.pxmap.height:
            raise StopIteration
        
        latlons = self.pxmap.filter(
            (pl.col("pixel") >= px_from) & (pl.col("pixel") < px_to)
        ).select(["lat", "lon"]).rows()
        results = self.reader.read(query={(x[0], x[1]): self.years for x in latlons})
        px_ts = [
            np.stack([results[((px[0], px[1]), year)] for year in self.years], axis=0) 
            for px in latlons
        ]
        df = np.stack(px_ts, axis=0)  # n_pixels x n_years x 366 (days) x n_features

        n_pixel, n_year, n_day, n_feature = df.shape
        self.i += 1

        return pl.DataFrame(
            {
                "pixel": np.repeat(np.arange(px_from, px_to), n_year * n_day),
                "year": np.tile(np.repeat(self.years, n_day), n_pixel),
                "day": np.tile(np.arange(1, n_day + 1), n_pixel * n_year),
                **{
                    self.reader.climate_vars[i]: df[..., i].reshape(-1)
                    for i in range(n_feature)
                },
            }
        )

def read_co2(path: Path, year_from: int) -> pl.DataFrame:
    data = pd.read_fwf(path)
    data.columns = ['year', 'co2']
    data.set_index('year', inplace=True)
    data = data.loc[year_from:]

    if not 2015 in data.index:  # Project to 2015
        data.loc[2015] = LinearRegression(fit_intercept=True) \
        .fit(np.array(data.iloc[-5:].index).reshape(-1, 1), data.iloc[-5:].values) \
        .predict(np.array([[2015]])).item()

    return pl.DataFrame(
            data.reset_index(), 
            schema={
                "year": pl.UInt16,
                "co2": pl.Float64
            }
        )
