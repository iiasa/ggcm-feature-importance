import os
import numpy as np
from pathlib import Path
from calendar import isleap


class ClimateBinReader:

    VALUE_BYTES: int = 4

    def __init__(self, data_dir: Path, landmap_path: Path, climate_vars: list, file_years: list = None):
        if not data_dir.exists():
            raise ValueError('Invalid data dir')
        if not landmap_path.exists():
            raise ValueError('Landmap file not found')
        if not climate_vars:
            raise ValueError('Provide list of climate variables')
        if file_years is None:
            file_years = [(1971, 1980, 3653), (1981, 1990, 3652), (1991, 2000, 3653), (2001, 2010, 3652), (2011, 2016, 2192)]
        self.data_dir = data_dir
        self.climate_vars = climate_vars
        self.file_years = file_years
        self.landmap_idx = self._init_landmap(landmap_path)
        self.year_offsets = self._init_day_offsets(self.file_years)

    @classmethod
    def read_landpixels(cls, path: Path):
        pos = {}
        counter = 0
        with open(path, mode='rb') as file:
            for lat in range(360):
                for lon in range(720):
                    px = np.frombuffer(file.read(4), dtype=np.int32)
                    if px > -1:
                        pos[(cls._idx_to_lat(lat), cls._idx_to_lon(lon))] = counter
                        counter += 1
        return pos

    def read(self, query: dict) -> dict:
        """
        Queries a set of binary climate files.
        
        Parameters
        ----------
        query: A query dictionary

        Examples
        --------
        >>> query = {
        >>>     (38.75, -119.25): np.array([1971, 1972, 1973]),
        >>>     (46.25, 6.25): np.array([1971, 1973, 1976]),
        >>>     ...
        >>> }
        >>> ClimateBinReader(...).read(query)
        """
        data = {}
        query_idx = self._query_locs_to_idx(query)
        file_years = self._filter_file_years(query_idx)

        for var_idx, var_name in enumerate(self.climate_vars):
            for year_from, year_to, num_days in file_years:
                with open(os.path.join(self.data_dir, f'gswp3-w5e5_obsclim_{var_name}_global_daily_{year_from}_{year_to}.bin'), mode='rb') as file:
                    for (idx, years), loc in zip(query_idx.items(), query.keys()):
                        loc_start = self.landmap_idx[idx] * num_days * self.VALUE_BYTES
                        for year in [x for x in years if year_from <= x <= year_to]:
                            file.seek(loc_start + self.year_offsets[year] * self.VALUE_BYTES)
                            leap = isleap(year)
                            days_in_year = 365 + leap
                            arr = np.frombuffer(file.read(days_in_year * self.VALUE_BYTES), count=days_in_year, dtype=np.float32)
                            while arr.shape[0] < 366:  # pad
                                arr = np.append(arr, np.nan)
                            key = (loc, year)
                            if not key in data:
                                data[key] = np.empty(shape=(366, len(self.climate_vars)), dtype=np.float32)
                            data[key][:, var_idx] = arr
        return data

    def land_pixels(self):
        locs = []
        for idx in self.landmap_idx.keys():
            locs.append(
                (self._idx_to_lat(idx[0]), self._idx_to_lon(idx[1]))
            )
        return locs

    def _lat_to_idx(self, lat):
        return (-lat + 89.75) * 2

    def _lon_to_idx(self, lon):
        return (lon + 179.75) * 2
    
    @staticmethod
    def _idx_to_lat(y):
        return -y/2 + 89.75

    @staticmethod
    def _idx_to_lon(x):
        return x/2 - 179.75

    def _query_locs_to_idx(self, query:dict):
        locs2 = {}
        for k, v in query.items():
            locs2[(int(self._lat_to_idx(k[0])), int(self._lon_to_idx(k[1])))] = v
        return locs2
    
    def _init_landmap(self, path: Path) -> dict:
        pos = {}
        counter = 0
        with open(path, mode='rb') as file:
            for lat in range(360):
                for lon in range(720):
                    px = np.frombuffer(file.read(4), dtype=np.int32)
                    if px > -1:
                        pos[(lat, lon)] = counter
                        counter += 1
        return pos

    def _filter_file_years(self, query):
        """
        Return only ranges that are actually used in locs.
        """
        ranges_filtered = []
        years = np.unique(np.hstack(list(query.values())))
        for year_from, year_to, num_days in self.file_years:
            if years[(years >= year_from) & (years <= year_to)].shape[0] > 0:
                ranges_filtered.append((year_from, year_to, num_days))
        return ranges_filtered

    def _init_day_offsets(self, file_years):
        offsets = {}
        for year_from, year_to, _ in file_years:
            counter = 0
            for year in np.arange(year_from, year_to + 1):
                offsets[year] = counter
                counter += 365 + isleap(year)
        return offsets
