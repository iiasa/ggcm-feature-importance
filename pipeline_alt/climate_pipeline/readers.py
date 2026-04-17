from __future__ import annotations

from abc import ABC, abstractmethod
from calendar import isleap
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Mapping
import numpy as np
import xarray as xr


class ClimateReader(ABC):
    @abstractmethod
    def land_pixels(self) -> list[tuple[float, float]]:
        raise NotImplementedError

    @abstractmethod
    def read(self, query: Mapping[tuple[float, float], np.ndarray]) -> dict[tuple[tuple[float, float], int], np.ndarray]:
        raise NotImplementedError


class BinaryClimateReader(ClimateReader):
    VALUE_BYTES = 4

    def __init__(
        self,
        data_dir: Path,
        landmap_path: Path,
        climate_vars: list[str],
        file_years: list[tuple[int, int, int]] | None = None,
    ):
        if file_years is None:
            file_years = [
                (1971, 1980, 3653),
                (1981, 1990, 3652),
                (1991, 2000, 3653),
                (2001, 2010, 3652),
                (2011, 2016, 2192),
            ]
        self.data_dir = data_dir
        self.landmap_path = landmap_path
        self.climate_vars = climate_vars
        self.file_years = file_years
        self.landmap_idx = self._init_landmap()
        self.year_offsets = self._init_day_offsets()
        self.year_ranges = self._init_year_ranges()

    def land_pixels(self) -> list[tuple[float, float]]:
        return [(self._idx_to_lat(lat), self._idx_to_lon(lon)) for lat, lon in self.landmap_idx]

    def read(self, query: Mapping[tuple[float, float], np.ndarray]) -> dict[tuple[tuple[float, float], int], np.ndarray]:
        query_items = [
            (
                (lat, lon),
                (int(self._lat_to_idx(lat)), int(self._lon_to_idx(lon))),
                np.asarray(years, dtype=int),
            )
            for (lat, lon), years in query.items()
        ]

        data: dict[tuple[tuple[float, float], int], np.ndarray] = {}
        for var_idx, var_name in enumerate(self.climate_vars):
            memmaps = self._memmaps_for_var(var_name)
            for loc, idx, years in query_items:
                land_pos = self.landmap_idx.get(idx)
                if land_pos is None:
                    continue
                for year in years:
                    year_from, year_to, _ = self.year_ranges[year]
                    block = memmaps[(year_from, year_to)]
                    start = self.year_offsets[year]
                    days = 365 + isleap(year)
                    values = np.full(366, np.nan, dtype=np.float32)
                    values[:days] = block[land_pos, start:start + days]
                    key = (loc, int(year))
                    if key not in data:
                        data[key] = np.empty((366, len(self.climate_vars)), dtype=np.float32)
                    data[key][:, var_idx] = values
        return data

    @cached_property
    def _land_pixels_count(self) -> int:
        return len(self.landmap_idx)

    def _init_landmap(self) -> dict[tuple[int, int], int]:
        positions: dict[tuple[int, int], int] = {}
        counter = 0
        with self.landmap_path.open("rb") as handle:
            for lat_idx in range(360):
                for lon_idx in range(720):
                    value = np.frombuffer(handle.read(4), dtype=np.int32)
                    if value > -1:
                        positions[(lat_idx, lon_idx)] = counter
                        counter += 1
        return positions

    def _init_day_offsets(self) -> dict[int, int]:
        offsets: dict[int, int] = {}
        for year_from, year_to, _ in self.file_years:
            counter = 0
            for year in range(year_from, year_to + 1):
                offsets[year] = counter
                counter += 365 + isleap(year)
        return offsets

    def _init_year_ranges(self) -> dict[int, tuple[int, int, int]]:
        mapping: dict[int, tuple[int, int, int]] = {}
        for year_from, year_to, num_days in self.file_years:
            for year in range(year_from, year_to + 1):
                mapping[year] = (year_from, year_to, num_days)
        return mapping

    @cached_property
    def _memmap_cache(self) -> dict[str, dict[tuple[int, int], np.memmap]]:
        cache: dict[str, dict[tuple[int, int], np.memmap]] = {}
        for var_name in self.climate_vars:
            blocks: dict[tuple[int, int], np.memmap] = {}
            for year_from, year_to, num_days in self.file_years:
                path = self.data_dir / f"gswp3-w5e5_obsclim_{var_name}_global_daily_{year_from}_{year_to}.bin"
                blocks[(year_from, year_to)] = np.memmap(
                    path,
                    dtype=np.float32,
                    mode="r",
                    shape=(self._land_pixels_count, num_days),
                )
            cache[var_name] = blocks
        return cache

    def _memmaps_for_var(self, var_name: str) -> dict[tuple[int, int], np.memmap]:
        return self._memmap_cache[var_name]

    @staticmethod
    def _lat_to_idx(lat: float) -> float:
        return (-lat + 89.75) * 2

    @staticmethod
    def _lon_to_idx(lon: float) -> float:
        return (lon + 179.75) * 2

    @staticmethod
    def _idx_to_lat(idx: int) -> float:
        return -idx / 2 + 89.75

    @staticmethod
    def _idx_to_lon(idx: int) -> float:
        return idx / 2 - 179.75


class NetCDFClimateReader(ClimateReader):
    def __init__(self, climate_files: Mapping[str, Path], climate_vars: list[str]):
        self.climate_files = {name: Path(path) for name, path in climate_files.items()}
        self.climate_vars = climate_vars
        missing = [var for var in climate_vars if var not in self.climate_files]
        if missing:
            raise ValueError(f"Missing NetCDF files for variables: {missing}")

    @cached_property
    def datasets(self) -> dict[str, xr.Dataset]:
        return {name: xr.open_dataset(path) for name, path in self.climate_files.items()}

    def land_pixels(self) -> list[tuple[float, float]]:
        reference = self._variable_array(self.climate_vars[0])
        first_step = reference.isel(time=0)
        valid = np.isfinite(first_step.values)
        lat_idx, lon_idx = np.where(valid)
        lats = first_step["lat"].values
        lons = first_step["lon"].values
        return [(float(lats[i]), float(lons[j])) for i, j in zip(lat_idx.tolist(), lon_idx.tolist())]

    def read(self, query: Mapping[tuple[float, float], np.ndarray]) -> dict[tuple[tuple[float, float], int], np.ndarray]:
        data: dict[tuple[tuple[float, float], int], np.ndarray] = {}
        for (lat, lon), years in query.items():
            for var_idx, var_name in enumerate(self.climate_vars):
                point = self._select_point(self._variable_array(var_name), lat, lon)
                if "time" not in point.coords:
                    raise ValueError(f"Variable `{var_name}` is missing a time coordinate.")
                years_at_point = point["time"].dt.year.values
                values = np.asarray(point.values, dtype=np.float32)
                for year in years:
                    mask = years_at_point == year
                    if not np.any(mask):
                        continue
                    daily = values[mask]
                    padded = np.full(366, np.nan, dtype=np.float32)
                    padded[: daily.shape[0]] = daily
                    key = ((lat, lon), int(year))
                    if key not in data:
                        data[key] = np.empty((366, len(self.climate_vars)), dtype=np.float32)
                    data[key][:, var_idx] = padded
        return data

    def _variable_array(self, var_name: str) -> xr.DataArray:
        dataset = self.datasets[var_name]
        if var_name in dataset.data_vars:
            return dataset[var_name]
        if len(dataset.data_vars) == 1:
            return dataset[next(iter(dataset.data_vars))]
        raise ValueError(f"Could not infer data variable for `{var_name}` in {self.climate_files[var_name]}.")

    @staticmethod
    def _select_point(data_array: xr.DataArray, lat: float, lon: float) -> xr.DataArray:
        try:
            return data_array.sel(lat=lat, lon=lon)
        except Exception:
            return data_array.sel(lat=lat, lon=lon, method="nearest")


@dataclass(frozen=True)
class CalendarEntry:
    plant_day: int
    season_length: int


class GrowingSeasonCalendar:
    def __init__(self, entries: dict[tuple[float, float, int], CalendarEntry]):
        self.entries = entries

    def lookup(self, lat: float, lon: float, year: int) -> CalendarEntry | None:
        return self.entries.get((lat, lon, year))

    def pixels(self) -> set[tuple[float, float]]:
        return {(lat, lon) for lat, lon, _ in self.entries}

    @classmethod
    def from_netcdf(
        cls,
        plant_day_path: Path,
        maturity_day_path: Path,
        year_from: int,
        year_to: int,
    ) -> "GrowingSeasonCalendar":
        plant_ds = xr.open_dataset(plant_day_path, decode_times=False)
        maturity_ds = xr.open_dataset(maturity_day_path, decode_times=False)

        plant_var = next(name for name in plant_ds.data_vars if "plantday" in name or "pday" in name)
        maturity_var = next(name for name in maturity_ds.data_vars if "matyday" in name or "mday" in name)

        plant_da = plant_ds[plant_var].transpose("time", "lat", "lon")
        maturity_da = maturity_ds[maturity_var].transpose("time", "lat", "lon")

        years = np.arange(year_from, year_to + 1, dtype=int)
        if plant_da.sizes["time"] != years.shape[0]:
            raise ValueError("Calendar year range does not match the NetCDF time dimension.")

        lats = plant_da["lat"].values
        lons = plant_da["lon"].values
        pd_values = np.asarray(plant_da.values)
        len_values = np.asarray(maturity_da.values)

        entries: dict[tuple[float, float, int], CalendarEntry] = {}
        valid = np.isfinite(pd_values) & np.isfinite(len_values)
        for t_idx, lat_idx, lon_idx in np.argwhere(valid):
            entries[(float(lats[lat_idx]), float(lons[lon_idx]), int(years[t_idx]))] = CalendarEntry(
                plant_day=int(pd_values[t_idx, lat_idx, lon_idx]),
                season_length=int(len_values[t_idx, lat_idx, lon_idx]),
            )
        return cls(entries)
