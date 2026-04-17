from __future__ import annotations

from calendar import isleap
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import numpy as np

from .aggregation import CLIMATE_VARS, ReferenceCrop, process_year
from .io import ParquetChunkWriter, read_co2, read_locations
from .readers import BinaryClimateReader, ClimateReader, GrowingSeasonCalendar, NetCDFClimateReader


@dataclass(frozen=True)
class PipelineConfig:
    source: str
    output_dir: Path
    locfile: Path
    co2file: Path
    year_from: int
    year_to: int
    shift: int = 0
    chunk_size: int = 5000
    overwrite_output: bool = True
    climate_vars: tuple[str, ...] = tuple(CLIMATE_VARS)
    climdir: Path | None = None
    landmap: Path | None = None
    climate_files: Mapping[str, Path] | None = None
    plant_day_file: Path | None = None
    maturity_day_file: Path | None = None
    clim_year_from: int = 1901
    clim_year_to: int = 2016


def process_climate(config: PipelineConfig) -> Path:
    reader = _build_reader(config)
    crop = ReferenceCrop.maize()
    locations = read_locations(config.locfile)
    co2_by_year = read_co2(config.co2file, config.year_from, config.year_to)
    calendar = _build_calendar(config)

    available_pixels = set(reader.land_pixels()) & set(locations)
    if calendar is not None:
        available_pixels &= calendar.pixels()

    writer = ParquetChunkWriter(config.output_dir, overwrite=config.overwrite_output)
    rows: list[dict] = []
    requested_years = np.arange(config.year_from, config.year_to + 1, dtype=int)

    for lat, lon in sorted(available_pixels):
        location = locations[(lat, lon)]
        climate_data = reader.read({(lat, lon): requested_years})

        for year in range(config.year_from + 1, config.year_to + 1):
            timing = _resolve_timing(location, calendar, lat, lon, year, config.shift)
            if timing is None:
                continue
            plant_day, harvest_day, crosses_year = timing

            if crosses_year:
                previous = climate_data.get(((lat, lon), year - 1))
                current = climate_data.get(((lat, lon), year))
                if previous is None or current is None:
                    continue
                climate_year = np.vstack([previous, current])
                harvest_idx = 366 + harvest_day
            else:
                current = climate_data.get(((lat, lon), year))
                if current is None:
                    continue
                climate_year = current
                harvest_idx = harvest_day

            aggregated = process_year(
                data_orig=climate_year,
                year=year,
                plant_day=plant_day,
                harvest_day=harvest_idx,
                lat=lat,
                elev=location.elev,
                phu=location.phu,
                prmt_74=location.prmt74,
                co2=co2_by_year.get(year, co2_by_year[max(co2_by_year)]),
                crop=crop,
                compute_growing_season=calendar is None,
            )
            if aggregated is None:
                continue

            rows.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "year": year,
                    "IA": crosses_year,
                    **aggregated,
                }
            )
            if len(rows) >= config.chunk_size:
                writer.write_rows(rows)
                rows.clear()

    if rows:
        writer.write_rows(rows)
    return config.output_dir


def _build_reader(config: PipelineConfig) -> ClimateReader:
    if config.source == "binary":
        if config.climdir is None or config.landmap is None:
            raise ValueError("Binary mode requires `climdir` and `landmap`.")
        return BinaryClimateReader(
            data_dir=config.climdir,
            landmap_path=config.landmap,
            climate_vars=list(config.climate_vars),
        )
    if config.source == "netcdf":
        if not config.climate_files:
            raise ValueError("NetCDF mode requires `climate_files`.")
        return NetCDFClimateReader(config.climate_files, list(config.climate_vars))
    raise ValueError(f"Unsupported climate source: {config.source}")


def _build_calendar(config: PipelineConfig) -> GrowingSeasonCalendar | None:
    if config.plant_day_file is None or config.maturity_day_file is None:
        return None
    return GrowingSeasonCalendar.from_netcdf(
        plant_day_path=config.plant_day_file,
        maturity_day_path=config.maturity_day_file,
        year_from=config.clim_year_from,
        year_to=config.clim_year_to,
    )


def _resolve_timing(location, calendar, lat: float, lon: float, year: int, shift: int) -> tuple[int, int, bool] | None:
    if calendar is None:
        if location.planting_day is None or location.harvest_day is None:
            raise ValueError("Location table must include PLDOY and HRDOY when no calendar files are supplied.")
        return location.planting_day, location.harvest_day, location.planting_day > location.harvest_day

    entry = calendar.lookup(lat, lon, year)
    if entry is None:
        return None

    plant_day = entry.plant_day
    days_in_year = 365 + isleap(year)
    harvest_day = (plant_day + entry.season_length) % days_in_year

    if plant_day > harvest_day and shift != 0:
        shifted = calendar.lookup(lat, lon, year + shift)
        if shifted is None:
            return None
        plant_day = shifted.plant_day
        harvest_day = (plant_day + shifted.season_length) % days_in_year

    return plant_day, harvest_day, plant_day > harvest_day
