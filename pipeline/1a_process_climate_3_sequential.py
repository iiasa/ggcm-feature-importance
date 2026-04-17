"""
Process daily climate features directly from ISIMIP-style NetCDF files.

This script mirrors the behavior of ``1a_process_climate_2.py`` but replaces
the proprietary binary reader with direct NetCDF access via xarray.
"""

from __future__ import annotations

import argparse
import calendar
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import unepic


class VarIdx:
    HURS = 0
    PRCP = 1
    RSDS = 2
    WIND = 3
    TMAX = 4
    TMIN = 5


CLIMATE_VARS = [
    "hurs",
    "pr",
    "rsds",
    "sfcwind",
    "tasmax",
    "tasmin",
]


@dataclass(frozen=True)
class PixelContext:
    plant_day: int
    harvest_day: int
    elev: float
    phu: float
    prmt_74: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year_from", default=1971, type=int)
    parser.add_argument("--year_to", default=2015, type=int)
    parser.add_argument("--clim_year_from", default=1901, type=int)
    parser.add_argument("--clim_year_to", default=2016, type=int)
    parser.add_argument("--shift", default=0, type=int)
    parser.add_argument(
        "--nc_dir",
        default=ROOT / "data" / "input" / "climate",
        type=Path,
        help="Directory containing daily climate NetCDF files.",
    )
    parser.add_argument(
        "--locfile",
        default=ROOT / "data" / "input" / "CLIMATE_LAT_PD_HD_PHU_ELEV_PRMT74_mai_noirr_fH_v3c.csv",
        type=Path,
    )
    parser.add_argument(
        "--co2file",
        default=ROOT / "data" / "input" / "co2_historical_annual_1765_2014.txt",
        type=Path,
    )
    parser.add_argument(
        "--mdayfile",
        default=None,
        type=Path,
        help="Optional maturity-day NetCDF.",
    )
    parser.add_argument(
        "--pdayfile",
        default=None,
        type=Path,
        help="Optional planting-day NetCDF.",
    )
    parser.add_argument(
        "--out",
        default=ROOT / "data" / "output" / "generated" / "climate_EPIC-IIASA_maize_rf_nc.h5",
        type=Path,
    )
    parser.add_argument(
        "--spatial_tolerance",
        default=0.26,
        type=float,
        help="Maximum allowed distance between requested and selected lat/lon coordinates.",
    )
    return parser.parse_args()


def read_co2(path: Path, year_from: int) -> dict[int, float]:
    data = pd.read_fwf(path)
    data.columns = ["YR", "CO2"]
    data.set_index("YR", inplace=True)
    data = data.loc[year_from:]
    if 2015 not in data.index:
        data.loc[2015] = (
            LinearRegression(fit_intercept=True)
            .fit(np.array(data.iloc[-5:].index).reshape(-1, 1), data.iloc[-5:].values)
            .predict(np.array([[2015]]))
            .item()
        )
    return data["CO2"].to_dict()


def count_consecutive(arr: np.ndarray) -> np.ndarray:
    return np.diff(np.where(np.concatenate(([arr[0]], arr[:-1] != arr[1:], [True])))[0])[::2]


def process_year(
    data_orig: np.ndarray,
    year: int,
    pd_idx: int,
    hd_idx: int,
    lat: float,
    elev: float,
    phu: float,
    prmt_74: float,
    co2: float,
    crop: unepic.Crop,
    compute_gs: bool = True,
) -> dict[str, float] | None:
    """
    Aggregate one growing season from a 2D array of shape (days, climate_features).
    """

    del year
    data = data_orig.copy()

    data[:, VarIdx.HURS] = data[:, VarIdx.HURS] / 100.0
    data[:, VarIdx.PRCP] = data[:, VarIdx.PRCP] * 24 * 60 * 60
    data[:, VarIdx.RSDS] = data[:, VarIdx.RSDS] * 60 * 60 * 24 / 1e6
    data[:, VarIdx.TMAX] = data[:, VarIdx.TMAX] - 273.15
    data[:, VarIdx.TMIN] = data[:, VarIdx.TMIN] - 273.15

    season, hui, gdd = unepic.compute_season(
        data[:, VarIdx.TMIN],
        data[:, VarIdx.TMAX],
        pd_idx,
        hd_idx,
        crop.tbsc,
        crop.gmhu,
        phu,
        dynamic=compute_gs,
    )
    if season is None:
        return None

    lai, cht = unepic.lai(
        hui=hui,
        dlap1=crop.dlap1,
        dlap2=crop.dlap2,
        dlai=crop.dlai,
        rlad=crop.rlad,
        dmla=crop.dmla,
        hmx=crop.hmx,
    )

    aggs: dict[str, float] = {}
    for period, idx in season.items():
        d = data[idx]
        plen = d.shape[0]
        if plen == 0:
            continue

        tav = (d[:, VarIdx.TMAX] + d[:, VarIdx.TMIN]) / 2.0
        pet = unepic.pet_pm(
            tav=tav,
            rad=d[:, VarIdx.RSDS],
            ws=d[:, VarIdx.WIND],
            hur=d[:, VarIdx.HURS],
            lai=lai[idx],
            cht=cht[idx],
            doy=(np.arange(idx.start, idx.start + plen) % 366) + 1,
            elev=elev,
            salb=0.15,
            lat=lat,
            co2=co2,
            vpth=crop.vpth,
            gsi=crop.gsi,
            vpd2=crop.gsi,
            prmt_74=prmt_74,
        )

        cmd = pet - d[:, VarIdx.PRCP]
        wet = d[:, VarIdx.PRCP] > 1.0
        dry = ~wet

        wet_sum = np.nansum(wet)
        dry_sum = np.nansum(dry)
        cwd_sum = np.max(count_consecutive(wet)) if wet_sum > 0 else 0
        cdd_sum = np.max(count_consecutive(dry)) if dry_sum > 0 else 0

        aggregates = {
            "TMXav": np.nanmean(d[:, VarIdx.TMAX]),
            "TMNav": np.nanmean(d[:, VarIdx.TMIN]),
            "TAVav": np.nanmean(tav),
            "PRCPsum": np.nansum(d[:, VarIdx.PRCP]),
            "RADsum": np.nansum(d[:, VarIdx.RSDS]),
            "WSDav": np.nanmean(d[:, VarIdx.WIND]),
            "HURav": np.nanmean(d[:, VarIdx.HURS]),
            "PETsum": np.nansum(pet),
            "GDDsum": np.nansum(gdd[idx]),
            "CMDsum": np.nansum(cmd),
            "LEN": plen,
            "HUIeop": hui[idx][-1],
        }

        fracs = {
            "HDD": np.nansum(d[:, VarIdx.TMAX] >= 30),
            "KDD": np.nansum(d[:, VarIdx.TMAX] >= 39),
            "FRT": np.nansum(d[:, VarIdx.TMIN] <= 0),
            "ICE": np.nansum(d[:, VarIdx.TMAX] <= 0),
            "R10": np.nansum(d[:, VarIdx.PRCP] >= 10.0),
            "R20": np.nansum(d[:, VarIdx.PRCP] >= 20.0),
            "WET": wet_sum,
            "DRY": dry_sum,
            "CWD": cwd_sum,
            "CDD": cdd_sum,
            "CMDlt0": np.nansum(cmd < 0),
        }

        aggs |= {f"{k}{period}": v for k, v in aggregates.items()}
        aggs |= {f"{k}frac{period}": v / plen for k, v in fracs.items()}

    return aggs


def get_int_dtype(value_range: float, signed: bool):
    if not signed:
        if value_range <= 255:
            return np.uint8
        if value_range <= 65535:
            return np.uint16
        return np.uint32

    half_range = value_range / 2
    if half_range <= 127:
        return np.int8
    if half_range <= 32767:
        return np.int16
    return np.int32


def flush_to_disk(data: dict[tuple[float, float, int], dict[str, float]], path_target: Path) -> None:
    df = pd.DataFrame(data).T
    df.index.set_names(("lat", "lon", "year"), inplace=True)

    dtypes = {}
    for var in df.columns:
        maxint = df[var].max(axis=0) * 100
        minint = df[var].min(axis=0) * 100
        dtypes[var] = get_int_dtype(maxint - minint, minint < 0)

    path_target.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(path_target, key="lat", mode="w", complevel=5)


def build_reference_crop() -> unepic.Crop:
    crop = unepic.Crop(
        wa=40.0,
        tbsc=8.0,
        dlai=0.8,
        rlad=1.0,
        dmla=6.0,
        dlap1=15.05,
        dlap2=50.95,
        top=25.0,
        rdmx=2.0,
        hi=0.5,
        hmx=2.0,
        vpth=0.5,
        vpd2=0.071,
        gsi=0.007,
        gmhu=100.0,
    )
    crop.dlap1, crop.dlap2 = unepic.ssolve(
        int(crop.dlap1) * 0.01,
        crop.dlap1 - int(crop.dlap1),
        int(crop.dlap2) * 0.01,
        crop.dlap2 - int(crop.dlap2),
    )
    return crop


def _find_nc_files(nc_dir: Path, variable: str) -> list[Path]:
    files = sorted(nc_dir.glob(f"*_{variable}_global_daily_*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found for variable '{variable}' in {nc_dir}")
    return files


def _find_coord_name(da: xr.DataArray, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in da.coords:
            return candidate
        if candidate in da.dims:
            return candidate
    raise KeyError(f"Could not find any of {candidates} in DataArray coordinates {list(da.coords)}")


def _find_data_var(ds: xr.Dataset, expected_name: str) -> str:
    if expected_name in ds.data_vars:
        return expected_name
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise KeyError(f"Could not resolve data variable for '{expected_name}'. Available: {list(ds.data_vars)}")


def _normalize_lon(lon: float, lon_values: np.ndarray) -> float:
    lon_min = float(np.nanmin(lon_values))
    lon_max = float(np.nanmax(lon_values))
    if lon_min >= 0 and lon < 0:
        return lon % 360
    if lon_max <= 180 and lon > 180:
        return ((lon + 180) % 360) - 180
    return lon


def _ensure_time_sorted(da: xr.DataArray, time_name: str) -> xr.DataArray:
    if da.indexes[time_name].is_monotonic_increasing:
        return da
    return da.sortby(time_name)


class NetCDFClimateStore:
    def __init__(self, nc_dir: Path, climate_vars: list[str], spatial_tolerance: float = 0.26):
        self.nc_dir = nc_dir
        self.climate_vars = climate_vars
        self.spatial_tolerance = spatial_tolerance
        self.datasets: dict[str, xr.Dataset] = {}
        self.var_names: dict[str, str] = {}
        self.lat_names: dict[str, str] = {}
        self.lon_names: dict[str, str] = {}
        self.time_names: dict[str, str] = {}

    def open(self) -> None:
        for var in self.climate_vars:
            files = _find_nc_files(self.nc_dir, var)
            ds = xr.open_mfdataset(
                files,
                combine="by_coords",
                decode_times=True,
                use_cftime=True,
                chunks={"time": 366},
            )
            data_var = _find_data_var(ds, var)
            da = ds[data_var]
            self.datasets[var] = ds
            self.var_names[var] = data_var
            self.lat_names[var] = _find_coord_name(da, ("lat", "latitude", "y"))
            self.lon_names[var] = _find_coord_name(da, ("lon", "longitude", "x"))
            self.time_names[var] = _find_coord_name(da, ("time",))

    def close(self) -> None:
        for ds in self.datasets.values():
            ds.close()

    def select_pixel(self, variable: str, lat: float, lon: float) -> xr.DataArray:
        ds = self.datasets[variable]
        data_var = self.var_names[variable]
        lat_name = self.lat_names[variable]
        lon_name = self.lon_names[variable]

        da = ds[data_var]
        lon_values = ds[lon_name].values
        lon_query = _normalize_lon(lon, lon_values)
        pixel = da.sel({lat_name: lat, lon_name: lon_query}, method="nearest")

        selected_lat = float(pixel[lat_name].item())
        selected_lon = float(pixel[lon_name].item())
        if abs(selected_lat - lat) > self.spatial_tolerance:
            raise KeyError(f"Latitude mismatch for ({lat}, {lon}): selected {selected_lat}")
        if abs(selected_lon - lon_query) > self.spatial_tolerance:
            raise KeyError(f"Longitude mismatch for ({lat}, {lon}): selected {selected_lon}")

        return _ensure_time_sorted(pixel, self.time_names[variable])

    def extract_year_array(self, pixel_series: dict[str, xr.DataArray], year: int) -> np.ndarray | None:
        arrays = []
        length = None
        for var in self.climate_vars:
            time_name = self.time_names[var]
            series = pixel_series[var]
            year_slice = series.sel({time_name: slice(f"{year}-01-01", f"{year}-12-31")})
            values = np.asarray(year_slice.values, dtype=float)
            if values.ndim != 1:
                values = np.ravel(values)
            if length is None:
                length = values.shape[0]
            elif values.shape[0] != length:
                warnings.warn(f"Skipping year {year}: inconsistent day count for variable '{var}'")
                return None
            arrays.append(values)

        if length is None or length == 0:
            return None
        data = np.column_stack(arrays)
        if np.all(np.isnan(data)):
            return None
        return data


def read_calendar(pday_path: Path, mday_path: Path, year_from: int, year_to: int) -> pd.DataFrame:
    ds = xr.open_mfdataset([pday_path, mday_path], decode_times=False)
    try:
        ds = ds.assign_coords(time=np.arange(year_from, year_to + 1))
        df = ds.to_dataframe().dropna(how="all")
    finally:
        ds.close()

    rename_map = {}
    for column in df.columns:
        lower = column.lower()
        if "plantday" in lower:
            rename_map[column] = "pd"
        elif "matyday" in lower:
            rename_map[column] = "len"
    df = df.rename(columns=rename_map)

    if "pd" not in df.columns or "len" not in df.columns:
        raise KeyError(f"Could not resolve planting/maturity columns from {list(df.columns)}")

    if list(df.index.names) != ["lon", "lat", "time"]:
        warnings.warn("Swapping growing season calendar coordinates to (lon, lat, time)")
        df.index = df.index.reorder_levels(["lon", "lat", "time"])

    return df[["pd", "len"]]


def resolve_pixel_context(row: pd.Series, calendar_mode: bool) -> PixelContext:
    if "PLDOY" in row and "HRDOY" in row:
        plant_day = int(row["PLDOY"])
        harvest_day = int(row["HRDOY"])
    elif not calendar_mode:
        raise ValueError("Location table must include PLDOY and HRDOY when no calendar files are supplied.")
    else:
        plant_day = -1
        harvest_day = -1

    return PixelContext(
        plant_day=plant_day,
        harvest_day=harvest_day,
        elev=float(row["ELEV"]),
        phu=float(row["PHU"]),
        prmt_74=float(row["PRMT74"]),
    )


def process_pixel(
    lat: float,
    lon: float,
    years: np.ndarray,
    location: PixelContext,
    climate_store: NetCDFClimateStore,
    crop: unepic.Crop,
    co2_data: dict[int, float],
    calendar_df: pd.DataFrame | None,
    shift: int,
    compute_gs: bool,
) -> dict[tuple[float, float, int], dict[str, float]]:
    try:
        pixel_series = {var: climate_store.select_pixel(var, lat, lon) for var in CLIMATE_VARS}
    except KeyError:
        return {}

    annual_arrays = {}
    for year in years:
        annual_arrays[year] = climate_store.extract_year_array(pixel_series, int(year))

    results: dict[tuple[float, float, int], dict[str, float]] = {}
    for year in range(int(years.min()) + 1, int(years.max()) + 1):
        plant_day = location.plant_day
        harvest_day = location.harvest_day

        if calendar_df is not None:
            if (lon, lat, year) not in calendar_df.index:
                continue
            gs_len, plant_day = calendar_df.loc[(lon, lat, year), ["len", "pd"]].astype(int)
            days_in_year = 365 + calendar.isleap(year)
            harvest_day = (plant_day + gs_len) % days_in_year

            if plant_day > harvest_day and shift != 0:
                shifted_key = (lon, lat, year + shift)
                if shifted_key not in calendar_df.index:
                    continue
                gs_len, plant_day = calendar_df.loc[shifted_key, ["len", "pd"]].astype(int)
                harvest_day = (plant_day + gs_len) % days_in_year

        if plant_day > harvest_day:
            previous = annual_arrays.get(year - 1)
            current = annual_arrays.get(year)
            if previous is None or current is None:
                continue
            data_year = np.vstack([previous, current])
            harvest_idx = 366 + harvest_day
            ia = True
        else:
            data_year = annual_arrays.get(year)
            if data_year is None:
                continue
            harvest_idx = harvest_day
            ia = False

        aggregated = process_year(
            data_orig=data_year,
            year=year,
            pd_idx=plant_day,
            hd_idx=harvest_idx,
            lat=lat,
            elev=location.elev,
            phu=location.phu,
            prmt_74=location.prmt_74,
            co2=co2_data.get(year, co2_data[max(co2_data)]),
            crop=crop,
            compute_gs=compute_gs,
        )
        if aggregated is None:
            continue
        results[(lat, lon, year)] = aggregated | {"IA": ia}

    return results


def main() -> None:
    args = parse_args()

    np.random.seed(42)
    random.seed(42)

    calendar_mode = args.mdayfile is not None and args.pdayfile is not None
    if calendar_mode:
        print("Reading growing season from pdayfile & mdayfile.")
    else:
        print("Estimating growing season from reference crop.")
    print("Processing")

    crop = build_reference_crop()
    co2_data = read_co2(args.co2file, args.year_from)
    loc_data = pd.read_csv(args.locfile)
    loc_data.set_index(["YLAT", "XLON"], inplace=True)

    calendar_df = None
    if calendar_mode:
        calendar_df = read_calendar(
            pday_path=args.pdayfile,
            mday_path=args.mdayfile,
            year_from=args.clim_year_from,
            year_to=args.clim_year_to,
        )

    climate_store = NetCDFClimateStore(
        nc_dir=args.nc_dir,
        climate_vars=CLIMATE_VARS,
        spatial_tolerance=args.spatial_tolerance,
    )
    climate_store.open()

    try:
        available_pixels = set(loc_data.index)
        if calendar_df is not None:
            calendar_pixels = set(zip(calendar_df.index.get_level_values("lat"), calendar_df.index.get_level_values("lon")))
            available_pixels &= calendar_pixels

        years = np.arange(args.year_from, args.year_to + 1, dtype=int)
        data: dict[tuple[float, float, int], dict[str, float]] = {}

        for lat, lon in tqdm(sorted(available_pixels)):
            location = resolve_pixel_context(loc_data.loc[(lat, lon)], calendar_mode)
            pixel_results = process_pixel(
                lat=lat,
                lon=lon,
                years=years,
                location=location,
                climate_store=climate_store,
                crop=crop,
                co2_data=co2_data,
                calendar_df=calendar_df,
                shift=args.shift,
                compute_gs=not calendar_mode,
            )
            data.update(pixel_results)

        if data:
            flush_to_disk(data, args.out)
        print("done")
    finally:
        climate_store.close()


if __name__ == "__main__":
    main()
