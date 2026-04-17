from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import shutil

try:
    import polars as pl
except ImportError as exc:
    raise RuntimeError("The refactored pipeline requires `polars` to be installed.") from exc


@dataclass(frozen=True)
class LocationRecord:
    lat: float
    lon: float
    planting_day: int | None
    harvest_day: int | None
    elev: float
    phu: float
    prmt74: float


def read_locations(path: Path) -> dict[tuple[float, float], LocationRecord]:
    frame = pl.read_csv(path)
    rows: dict[tuple[float, float], LocationRecord] = {}
    for row in frame.iter_rows(named=True):
        lat = float(row["YLAT"])
        lon = float(row["XLON"])
        rows[(lat, lon)] = LocationRecord(
            lat=lat,
            lon=lon,
            planting_day=int(row["PLDOY"]) if row.get("PLDOY") not in (None, "") else None,
            harvest_day=int(row["HRDOY"]) if row.get("HRDOY") not in (None, "") else None,
            elev=float(row["ELEV"]),
            phu=float(row["PHU"]),
            prmt74=float(row["PRMT74"]),
        )
    return rows


def read_co2(path: Path, year_from: int, year_to: int) -> dict[int, float]:
    series: dict[int, float] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2 or not parts[0].lstrip("-").isdigit():
            continue
        year = int(parts[0])
        value = float(parts[1])
        if year >= year_from:
            series[year] = value

    if year_to not in series and len(series) >= 5:
        recent_years = sorted(series)[-5:]
        x = pl.Series(recent_years, dtype=pl.Float64).to_numpy()
        y = pl.Series([series[year] for year in recent_years], dtype=pl.Float64).to_numpy()
        slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
        intercept = y.mean() - slope * x.mean()
        series[year_to] = float(intercept + slope * year_to)

    return series


class ParquetChunkWriter:
    def __init__(self, output_dir: Path, overwrite: bool = False):
        self.output_dir = output_dir
        self.part_idx = 0
        if overwrite and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_rows(self, rows: Iterable[dict]) -> int:
        materialized = list(rows)
        if not materialized:
            return 0
        frame = pl.DataFrame(materialized)
        target = self.output_dir / f"part-{self.part_idx:05d}.parquet"
        frame.write_parquet(target, statistics=True)
        self.part_idx += 1
        return frame.height
