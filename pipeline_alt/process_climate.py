from __future__ import annotations

import argparse
from pathlib import Path

from climate_pipeline.pipeline import PipelineConfig, process_climate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refactored climate-processing pipeline with Parquet output.")
    parser.add_argument("--source", choices=["binary", "netcdf"], required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--locfile", required=True, type=Path)
    parser.add_argument("--co2file", required=True, type=Path)
    parser.add_argument("--year-from", default=1971, type=int)
    parser.add_argument("--year-to", default=2015, type=int)
    parser.add_argument("--shift", default=0, type=int)
    parser.add_argument("--chunk-size", default=5000, type=int)
    parser.add_argument("--clim-year-from", default=1901, type=int)
    parser.add_argument("--clim-year-to", default=2016, type=int)
    parser.add_argument("--climdir", type=Path)
    parser.add_argument("--landmap", type=Path)
    parser.add_argument("--climate-file", action="append", default=[], help="Format: name=path/to/file.nc")
    parser.add_argument("--pdayfile", type=Path)
    parser.add_argument("--mdayfile", type=Path)
    return parser.parse_args()


def parse_climate_files(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --climate-file value: {value}")
        name, raw_path = value.split("=", 1)
        parsed[name] = Path(raw_path)
    return parsed


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        source=args.source,
        output_dir=args.output_dir,
        locfile=args.locfile,
        co2file=args.co2file,
        year_from=args.year_from,
        year_to=args.year_to,
        shift=args.shift,
        chunk_size=args.chunk_size,
        clim_year_from=args.clim_year_from,
        clim_year_to=args.clim_year_to,
        climdir=args.climdir,
        landmap=args.landmap,
        climate_files=parse_climate_files(args.climate_file),
        plant_day_file=args.pdayfile,
        maturity_day_file=args.mdayfile,
    )
    output_dir = process_climate(config)
    print(f"Wrote parquet parts to {output_dir}")


if __name__ == "__main__":
    main()
