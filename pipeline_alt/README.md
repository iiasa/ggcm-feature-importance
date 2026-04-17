This folder contains a refactored climate-processing pipeline that preserves the core growing-season aggregation logic from `pipeline/1a_process_climate_2.py` while separating I/O, aggregation, and output concerns.

Highlights:

- Reads climate data through a pluggable reader interface.
- Supports both the legacy binary climate files and direct NetCDF inputs.
- Uses Polars for tabular I/O and writes chunked Parquet output.
- Streams rows to disk in chunks instead of collecting one giant in-memory dictionary.

Main entrypoint:

- `new/process_climate.py`

Example usage with legacy binaries:

```powershell
python new/process_climate.py `
  --source binary `
  --climdir data/input/climate `
  --landmap data/input/climate/land-map.bin `
  --locfile data/input/CLIMATE_LAT_PD_HD_PHU_ELEV_PRMT74_mai_noirr_fH_v3c.csv `
  --co2file data/input/co2_historical_annual_1765_2014.txt `
  --output-dir data/output/generated/climate_parquet `
  --year-from 1971 `
  --year-to 2015
```

Example usage with original NetCDF files:

```powershell
python new/process_climate.py `
  --source netcdf `
  --climate-file hurs=path/to/hurs.nc `
  --climate-file pr=path/to/pr.nc `
  --climate-file rsds=path/to/rsds.nc `
  --climate-file sfcwind=path/to/sfcwind.nc `
  --climate-file tasmax=path/to/tasmax.nc `
  --climate-file tasmin=path/to/tasmin.nc `
  --locfile data/input/CLIMATE_LAT_PD_HD_PHU_ELEV_PRMT74_mai_noirr_fH_v3c.csv `
  --co2file data/input/co2_historical_annual_1765_2014.txt `
  --output-dir data/output/generated/climate_parquet `
  --year-from 1971 `
  --year-to 2015
```

Calendar mode:

- Pass both `--pdayfile` and `--mdayfile` to use GGCM-specific planting and maturity timing.
- Without them, the pipeline estimates season length dynamically using the EPIC-style biophysical functions.

Output layout:

- The writer creates a directory of Parquet parts such as `part-00000.parquet`.
- This is intended for Polars lazy scans via `pl.scan_parquet(".../*.parquet")`.
