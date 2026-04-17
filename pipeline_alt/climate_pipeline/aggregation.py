from dataclasses import dataclass
from enum import IntEnum
from typing import Any
import numpy as np

from . import unepic


class VarIdx(IntEnum):
    HURS = 0
    PRCP = 1
    RSDS = 2
    WIND = 3
    TMAX = 4
    TMIN = 5


CLIMATE_VARS = ["hurs", "pr", "rsds", "sfcwind", "tasmax", "tasmin"]


@dataclass(frozen=True)
class ReferenceCrop:
    crop: unepic.Crop

    @classmethod
    def maize(cls) -> "ReferenceCrop":
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
        dlap1, dlap2 = unepic.ssolve(
            int(crop.dlap1) * 0.01,
            crop.dlap1 - int(crop.dlap1),
            int(crop.dlap2) * 0.01,
            crop.dlap2 - int(crop.dlap2),
        )
        return cls(crop=unepic.Crop(**{**crop.__dict__, "dlap1": dlap1, "dlap2": dlap2}))


def longest_spell(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    if edges.size == 0:
        return 0
    return int(np.max(edges[1::2] - edges[::2]))


def _convert_units(data: np.ndarray) -> np.ndarray:
    converted = data.astype(np.float64, copy=True)
    converted[:, VarIdx.HURS] /= 100.0
    converted[:, VarIdx.PRCP] *= 24 * 60 * 60
    converted[:, VarIdx.RSDS] *= 24 * 60 * 60 / 1e6
    converted[:, VarIdx.TMAX] -= 273.15
    converted[:, VarIdx.TMIN] -= 273.15
    return converted


def process_year(
    data_orig: np.ndarray,
    year: int,
    plant_day: int,
    harvest_day: int,
    lat: float,
    elev: float,
    phu: float,
    prmt_74: float,
    co2: float,
    crop: ReferenceCrop,
    compute_growing_season: bool = True,
) -> dict[str, Any] | None:
    data = _convert_units(data_orig)
    crop_state = crop.crop

    season, hui, gdd = unepic.compute_season(
        data[:, VarIdx.TMIN],
        data[:, VarIdx.TMAX],
        plant_day,
        harvest_day,
        crop_state.tbsc,
        crop_state.gmhu,
        phu,
        dynamic=compute_growing_season,
    )
    if season is None:
        return None

    lai, cht = unepic.lai(
        hui=hui,
        dlap1=crop_state.dlap1,
        dlap2=crop_state.dlap2,
        dlai=crop_state.dlai,
        rlad=crop_state.rlad,
        dmla=crop_state.dmla,
        hmx=crop_state.hmx,
    )

    aggregates: dict[str, Any] = {}

    for period, day_slice in season.items():
        window = data[day_slice]
        if window.size == 0:
            continue

        season_len = int(window.shape[0])
        tav = (window[:, VarIdx.TMAX] + window[:, VarIdx.TMIN]) / 2.0
        doy = (np.arange(day_slice.start, day_slice.start + season_len) % 366) + 1

        pet = unepic.pet_pm(
            tav=tav,
            rad=window[:, VarIdx.RSDS],
            ws=window[:, VarIdx.WIND],
            hur=window[:, VarIdx.HURS],
            lai=lai[day_slice],
            cht=cht[day_slice],
            doy=doy,
            elev=elev,
            salb=0.15,
            lat=lat,
            co2=co2,
            vpth=crop_state.vpth,
            gsi=crop_state.gsi,
            vpd2=crop_state.vpd2,
            prmt_74=prmt_74,
        )

        cmd = pet - window[:, VarIdx.PRCP]
        wet_days = window[:, VarIdx.PRCP] > 1.0
        dry_days = ~wet_days
        wet_sum = int(np.nansum(wet_days))
        dry_sum = int(np.nansum(dry_days))

        scalar_aggs = {
            "TMXav": float(np.nanmean(window[:, VarIdx.TMAX])),
            "TMNav": float(np.nanmean(window[:, VarIdx.TMIN])),
            "TAVav": float(np.nanmean(tav)),
            "PRCPsum": float(np.nansum(window[:, VarIdx.PRCP])),
            "RADsum": float(np.nansum(window[:, VarIdx.RSDS])),
            "WSDav": float(np.nanmean(window[:, VarIdx.WIND])),
            "HURav": float(np.nanmean(window[:, VarIdx.HURS])),
            "PETsum": float(np.nansum(pet)),
            "GDDsum": float(np.nansum(gdd[day_slice])),
            "CMDsum": float(np.nansum(cmd)),
            "LEN": season_len,
            "HUIeop": float(hui[day_slice][-1]),
        }

        counts = {
            "HDD": int(np.nansum(window[:, VarIdx.TMAX] >= 30)),
            "KDD": int(np.nansum(window[:, VarIdx.TMAX] >= 39)),
            "FRT": int(np.nansum(window[:, VarIdx.TMIN] <= 0)),
            "ICE": int(np.nansum(window[:, VarIdx.TMAX] <= 0)),
            "R10": int(np.nansum(window[:, VarIdx.PRCP] >= 10.0)),
            "R20": int(np.nansum(window[:, VarIdx.PRCP] >= 20.0)),
            "WET": wet_sum,
            "DRY": dry_sum,
            "CWD": longest_spell(wet_days) if wet_sum else 0,
            "CDD": longest_spell(dry_days) if dry_sum else 0,
            "CMDlt0": int(np.nansum(cmd < 0)),
        }

        for key, value in scalar_aggs.items():
            aggregates[f"{key}{period}"] = value
        for key, value in counts.items():
            aggregates[f"{key}frac{period}"] = value / season_len

    aggregates["co2_year"] = year
    return aggregates
