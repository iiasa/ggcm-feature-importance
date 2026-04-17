"""
Minimal EPIC-style biophysical helpers copied into the refactor so the new
pipeline is self-contained inside `new/`.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Crop:
    wa: float
    tbsc: float
    dlai: float
    rlad: float
    dmla: float
    dlap1: float
    dlap2: float
    top: float
    rdmx: float
    hi: float
    hmx: float
    vpth: float
    vpd2: float
    gsi: float
    gmhu: float


def lai(hui: np.ndarray, dlap1: float, dlap2: float, dlai: float, rlad: float, dmla: float, hmx: float):
    huf = hui / (hui + np.exp(dlap1 - dlap2 * hui))
    huf = np.nan_to_num(huf, nan=0.0, posinf=0.0, neginf=0.0)
    f3 = np.sqrt(huf + 1e-10)
    cht = f3 * hmx
    lai_value = huf * dmla
    decline = hui >= dlai
    lai_value[decline] = dmla * ((1 - hui[decline]) / (1 - dlai)) ** rlad
    lai_value[lai_value < 0] = 0
    return lai_value, cht


def pet_pm(
    tav: np.ndarray,
    rad: np.ndarray,
    ws: np.ndarray,
    hur: np.ndarray,
    lai: np.ndarray,
    cht: np.ndarray,
    doy: np.ndarray,
    elev: float,
    salb: float,
    lat: float,
    co2: float,
    vpth: float,
    gsi: float,
    vpd2: float,
    prmt_1: float = 1.0,
    prmt_41: float = 0.0,
    prmt_74: float = 0.7,
) -> np.ndarray:
    pit = 58.13
    clt = 57.296
    ylat = lat / clt
    ylts = np.sin(ylat)
    yltc = np.cos(ylat)
    ytn = np.tan(ylat)
    dd = 1 + 0.0335 * np.sin((doy + 88.2) / pit)
    sd = 0.4102 * np.sin((doy - 80.25) / pit)
    ch = -ytn * np.tan(sd)

    h = np.zeros_like(ch)
    h[ch > 1] = 0.0
    h[ch < -1] = np.pi
    between = (ch >= -1) & (ch <= 1)
    h[between] = np.arccos(ch[between])

    lai_safe = lai.copy() + 0.01
    cht_safe = np.maximum(cht.copy(), 0.01)
    ramx = 30 * dd * (h * ylts * np.sin(sd) + yltc * np.cos(sd) * np.sin(h))

    eaj = np.exp(-np.maximum(0.4 * lai_safe, prmt_41 * 0.1))
    alb = 0.23 * (1 - eaj) + salb * eaj

    tk = tav + 273
    xl = 2.501 - 2.2e-3 * tav
    ea = 0.1 * np.exp(54.879 - 5.029 * np.log(tk) - 6790.5 / tk)
    ed = ea * hur
    vpd = ea - ed
    ralb1 = rad * (1 - alb)
    dlt = ea * (6790.5 / tk - 5.029) / tk
    pb = 101.3 - elev * (0.01152 - 5.44e-7 * elev)
    gma = 6.595e-4 * pb

    rbo = (0.34 - 0.14 * np.sqrt(ed)) * 4.9e-9 * tk**4
    rto = np.minimum(rad / (ramx + 0.1), 0.99)
    rn = ralb1 - rbo * (0.9 * rto + 0.1)
    x2 = rn * dlt

    rho = 0.01276 * pb / (1 + 0.00367 * tav)
    zz = np.maximum(cht_safe + 2.0, 10.0)
    uzz = np.empty_like(cht_safe)
    low = zz <= 10.0
    uzz[low] = ws[low]
    uzz[~low] = ws[~low] * np.log(zz[~low] / 0.0005) / 9.9035

    x1 = np.log10(cht_safe + 0.01)
    z0 = 10 ** (0.997 * x1 - 0.883)
    zd = 10 ** (0.979 * x1 - 0.154)
    rv = 6.25 * np.log((zz - zd) / z0) ** 2 / uzz
    x3 = vpd - vpth

    fvpd = np.maximum(1 - vpd2 * x3, 0.1)
    fvpd[x3 <= 0] = 1
    g1 = gsi * fvpd

    rc = prmt_1 / ((lai_safe + 0.01) * g1 * np.exp(0.00155 * (330 - co2)))
    epp = prmt_74 * (x2 + 86.66 * rho * vpd / rv) / (xl * (dlt + gma * (1 + rc / rv)))

    rv2 = 350 / ws
    eo = prmt_74 * (x2 + 86.66 * rho * vpd / rv2) / (xl * (dlt + gma))
    return np.maximum(epp, eo)


def ssolve(x1: float, y1: float, x2: float, y2: float):
    xx = np.log(x1 / y1 - x1)
    b2 = (xx - np.log(x2 / y2 - x2)) / (x2 - x1)
    b1 = xx + x1 * b2
    return b1, b2


_GS_SEGMENTS = {
    "GSv": (0, 0.5),
    "GSr": (0.5, 1),
    "GSe": (0, 0.25),
    "GSd": (0.25, 0.5),
    "GSf": (0.5, 0.75),
    "GSg": (0.75, 1),
}


def compute_season(
    tmin: np.ndarray,
    tmax: np.ndarray,
    pd_idx: int,
    hd_idx: int,
    tbsc: float,
    gmhu: float,
    phu: float,
    dynamic: bool = True,
):
    gdd = np.maximum((tmax[pd_idx:] + tmin[pd_idx:]) / 2 - tbsc, 0.0)

    germ_ridx = np.where(np.nancumsum(gdd) >= gmhu)[0]
    if len(germ_ridx) == 0:
        return None, None, None
    germ_ridx = germ_ridx[0] + 1

    hui = np.hstack([np.zeros(germ_ridx), np.cumsum(gdd[germ_ridx:]) / phu])

    if dynamic:
        hd_ridx = np.where(hui >= 1.0)[0]
        if len(hd_ridx) == 0:
            hd_ridx = hd_idx - pd_idx + 21
        else:
            hd_ridx = hd_ridx[0]
        hd_idx = pd_idx + hd_ridx
    else:
        hd_ridx = hd_idx - pd_idx

    periods = {
        "GS": slice(pd_idx, hd_idx),
        "GSp": slice(pd_idx - 30, pd_idx - 1),
    }

    for name, bounds in _GS_SEGMENTS.items():
        lower = np.where(hui >= bounds[0])[0]
        upper = np.where(hui >= bounds[1])[0]
        if len(lower) == 0 or len(upper) == 0 or lower[0] > hd_ridx or upper[0] > hd_ridx:
            continue
        periods[name] = slice(pd_idx + lower[0], pd_idx + upper[0] - 1)

    return periods, np.hstack([np.zeros(pd_idx), hui[:hd_ridx]]), np.hstack([np.zeros(pd_idx), gdd[:hd_ridx]])
