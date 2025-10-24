"""
Collection of biophysical functions, aligned to the EPIC crop model v.1102.
"""

from typing import Tuple
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from dataclasses import dataclass

@dataclass
class Crop:
    wa: float  # Biomass-Energy ratio
    tbsc: float  # Minimum temperature for plant growth (°C)
    dlai: float  # Fraction of growing season when leaf area declines
    rlad: float  # Leaf area index decline paramter
    dmla: float  # Maximum potential leaf area index
    dlap1: float
    dlap2: float
    top: float  # Optimal growing temperature (°C)
    rdmx: float  # Maximum root depth [m]
    hi: float  # (Max) harvest index
    hmx: float  # Max height

    vpth: float
    vpd2: float
    gsi: float  # Maximum Stomatal Conductance [m/s]

    gmhu: float

def hrlt(doy: np.ndarray[int], lat: np.ndarray[float]) -> np.ndarray[float]:
    """
    Returns day length for specified days per year and latitude
    """
    sd = 0.4102 * np.sin(((2 * np.pi) / 365) * (doy - 80.25))
    return 7.64 * np.arccos(-np.tan(((2 * np.pi) / 365) * lat) * np.tan(sd))

def hui(tmn: np.ndarray[float], tmx: np.ndarray[float], tb: float, phu: float, hd: int, buffer: int = 21) -> np.ndarray[float]:
    """
    Return heat unit index (step fn of aggregated heat units).
    This function operates on a data slice that is already cut at the planting date. This means that index 0 
    in tmn and tmx is the planting data. HD also refers to the HD from the view of this slice (so HD = HD - PD should be passed).

    :returns: HUI dataframe with size = season length
    """
    huk = np.maximum((tmx + tmn) / 2 - tb, 0.)
    hui = np.cumsum(huk) / phu
    doh = np.where(hui >= 1.)[0]
    doh = min(doh[0] if len(doh) > 0 else np.inf, hd + buffer)
    return hui[:doh + 1]  # +1

def lai(hui: np.ndarray, dlap1: float, dlap2: float, dlai: float, rlad: float, dmla: float, hmx: float):
    """
    LAI approximation
    :param hmx: Max plant height
    :returns: LAI and crop height CHT
    """
    huf = hui / (hui + np.exp(dlap1 - dlap2 * hui))  # f2, in epic: heat unit factor
    f3 = np.sqrt(huf + 1e-10)
    cht = f3 * hmx  # Crop height
    lai = huf * dmla
    lai[hui >= dlai] = dmla * ((1 - hui[hui >= dlai]) / (1 - dlai))**rlad
    lai[lai < 0] = 0
    return lai, cht

def lai_epic(hui: np.ndarray, dlap1: float, dlap2: float, dlai: float, rlad: float, dmla: float, doh: int):
    """
    Stress-free, loop-based EPIC version of LAI
    """
    huf = hui / (hui + np.exp(dlap1 - dlap2 * hui))  # f2, in epic: heat unit factor
    lai = np.zeros_like(hui)
    # reg = 1.0  # no stress
    dold = np.where(hui >= dlai)[0][0]  #  DOY where LAI starts declining
    for i in range(1, dold):
        dhuf = huf[i] - huf[i - 1]
        lai[i] = lai[i - 1] + dhuf * dmla * (1 - np.exp(5. * (lai[i - 1] - dmla)))  # * np.sqrt(reg)
    lai[dold:doh] = np.maximum(lai[dold - 1] * ((1 - hui[dold:doh]) / (1 - hui[dold - 1])), 0)**rlad
    
    return lai, dold

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def surface_temp_bare(tmn: np.ndarray, tmx: np.ndarray, prcp: np.ndarray):
    """
    Returns soil surface temperature for base soil. Still needs to be properly verfied as there 
    are many inconsistencies in the EPIC documentation.
    """
    wet = prcp > 0
    scaling = 0.1  #  wet day scaling factor
    tgbw = tmn + scaling * (tmx - tmn)
    r = wet.sum() / wet.shape[0]
    tgbd = ((tmx + tmn) / 2 - r * tgbw) / (1 - r)
    tgb = tgbw * wet + tgbd * (1 - wet)
    # tgb_star = moving_average(tgb, 5)
    tgb_star = sliding_window_view(np.pad(tgb, (5, 0), mode='constant'), 5)[:-1].mean(axis=1)
    return tgb_star

def stress_temp(tmn: np.ndarray, tmx: np.ndarray, prcp: np.ndarray, tbsc: float, top: float) -> np.ndarray:
    """
    Returns temperature stress per day for given climate vectors. Assumes bare soil.
    """
    tg = surface_temp_bare(tmn, tmx, prcp)
    ts = np.sin((np.pi / 2) * (tg - tbsc) / (top - tbsc))
    return np.maximum(ts, 0.)

def pet(tav: np.ndarray, rh: np.ndarray, rad: np.ndarray, ws: np.ndarray, doy: np.ndarray[int], elev: float, salb: float, lat: float, ):
    """
    Penman (1984) EPIC Potential Evapotranspiration

    :param tav: Average daily temperature [°C]
    :param rh: Relative humidity [% as fraction]
    :param rad: Solar radiation [MJ/m²] or, presumable rather [MJ/m²/day] (not explicitly documented)
    :param ws: Average daily windspeed at 10m height [m/s]
    :param doy: Day indices of the year for which to calculate PET
    :param elev: Elevation
    :param salb: Soil albedo
    :param lat: Latitude
    :returns: Potential evapotranspiration per day [mm]
    """
    ea = 0.1 * np.exp(54.88 - 5.03 * np.log(tav + 273) - (6791 / (tav + 273)))  # saturation vapor pressure at mean air temperature (kPa)
    ed = ea * rh  # vapor pressure at mean air temperature (kPa)
    delta = (ea / (tav + 273)) * (6791 / (tav + 273) - 5.03)  # slope of saturation vapor pressure curve
    pb = 101.3 - elev * (0.01152 - 5.44e-7 * elev)  # barometric pressure
    gamma = 6.595e-4 * pb  # psychrometer constant
    hv = 2.50 - 0.0022 * tav  # latent heat of evaporization (MJ/kg)

    tav_prev3 = sliding_window_view(np.pad(tav, (3, 0), mode='constant'), 3)[:-1].sum(axis=1) / 3
    g = 0.12 * (tav - tav_prev3)  # soil heat flux (MJ/m²)
    g[:3] = 0.

    alb = 0.23 * (1.0 - ea) + salb * ea  # albedo for plants from soil albedo
    rab = 4.9e-9 * (0.34 - 0.14 * np.sqrt(ed)) * (tav + 273)**4  # net outgoing long wave radiation (MJ/m2) 

    pit = 58.13
    clt = 57.296
    ylat = lat / clt
    ylts = np.sin(ylat)
    yltc = np.cos(ylat)
    ytn = np.tan(ylat)
    dd = 1 + 0.335 * np.sin((doy + 88.2) / pit)
    sd = 0.4102 * np.sin((doy - 80.25) / pit)
    ch = -ytn * np.tan(sd)

    h = np.zeros_like(ch)
    h[ch < -1] = np.pi
    idx = (ch >= -1) & (ch <= 1)
    h[idx] = np.arccos(ch[idx])

    ramx = 30 * dd * (h * ylts * np.sin(sd) + yltc * np.cos(sd) * np.sin(h))    
    h0 = rad * (1.0 - alb) - rab * (((0.9 * rad) / ramx) * 0.1)  # net radiation (MJ/m2)

    # Windspeed function
    fv = 7.2 + 1.63 * ws

    pet = (delta / (delta + gamma)) * ((h0 - g) / hv) + (gamma / (delta + gamma)) * fv * (ea - ed)
    return pet

def pet_pm(
    tav: np.ndarray, rad: np.ndarray, ws: np.ndarray, hur: np.ndarray, lai: np.ndarray, cht: np.ndarray, 
    doy: np.ndarray[int], elev: float, salb: float, lat: float, co2: float, 
    vpth: float, gsi: float, vpd2: float, 
    prmt_1: float = 1., prmt_41: float = 0., prmt_74: float = 0.7):
    """
    Penman-Monteith PET calculation

    :param tav: Average temperature, usually calculated as (tmn + tmx / 2) [°C]
    :param rad: Shortwave radiation [MJ/m²/day]
    :param ws: Near-Surface Wind Speed
    :param hur: Relative humidity [% / 100]
    :param lai: Leaf area index
    :param hai: Crop height [m]
    :param prmt_1: Crop canopy-PET (1-2) factor used to adjust crop canopy resistance in the Penman-Monteith PET equation.
    :param prmt_41: Soil evaporation-cover coefficient
    :param prmt_74: Penman-Monteith adjustment factor
    :param vpth: Crop's threshold VPD (0.5 for maize)
    :param gsi: Involved in calculating leaf conductance
    :param vpd2: Crop coefficient involved in calculating leaf conductance
    """
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
    h[ch > 1] = 0.
    h[ch < -1] = np.pi
    idx = (ch >= -1) & (ch <= 1)
    h[idx] = np.arccos(ch[idx])

    # EPIC fix?
    sum = lai.copy() + 0.01
    cht = np.maximum(cht.copy(), 0.01)

    # RAMX = maximum possible solar radiation
    ramx = 30 * dd * (h * ylts * np.sin(sd) + yltc * np.cos(sd) * np.sin(h))

    # Albedo
    eaj = np.exp(-np.maximum(0.4 * sum, prmt_41 * 0.1))  # Assume above-ground biomass = CV = 0
    alb = 0.23 * (1 - eaj) + salb * eaj

    tk = tav + 273
    xl = 2.501 - 2.2e-3 * tav  # XL = HV = latent heat of vaporization in MJ kg-1 (see eq manual, Eq 51)
    ea = 0.1 * np.exp(54.879 - 5.029 * np.log(tk) - 6790.5 / tk)
    ed = ea * hur  # ED = actual vapor pressure
    vpd = ea - ed  # VPD = vapor pressure deficit
    ralb1 = rad * (1 - alb)  # RALB1 = net short-wave radiation for max plant ET
    dlt = ea * (6790.5 / tk - 5.029) / tk  # The slope of the saturation vapor pressure curve (see eq manual)
    pb = 101.3 - elev * (0.01152 - 5.44e-7 * elev)  # PB = barometric pressure (see eq manual, copied form MAIN.f90), ELEV = elevation [m]
    gma = 6.595e-4 * pb  # GMA = psychrometer constant

    rbo = (0.34 - 0.14 * np.sqrt(ed)) * 4.9e-9 * tk**4  # RBO = net emissivity  equation 2.2.20 in SWAT manual
    rto = np.minimum(rad / (ramx + 0.1), 0.99)  # cloud cover factor equation 2.2.19 in SWAT manual
    rn = ralb1 - rbo * (0.9 * rto + 0.1)  # RN = net long-wave radiation
    x2 = rn * dlt  # X2 = intermediary of net radiation and slope of VP saturation used in final EPP equation

    rho = 0.01276 * pb / (1 + 0.00367 * tav)  # RHO = air density (AD) (see eq manual Eq. 66); different eq in SWAT function ?!?
    zz = np.maximum(cht + 2., 10.)
    uzz = np.empty_like(cht)
    uzz[zz <= 10.] = ws[zz <= 10.]
    uzz[zz > 10.] = ws[zz > 10.] * np.log(zz[zz > 10.] / 0.0005) / 9.9035

    x1 = np.log10(cht + 0.01)  # X1 = intermediate of CHMX for ZD, ZO
    z0 = 10**(0.997 * x1 - 0.883)  # ZO = surface roughness parameter in m
    zd = 10**(0.979 * x1 - 0.154)  # ZD = displacement height of the crop in m
    rv = 6.25 * np.log((zz - zd) / z0)**2 / uzz  # RV = aerodynamic resistance (AR) (see eq manual Eq. 67)
    x3 = vpd - vpth

    fvpd = np.maximum(1 - vpd2 * x3, 0.1)  # FVPD = VPD correction factor
    fvpd[x3 <= 0] = 1

    g1 = gsi * fvpd  # G1 = Leaf conductance
    rc = prmt_1 / ((sum + 0.01) * g1 * np.exp(0.00155 * (330 - co2)))  # canopy resistance (CR) (see eq manual Eq. 71, which is different)
    epp = prmt_74 * (x2 + 86.66 * rho * vpd / rv) / (xl * (dlt + gma * (1 + rc / rv)))  # potential plant evaporation; PRMT(74) = Penman-Monteith adjustment factor

    rv2 = 350 / ws  # RV = aerodynamic resistance if no crop is grown
    eo = prmt_74 * (x2 + 86.66 * rho * vpd / rv2) / (xl * (dlt + gma))  # EO = potential ET

    pet = np.maximum(epp, eo)
    return pet

def water_use(pet: np.ndarray, lai: np.ndarray, z:float, rz: float, lambda_: float = 5.0):
    """
    The potential, total water use from the soil surface to any root depth z. [2.208]

    :param lambda_: Exponential coefficient in potential water use root growth distribution equation; PRMT(54)
    """
    ep = pet * (lai / 3)  # potential water use as fraction of PET
    ep = np.minimum(ep, pet)  # EP never larger as PET
    up = (ep / (1 - np.exp(-lambda_))) * (1 - np.exp(-lambda_ * (z / rz)))
    return up

def ssolve(x1: float, y1: float, x2: float, y2: float):
    """
    Solves y = x / (x + np.exp(b1 - b2 * x)) for b1 and b2, given that the function 
    passes to the two points provided. y is normalized (0-1) while x is not.
    Many different behaviors are described using this two-point S-function.
    """
    xx = np.log(x1 / y1 - x1)
    b2 = (xx - np.log(x2 / y2 - x2)) / (x2 - x1)
    b1 = xx + x1 * b2
    return b1, b2

def sfun(x, b1, b2):
    """
    Returns normalized (0-1) value on the S-curve.
    """
    return x / (x + np.exp(b1 - b2 * x))
