from typing import Literal
import numpy as np
import polars as pl

from models import unepic


def init_gs(df: pl.DataFrame, mode: Literal["calendar", "pd_only"] = "calendar") -> pl.DataFrame:
    """
    Initializes the growing season (GS) column in the dataframe. 
    The first step a GS is defined as the period from PD to next PD. 
    This works whether the HD is defined by a crop calendar or not. 
    Needs PD column (planting date)

    :param: 
    """
    if mode == "calendar":
        return df.with_columns(
            season_start=(pl.col("day") == pl.col("pd")).cast(pl.Int32),
            in_gs=pl.when(pl.col("pd") <= pl.col("hd"))
            .then(
                pl.col("day").is_between(pl.col("pd"), pl.col("hd"), closed="both")
            )
            .otherwise(
                (pl.col("day") >= pl.col("pd")) | (pl.col("day") <= pl.col("hd"))
            ),
        ).with_columns(
            gs_id_raw=pl.col("season_start").cum_sum().over("pixel"),
        ).with_columns(
            gs=pl.when(pl.col("in_gs"))
            .then(pl.col("gs_id_raw"))
            .otherwise(0)
            .cast(pl.Int32)
        ).drop(["season_start", "in_gs", "gs_id_raw"])
    
    elif mode == "pd_only":
        return (
            df.with_columns(
                season_start=(pl.col("day") == pl.col("pd")).cast(pl.Int32),
            )
            .with_columns(
                gs = pl.col("season_start").cum_sum().over("pixel").cast(pl.Int32),
            )
            .with_columns(
                gs = pl.when(pl.col("gs") > 0)
                .then(pl.col("gs"))
                .otherwise(0)
                .cast(pl.Int32)
            )
            .drop("season_start")
        )
    
    else:
        raise ValueError("Invalid value for 'mode'.")

def add_gdd(df: pl.DataFrame, tbsc: float) -> pl.DataFrame:
    return (
        df.with_columns(
            gdd = (pl.col("tav") - tbsc)
            .clip(lower_bound=0)
            .fill_nan(0)
            .fill_null(0)
        )
        .with_columns(
            gdd_cum = pl.when(pl.col("gs") > 0)
            .then(pl.col("gdd").cum_sum().over(["pixel", "gs"]))
            .otherwise(0.0)
        )
    )

def add_hui(df: pl.DataFrame, gmhu: float) -> pl.DataFrame:
    return (
        df.with_columns(
            # Step 1: flag when threshold is reached
            hui_start = (pl.col("gdd_cum") >= gmhu).cast(pl.Int32)
        )
        .with_columns(
            # Step 2: propagate activation forward within each (pixel, GS)
            hui_active = pl.col("hui_start")
            .cum_max()
            .over(["pixel", "gs"])
        )
        .with_columns(
            # Step 3: cumulative sum only when active
            hui_raw = pl.when(pl.col("hui_active") == 1)
            .then(pl.col("gdd"))
            .otherwise(0.0)
            .cum_sum()
            .over(["pixel", "gs"])
        )
        .with_columns(
            # Step 4: normalize by PHU
            hui = (pl.col("hui_raw") / pl.col("phu")).fill_null(0.0).cast(pl.Float32)
        )
        .drop(["hui_start", "hui_active", "hui_raw"])
        
    )

def add_hd(df: pl.DataFrame, mode: Literal["semi_dynamic", "dynamic"] = "semi_dynamic") -> pl.DataFrame:
    """Calculates HD column in case it is not provided from crop calendar.

    :param df: Target Polars df
    :param mode: 
        "dynamic" to calculate HD based on HD >= 1; "semi_dynamic" is same as "dynamic", but caps the  
        calculated HD at calendar HD + 21 days. Defaults to "semi_dynamic".
    """
    if mode == "semi_dynamic":
        return df.with_columns(
            hd = pl.coalesce([
                pl.when(
                    (pl.col("hui") >= 1) &
                    (pl.col("day") <= pl.col("hd").first().over(["pixel", "gs"]) + 21)
                )
                .then(pl.col("day"))
                .otherwise(None)
                .min()
                .over(["pixel", "gs"]),
                pl.col("hd").first().over(["pixel", "gs"]) + 21
            ])
        )

    elif mode =="dynamic":
        return df.with_columns(
            hd = (
                pl.when(pl.col("hui") >= 1)
                .then(pl.col("day"))
                .otherwise(None)
                .min()
                .over(["pixel", "gs"])
            )
        )
    
    else:
        raise ValueError("Invalid value for 'mode'")
    

def add_lai_chd(df: pl.DataFrame, dlap1: float, dlap2: float, hmx: float, dlai: float, dmla: float, rlad: float = 1.0) -> pl.DataFrame:
    return (
        df.with_columns(
            huf = pl.col("hui") / (pl.col("hui") + (dlap1 - dlap2 * pl.col("hui")).exp())
        )
        .with_columns(
            cht = (pl.col("huf") + 1e-10).sqrt() * hmx,
            lai = (
                pl.when(pl.col("hui") >= dlai)
                .then(dmla * ((1 - pl.col("hui")) / (1 - dlai))**rlad)
                .otherwise(pl.col("huf") * dmla)
                .clip(lower_bound=0)
            )
        )
        .drop('huf')
    )


def add_pet(df: pl.DataFrame, vpth: float, gsi: float, vpd2: float, salb: float = 0.15, prmt_1: float = 1.0, prmt_41: float = 0.0) -> pl.DataFrame:
    return (
        df.with_columns(
            lai_eff = pl.col("lai") + 0.01,
            cht_eff = pl.col("cht").clip(lower_bound=0.01),
            tk = pl.col("tav") + 273.15,  # .15?
        )
        .with_columns(
            # Solar geometry
            sd = 0.4102 * ((pl.col("day") - 80.25) / 58.13).sin(),
            dd = 1 + 0.0335 * ((pl.col("day") + 88.2) / 58.13).sin(),
        )
        .with_columns(
            ch = -(pl.col("lat") / 57.296).tan() * pl.col("sd").tan()
        )
        .with_columns(
            h = (
                pl.when(pl.col("ch") > 1).then(0.0)
                .when(pl.col("ch") < -1).then(np.pi)
                .otherwise(pl.col("ch").arccos())
            )
        )
        .with_columns(
            # Solar radiation potential
            ramx = 30 * pl.col("dd") * (
                pl.col("h") * (pl.col("lat") / 57.296).sin() * pl.col("sd").sin() +
                (pl.col("lat") / 57.296).cos() * pl.col("sd").cos() * pl.col("h").sin()
            )
        )
        .with_columns(
            # Albedo
            eaj = (-pl.max_horizontal(
                0.4 * pl.col("lai_eff"),
                prmt_41 * 0.1
            )).exp(),
            ea = 0.1 * ((54.879 - 5.029 * pl.col("tk").log() - 6790.5 / pl.col("tk")).exp()),
        )
        .with_columns(
            ed = pl.col("ea") * pl.col("hurs"),
        )
        .with_columns(
            alb = 0.23 * (1 - pl.col("eaj")) + salb * pl.col("eaj"),
            # Vapor + radiation
            vpd = pl.col("ea") - pl.col("ed"),
            dlt = pl.col("ea") * (6790.5 / pl.col("tk") - 5.029) / pl.col("tk"),
            pb = 101.3 - pl.col("elev") * (0.01152 - 5.44e-7 * pl.col("elev")),
            rbo = (0.34 - 0.14 * pl.col("ed").sqrt()) * 4.9e-9 * pl.col("tk").pow(4),
            rto = (pl.col("rsds") / (pl.col("ramx") + 0.1)).clip(upper_bound=0.99),
        )
        .with_columns(
            gma = 6.595e-4 * pl.col("pb"),
            xl = 2.501 - 2.2e-3 * pl.col("tav"),  # C°
            rn = pl.col("rsds") * (1 - pl.col("alb")) -
                pl.col("rbo") * (0.9 * pl.col("rto") + 0.1),
        )
        .with_columns(
            x2 = pl.col("rn") * pl.col("dlt"),
            # Air density
            rho = 0.01276 * pl.col("pb") / (1 + 0.00367 * pl.col("tav")),  # C°
            zz = (pl.col("cht_eff") + 2).clip(lower_bound=10),
        )
        .with_columns(
            # Wind scaling
            uzz = pl.when(pl.col("zz") <= 10)
                .then(pl.col("sfcwind"))
                .otherwise(
                    pl.col("sfcwind") *
                    (pl.col("zz") / 0.0005).log() / 9.9035
                ),
            x1 = (pl.col("cht_eff") + 0.01).log10(),
            x3 = pl.col("vpd") - vpth,
        )
        .with_columns(
            # Surface roughness
            z0 = (10 ** (0.997 * pl.col("x1") - 0.883)),
            zd = (10 ** (0.979 * pl.col("x1") - 0.154)),
            # Canopy resistance
            fvpd = pl.when(pl.col("x3") <= 0)
                .then(1.0)
                .otherwise(
                    (1 - vpd2 * pl.col("x3")).clip(lower_bound=0.1)
                ),
        )
        .with_columns(
            rv = 6.25 * (
                ((pl.col("zz") - pl.col("zd")) / pl.col("z0")).log() ** 2
            ) / pl.col("uzz"),
            g1 = gsi * pl.col("fvpd"),
        )
        .with_columns(
            rc = prmt_1 / (
                (pl.col("lai_eff") + 0.01) *
                pl.col("g1") *
                (0.00155 * (330 - pl.col("co2"))).exp()
            ),
        )
        .with_columns(
            # PET
            epp = pl.col("prmt74") * (
                pl.col("x2") + 86.66 * pl.col("rho") * pl.col("vpd") / pl.col("rv")
            ) / (
                pl.col("xl") * (
                    pl.col("dlt") +
                    pl.col("gma") * (1 + pl.col("rc") / pl.col("rv"))
                )
            ),

            eo = pl.col("prmt74") * (
                pl.col("x2") + 86.66 * pl.col("rho") * pl.col("vpd") / (350 / pl.col("sfcwind"))
            ) / (
                pl.col("xl") * (pl.col("dlt") + pl.col("gma"))
            ),
        )
        .with_columns(
            pet = pl.max_horizontal("epp", "eo")
        )
        .drop([
            # cleanup
            "tk","lai_eff","cht_eff","sd","dd","ch","h","ramx",
            "eaj","alb","ea","ed","vpd","dlt","pb","gma","xl",
            "rbo","rto","rn","x2","rho","zz","uzz","x1","z0","zd",
            "rv","x3","fvpd","g1","rc","epp","eo"
        ])
    )


def add_subgs(df: pl.DataFrame, col_name: str, segments: dict[str, tuple[float, float]]) -> pl.DataFrame:
    expr = None

    for name, (lo, hi) in segments.items():
        cond = (pl.col("hui") >= lo) & (pl.col("hui") < hi)
        if expr is None:
            expr = pl.when(cond).then(pl.lit(name))
        else:
            expr = expr.when(cond).then(pl.lit(name))
    expr = expr.otherwise(None).cast(pl.Categorical)

    return df.with_columns(expr.alias(col_name))

def clip_gs(df: pl.DataFrame) -> pl.DataFrame:
    """
    Sets the GS column to zero after the harvest day.
    """
    return df.with_columns(
        in_gs = pl.when(pl.col("pd") <= pl.col("hd"))
            .then(
                pl.col("day").is_between(pl.col("pd"), pl.col("hd"), closed="both")
            )
            .otherwise(
                (pl.col("day") >= pl.col("pd")) | (pl.col("day") <= pl.col("hd"))
            )
        ).with_columns(
            gs = pl.when(pl.col("in_gs")).then(pl.col("gs")).otherwise(0)
        ).drop("in_gs")


def add_streak_ids(df: pl.DataFrame) -> pl.DataFrame:

    return (
        df.with_columns(
            wd = (pl.col("pr") > 1.0),
            dd = (pl.col("pr") <= 1.0),
        )
        .with_columns(
            # wet streak id
            wd_grp = (
                pl.col("wd")
                .cast(pl.Int32)
                .diff()
                .fill_null(0)
                .ne(0)
                .cum_sum()
                .over(["pixel", "gs"])
            ),

            # dry streak id
            dd_grp = (
                pl.col("dd")
                .cast(pl.Int32)
                .diff()
                .fill_null(0)
                .ne(0)
                .cum_sum()
                .over(["pixel", "gs"])
            )
        )
    )

def longest_streak(df, keys, flag_col, run_col, out):
    return (
        df.group_by(keys + [run_col])
        .agg(
            pl.when(pl.col(flag_col).any())
            .then(pl.len())
            .otherwise(0)
            .alias("_n")
        )
        .group_by(keys)
        .agg(
            pl.col("_n").max().alias(out)
        )
    )

# Main climate pipeline aggregator
def process_climate_features(df: pl.DataFrame, crop: unepic.Crop, hd_mode: Literal["strict", "semi_dynamic", "dynamic"]) -> pl.DataFrame:
    
    df = init_gs(df, "calendar" if hd_mode == "strict" else "pd_only")
    df = add_gdd(df, crop.tbsc)
    df = add_hui(df, crop.gmhu)

    if hd_mode != "strict":
        # Overwrite HD with calculated HD
        df = add_hd(df, mode=hd_mode)
        df = clip_gs(df)

    # Remove days outside GS, as they play no further part in aggregation.
    df = df.filter(pl.col("gs") > 0)

    df = add_lai_chd(df, crop.dlap1, crop.dlap2, crop.hmx, crop.dlai, crop.dmla, crop.rlad)
    df = add_pet(df, crop.vpth, crop.gsi, crop.vpd2)

    # Tier 1 subseasons: vegetative and reproductive
    # (We use 3 as the upper limit because in strict mode hui can be > 1)
    df = add_subgs(
        df, 
        "gs_sub1", 
        {
            'gs_v': (0, 0.5),
            'gs_r': (0.5, 3),
        }
    )
    # Tier 2 subseasons: emergence, development, flowering, grain filling
    df = add_subgs(
        df, 
        "gs_sub2", 
        {
            'gs_e': (0, 0.25),
            'gs_d': (0.25, 0.5),
            'gs_f': (0.5, 0.75),
            'gs_g': (0.75, 3), 
        }
    )

    df = df.with_columns(cmd = pl.col("pet") - pl.col("pr"))
    df = add_streak_ids(df)

    aggregates = [
        pl.col("tasmax").mean().alias("TMXav"),
        pl.col("tasmin").mean().alias("TMNav"),
        pl.col("tav").mean().alias("TAVav"),

        pl.col("pr").sum().alias("PRCPsum"),
        pl.col("rsds").sum().alias("RADsum"),

        pl.col("sfcwind").mean().alias("WSDav"),
        pl.col("hurs").mean().alias("HURav"),

        pl.col("pet").sum().alias("PETsum"),
        pl.col("gdd").sum().alias("GDDsum"),
        pl.col("cmd").sum().alias("CMDsum"),

        pl.len().alias("LEN"),
        pl.col("hui").last().alias("HUIeop"),
        pl.col("year").first().alias("YR")
    ]

    counts = [
        # temperature extremes
        (pl.col("tasmax") >= 30).sum().alias("HDD"),
        (pl.col("tasmax") >= 39).sum().alias("KDD"),
        (pl.col("tasmin") <= 0).sum().alias("FRT"),
        (pl.col("tasmax") <= 0).sum().alias("ICE"),

        # precipitation thresholds
        (pl.col("pr") >= 10.0).sum().alias("R10"),
        (pl.col("pr") >= 20.0).sum().alias("R20"),

        # wet / dry days
        (pl.col("pr") > 1.0).sum().alias("WET"),
        (pl.col("pr") <= 1.0).sum().alias("DRY"),

        # CMD < 0
        ((pl.col("cmd")) > 0).sum().alias("CMDgt0"),
    ]

    # Replace NaNs with null to make aggregation work across
    # leap year placeholders.
    df = df.fill_nan(None)

    final = []
    for lvl, agg in enumerate([["pixel", "gs"], ["pixel", "gs", "gs_sub1"], ["pixel", "gs", "gs_sub2"]]):

        cwd = longest_streak(df, agg, "wd", "wd_grp", "cwd")
        cdd = longest_streak(df, agg, "dd", "dd_grp", "cdd")
        df_agg = (
            df.group_by(agg).agg(
                *aggregates, 
                *counts,         
            )
            .join(cwd, on=agg, how="left")
            .join(cdd, on=agg, how="left")
        )
        if lvl == 0:
            df_agg = df_agg.with_columns(period = pl.lit('gs').cast(pl.Categorical))
        elif lvl == 1:
            df_agg = df_agg.with_columns(period = pl.col("gs_sub1")).drop(['gs_sub1'])
        elif lvl == 2:
            df_agg = df_agg.with_columns(period = pl.col("gs_sub2")).drop(['gs_sub2'])
        final.append(df_agg)

    df_agg = pl.concat(final, how="vertical").sort(["pixel", "gs", "period"])

    # Remove post-GS and otherwise uncaught periods
    df_agg = df_agg.filter(~pl.col("period").is_null())

    return df_agg

# Utility expressions for processing crop calendar
def cal_len_to_hd(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        hd=(
            (pl.col("pd") + pl.col("hd")) %
            pl.when(
                ((pl.col("yr") % 4 == 0) & (pl.col("yr") % 100 != 0))
                | (pl.col("yr") % 400 == 0)
            )
            .then(366)
            .otherwise(365)
        )
    )

def cal_fix_shift(df: pl.DataFrame, shift: int) -> pl.DataFrame:
    lookup = df.select(
        "lon",
        "lat",
        pl.col("yr").alias("lookup_year"),
        pl.col("pd").alias("pd_shift"),
        pl.col("hd").alias("hd_shift"),
    )
    return (
        df.with_columns(
            lookup_year=pl.col("yr") + shift
        )
        .join(
            lookup,
            on=["lon", "lat", "lookup_year"],
            how="left"
        )
        .with_columns(
            pd=pl.when(
                (shift != 0) &
                (pl.col("hd") < pl.col("pd")) &
                pl.col("pd_shift").is_not_null()
            )
            .then(pl.col("pd_shift"))
            .otherwise(pl.col("pd")),

            hd=pl.when(
                (shift != 0) &
                (pl.col("hd") < pl.col("pd")) &
                pl.col("hd_shift").is_not_null()
            )
            .then(pl.col("hd_shift"))
            .otherwise(pl.col("hd"))
        )
        .drop("lookup_year", "pd_shift", "hd_shift")
    )
