import numpy as np
import polars as pl


def init_gs(df: pl.DataFrame) -> pl.DataFrame:
    """
    The first step a GS is defined as the period from PD to next PD. 
    This works whether the HD is defined by a crop calendar or not. 
    Needs PD column (planting date)
    """
    return (
        df.with_columns(
            season_start=(pl.col("day") == pl.col("PD")).cast(pl.Int32),
        )
        .with_columns(
            GS = pl.col("season_start").cum_sum().over("pixel").cast(pl.Int32),
        )
        .with_columns(
            GS = pl.when(pl.col("GS") > 0)
            .then(pl.col("GS"))
            .otherwise(0)
            .cast(pl.Int32)
        )
        .drop("season_start")
    )

def add_gdd(df: pl.DataFrame, tbsc: float) -> pl.DataFrame:
    return (
        df.with_columns(
            GDD = (pl.col("tav") - tbsc)
            .clip(lower_bound=0)
            .fill_nan(0)
            .fill_null(0)
        )
        .with_columns(
            GDDcum = pl.when(pl.col("GS") > 0)
            .then(pl.col("GDD").cum_sum().over(["pixel", "GS"]))
            .otherwise(0.0)
        )
    )

def add_hui(df: pl.DataFrame, gmhu: float) -> pl.DataFrame:
    return (
        df.with_columns(
            # Step 1: flag when threshold is reached
            hui_start = (pl.col("GDDcum") >= gmhu).cast(pl.Int32)
        )
        .with_columns(
            # Step 2: propagate activation forward within each (pixel, GS)
            hui_active = pl.col("hui_start")
            .cum_max()
            .over(["pixel", "GS"])
        )
        .with_columns(
            # Step 3: cumulative sum only when active
            HUI_raw = pl.when(pl.col("hui_active") == 1)
            .then(pl.col("GDD"))
            .otherwise(0.0)
            .cum_sum()
            .over(["pixel", "GS"])
        )
        .with_columns(
            # Step 4: normalize by PHU
            HUI = (pl.col("HUI_raw") / pl.col("PHU")).fill_null(0.0).cast(pl.Float32)
        )
        .drop(["hui_start", "hui_active", "HUI_raw"])
        
    )

def add_hd(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates HD column in case it is not provided from crop calendar.
    """
    return df.with_columns(
        HD = (
            pl.when(pl.col("HUI") >= 1)
            .then(pl.col("day"))
            .otherwise(None)
            .min()
            .over(["pixel", "GS"])
        )
    )

def add_lai_chd(df: pl.DataFrame, dlap1: float, dlap2: float, hmx: float, dlai: float, dmla: float, rlad: float = 1.0) -> pl.DataFrame:
    return (
        df.with_columns(
            HUF = pl.col("HUI") / (pl.col("HUI") + (dlap1 - dlap2 * pl.col("HUI")).exp())
        )
        .with_columns(
            CHT = (pl.col("HUF") + 1e-10).sqrt() * hmx,
            LAI = (
                pl.when(pl.col("HUI") >= dlai)
                .then(dmla * ((1 - pl.col("HUI")) / (1 - dlai))**rlad)
                .otherwise(pl.col("HUF") * dmla)
                .clip(lower_bound=0)
            )
        )
        .drop('HUF')
    )


def add_pet(df: pl.DataFrame, vpth: float, gsi: float, vpd2: float, salb: float = 0.15, prmt_1: float = 1.0, prmt_41: float = 0.0) -> pl.DataFrame:
    return (
        df.with_columns(
            lai_eff = pl.col("LAI") + 0.01,
            cht_eff = pl.col("CHT").clip(lower_bound=0.01),
            tk = pl.col("tav") + 273.15,  # .15?
        )
        .with_columns(
            # Solar geometry
            sd = 0.4102 * ((pl.col("day") - 80.25) / 58.13).sin(),
            dd = 1 + 0.0335 * ((pl.col("day") + 88.2) / 58.13).sin(),
        )
        .with_columns(
            ch = -(pl.col("LAT") / 57.296).tan() * pl.col("sd").tan()
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
                pl.col("h") * (pl.col("LAT") / 57.296).sin() * pl.col("sd").sin() +
                (pl.col("LAT") / 57.296).cos() * pl.col("sd").cos() * pl.col("h").sin()
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
            pb = 101.3 - pl.col("ELEV") * (0.01152 - 5.44e-7 * pl.col("ELEV")),
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
                (0.00155 * (330 - pl.col("CO2"))).exp()
            ),
        )
        .with_columns(
            # PET
            epp = pl.col("PRMT74") * (
                pl.col("x2") + 86.66 * pl.col("rho") * pl.col("vpd") / pl.col("rv")
            ) / (
                pl.col("xl") * (
                    pl.col("dlt") +
                    pl.col("gma") * (1 + pl.col("rc") / pl.col("rv"))
                )
            ),

            eo = pl.col("PRMT74") * (
                pl.col("x2") + 86.66 * pl.col("rho") * pl.col("vpd") / (350 / pl.col("sfcwind"))
            ) / (
                pl.col("xl") * (pl.col("dlt") + pl.col("gma"))
            ),
        )
        .with_columns(
            PET = pl.max_horizontal("epp", "eo")
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
        cond = (pl.col("HUI") >= lo) & (pl.col("HUI") < hi)
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
        GS = pl.when(
            (pl.col("GS") > 0) & (pl.col("day") <= pl.col("HD"))
        )
        .then(pl.col("GS"))
        .otherwise(0)
    )

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
                .over(["pixel", "GS"])
            ),

            # dry streak id
            dd_grp = (
                pl.col("dd")
                .cast(pl.Int32)
                .diff()
                .fill_null(0)
                .ne(0)
                .cum_sum()
                .over(["pixel", "GS"])
            )
        )
    )

def longest_streak(df, keys, flag_col, run_col, out):
    return (
        df.filter(pl.col(flag_col))
        .group_by(keys + [run_col])
        .agg(pl.len().alias("_n"))
        .group_by(keys)
        .agg(pl.col("_n").max().alias(out))
    )