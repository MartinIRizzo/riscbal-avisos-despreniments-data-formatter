import numpy as np
import pandas as pd


def _dms_to_decimal(dms_str: str) -> float:
    """Convert an AEMET DMS coordinate string to decimal degrees.

    AEMET encodes coordinates as DDMMSS (latitude) or DDDMMSS (longitude).
    For Balearic longitudes, a trailing '1' indicates East and must be stripped
    before parsing.
    """
    degrees = int(dms_str[:-4])
    minutes = int(dms_str[-4:-2])
    seconds = int(dms_str[-2:])
    return degrees + minutes / 60 + seconds / 3600


def _distribute_minus4(values: list) -> list:
    """Distribute accumulated rainfall (-4 codes) in a daily series.

    When a positive value follows a run of -4 entries, that positive value
    represents the total rainfall for the run plus the day itself. The total
    is divided equally among all those days.
    Values that are NaN are left as-is.
    """
    result = values.copy()
    run_start = None
    for i, v in enumerate(result):
        if np.isnan(v):
            continue
        if v == -4:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                n_days = i - run_start + 1
                share = round(v / n_days, 1)
                for j in range(run_start, i + 1):
                    result[j] = share
                run_start = None
    # If a run of -4 reaches the end without a closing positive value,
    # we cannot distribute — set them to NaN (unknown).
    if run_start is not None:
        for j in range(run_start, len(result)):
            if result[j] == -4:
                result[j] = np.nan
    return result


def format_aemet_data(df: pd.DataFrame) -> pd.DataFrame:
    """Format raw AEMET precipitation data into a daily DataFrame.

    Transformations applied:
    - Coordinates converted from AEMET DMS to decimal degrees.
      Longitude trailing '1' (East indicator for Balearics) is stripped.
    - Precipitation columns (P1..P31) are melted into one row per day.
    - Special code -3 (trace/inapreciable) is replaced with 0.
    - Special code -4 (accumulated) is distributed: the first positive value
      after a run of -4s is split evenly across all those days.
    - Values are converted from tenths of mm to mm.
    """
    df = df.copy()

    # --- Coordinates ---
    df["LONGITUD"] = df["LONGITUD"].astype(str).str.strip()
    df["LATITUD"] = df["LATITUD"].astype(str).str.strip()
    # Strip trailing '1' (East indicator) from Balearic longitudes
    df["LONGITUD"] = df["LONGITUD"].str[:-1]
    df["LONGITUD"] = df["LONGITUD"].apply(_dms_to_decimal)
    df["LATITUD"] = df["LATITUD"].apply(_dms_to_decimal)

    # --- Identify precipitation columns ---
    p_cols = [f"P{d}" for d in range(1, 32)]

    # Coerce precipitation columns to numeric (empty strings / blanks → NaN)
    for col in p_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Handle -3 (trace rain → 0) ---
    for col in p_cols:
        df[col] = df[col].replace(-3, 0)

    # --- Handle -4 (accumulated rainfall distribution) ---
    minus4_mask = (df[p_cols] == -4).any(axis=1)
    if minus4_mask.any():
        df.loc[minus4_mask, p_cols] = (
            df.loc[minus4_mask, p_cols]
            .apply(
                lambda row: pd.Series(
                    _distribute_minus4(row.tolist()), index=p_cols
                ),
                axis=1,
            )
        )

    # --- Convert tenths of mm to mm ---
    for col in p_cols:
        df[col] = df[col] / 10.0

    # --- Melt to one row per day ---
    id_cols = [c for c in df.columns if c not in p_cols]
    df_melted = df.melt(
        id_vars=id_cols,
        value_vars=p_cols,
        var_name="DIA_COL",
        value_name="PRECIPITACION",
    )
    # Extract day number from column name (P1 → 1, P31 → 31)
    df_melted["DIA"] = df_melted["DIA_COL"].str.replace("P", "").astype(int)
    df_melted.drop(columns=["DIA_COL"], inplace=True)

    # Build date column, dropping invalid dates (e.g. Feb 30)
    df_melted["FECHA"] = pd.to_datetime(
        {
            "year": df_melted["AÑO"],
            "month": df_melted["MES"],
            "day": df_melted["DIA"],
        },
        errors="coerce",
    )
    df_melted = df_melted.dropna(subset=["FECHA"])

    # Drop intermediate date-part columns now that FECHA is built
    df_melted.drop(columns=["AÑO", "MES", "DIA"], inplace=True)

    # Drop rows where precipitation is NaN (missing / empty in original)
    df_melted = df_melted.dropna(subset=["PRECIPITACION"])

    # Sort and reset index
    df_melted = df_melted.sort_values(
        ["INDICATIVO", "FECHA"]
    ).reset_index(drop=True)

    df_melted = df_melted.rename(columns={
        "INDICATIVO": "code",
        "FECHA": "date",
        "PRECIPITACION": "precipitation",
        "LONGITUD": "lon",
        "LATITUD": "lat",
    })

    return df_melted