"""Reshape the AEMET daily precipitation file into one row per gauge and day.

The raw file (``data/historical_aemet_data.csv``) has one row per gauge and
month, with the daily values in columns P1..P31, in tenths of mm. Two special
codes appear:

- ``-3``: trace (rain below 0.1 mm), read as 0 mm;
- ``-4``: accumulated. A run of ``-4`` days is closed by the first positive
  value after it, which holds the total of the whole run (the ``-4`` days plus
  the closing day). Some manual gauges are read once a month: ``-4`` on every
  day but the last.

The ``-4`` totals are spread evenly over the days of their run, which keeps
the totals right but not their timing. Those days are flagged in the
``accumulated`` column, so later steps can tell a real daily reading from a
spread one (see the vault note "Accumulated -4 readings").
"""

import numpy as np
import pandas as pd

TRACE_CODE = -3
ACCUMULATED_CODE = -4
TENTHS_PER_MM = 10.0
MINUTES_PER_DEGREE = 60
SECONDS_PER_DEGREE = 3600
# Decimals of the spread share, in tenths of mm (as in the original code).
SHARE_DECIMALS = 1
DAY_COLUMNS = [f"P{day}" for day in range(1, 32)]


def _dms_to_decimal(dms_text: str) -> float:
    """Convert an AEMET DMS coordinate to decimal degrees.

    Parameters
    ----------
    dms_text : str
        DDMMSS (latitude) or DDDMMSS (longitude). The trailing East
        indicator of Balearic longitudes must be stripped before.

    Returns
    -------
    float
        Decimal degrees.
    """
    degrees = int(dms_text[:-4])
    minutes = int(dms_text[-4:-2])
    seconds = int(dms_text[-2:])
    return (
        degrees + minutes / MINUTES_PER_DEGREE + seconds / SECONDS_PER_DEGREE
    )


def _spread_accumulated(
    values: list[float],
) -> tuple[list[float], list[bool]]:
    """Spread the ``-4`` totals of one month over the days of their run.

    A run is a sequence of ``-4`` days closed by the first positive value
    after it. That value is the total of the run plus the closing day, so it
    is divided equally among all those days. A run that reaches the end of
    the month without a closing value can't be spread: its days become NaN
    (unknown). NaN values are left as they are.

    Parameters
    ----------
    values : list of float
        The month's daily values (P1..P31), in tenths of mm, with ``-4``
        codes and NaN for empty days.

    Returns
    -------
    spread : list of float
        The values with every closed run replaced by its share.
    accumulated : list of bool
        For each day, whether it belongs to a run (the ``-4`` days and the
        closing day).
    """
    spread = values.copy()
    accumulated = [False] * len(values)
    run_start = None
    for position, value in enumerate(spread):
        if np.isnan(value):
            continue
        if value == ACCUMULATED_CODE:
            if run_start is None:
                run_start = position
        elif run_start is not None:
            n_days = position - run_start + 1
            share = round(value / n_days, SHARE_DECIMALS)
            for day in range(run_start, position + 1):
                spread[day] = share
                accumulated[day] = True
            run_start = None
    if run_start is not None:
        for day in range(run_start, len(spread)):
            if spread[day] == ACCUMULATED_CODE:
                spread[day] = np.nan
                accumulated[day] = True
    return spread, accumulated


def format_aemet_data(raw_aemet: pd.DataFrame) -> pd.DataFrame:
    """Reshape the raw AEMET file into one row per gauge and day.

    Parameters
    ----------
    raw_aemet : pandas.DataFrame
        The raw AEMET file: one row per gauge (``INDICATIVO``) and month
        (``AÑO``, ``MES``), daily values in tenths of mm in ``P1``..``P31``,
        DMS coordinates in ``LONGITUD`` and ``LATITUD``.

    Returns
    -------
    pandas.DataFrame
        One row per gauge and day with a value, sorted by gauge and date.
        Columns: ``code``, ``date``, ``precipitation`` (mm), ``accumulated``
        (the day belongs to a ``-4`` run, so its value is a spread share of
        a multi-day total), ``lon`` and ``lat`` (decimal degrees), plus the
        other columns of the raw file (``NOMBRE``, ``ALTITUD``). Days without
        a value, and dates that don't exist (e.g. 30 February), have no row.
    """
    monthly = raw_aemet.copy()

    # Coordinates: strip the trailing '1' (East) of Balearic longitudes.
    monthly["LONGITUD"] = monthly["LONGITUD"].astype(str).str.strip()
    monthly["LATITUD"] = monthly["LATITUD"].astype(str).str.strip()
    monthly["LONGITUD"] = monthly["LONGITUD"].str[:-1]
    monthly["LONGITUD"] = monthly["LONGITUD"].apply(_dms_to_decimal)
    monthly["LATITUD"] = monthly["LATITUD"].apply(_dms_to_decimal)

    # Empty strings and blanks become NaN.
    for column in DAY_COLUMNS:
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce")
        monthly[column] = monthly[column].replace(TRACE_CODE, 0)

    flags = pd.DataFrame(False, index=monthly.index, columns=DAY_COLUMNS)
    has_accumulated = (monthly[DAY_COLUMNS] == ACCUMULATED_CODE).any(axis=1)
    for index in monthly.index[has_accumulated]:
        spread, accumulated = _spread_accumulated(
            monthly.loc[index, DAY_COLUMNS].tolist()
        )
        monthly.loc[index, DAY_COLUMNS] = spread
        flags.loc[index, DAY_COLUMNS] = accumulated

    for column in DAY_COLUMNS:
        monthly[column] = monthly[column] / TENTHS_PER_MM

    # One row per day. The flags are melted in the same order, so their
    # rows line up with the values before anything is filtered.
    id_columns = [c for c in monthly.columns if c not in DAY_COLUMNS]
    daily = monthly.melt(
        id_vars=id_columns,
        value_vars=DAY_COLUMNS,
        var_name="DIA_COL",
        value_name="PRECIPITACION",
    )
    daily["accumulated"] = flags.melt(value_vars=DAY_COLUMNS)[
        "value"
    ].to_numpy()
    daily["DIA"] = daily["DIA_COL"].str.replace("P", "").astype(int)
    daily = daily.drop(columns=["DIA_COL"])

    # Dates that don't exist (e.g. 30 February) are dropped.
    daily["FECHA"] = pd.to_datetime(
        {"year": daily["AÑO"], "month": daily["MES"], "day": daily["DIA"]},
        errors="coerce",
    )
    daily = daily.dropna(subset=["FECHA"])
    daily = daily.drop(columns=["AÑO", "MES", "DIA"])
    daily = daily.dropna(subset=["PRECIPITACION"])

    daily = daily.sort_values(["INDICATIVO", "FECHA"]).reset_index(drop=True)
    return daily.rename(
        columns={
            "INDICATIVO": "code",
            "FECHA": "date",
            "PRECIPITACION": "precipitation",
            "LONGITUD": "lon",
            "LATITUD": "lat",
        }
    )
