"""Rain windows of each sample: the event day and the 7, 30 and 90 days before.

AEMET gauges are read from the reshaped daily table (aemet_formatter.py);
the RiscBal gauges (``gauges.riscbal_stations`` in config.yaml) from the
RiscBal database.

Missing days count as 0 mm: AEMET days absent from the file are simply not
summed, and the database returns no rows. ``rain_days_90d`` (controls.py)
shows how many days were really measured.
"""

import os
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from data_formatter.controls import rain_days_90d

RAIN_DECIMALS = 1
# Window lengths in days, before the event day (which is excluded).
WEEK_DAYS = 7
MONTH_DAYS = 30
THREE_MONTHS_DAYS = 90
RAIN_COLUMNS = [
    "event_day_acc",
    "last_week_acc",
    "last_month_acc",
    "last_three_months_acc",
]

# Per AEMET gauge: its sorted dates and the rain of each date (mm).
RainIndex = dict[str, tuple[np.ndarray, np.ndarray]]


def index_rain_by_gauge(
    aemet_by_station: dict[str, pd.DataFrame],
) -> RainIndex:
    """Dates and rain of every AEMET gauge, ready for window lookups.

    Parameters
    ----------
    aemet_by_station : dict of str to pandas.DataFrame
        The reshaped AEMET table split by gauge code. Each frame is sorted by
        date, as ``format_aemet_data`` returns it.

    Returns
    -------
    dict of str to tuple of numpy.ndarray
        For each gauge code, its dates (``datetime64[ns]``) and the rain of
        each date in mm.
    """
    return {
        code: (
            station["date"].to_numpy(dtype="datetime64[ns]"),
            station["precipitation"].to_numpy(),
        )
        for code, station in aemet_by_station.items()
    }


def _get_riscbal_precipitation_acc(
    station_code: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    day_offset: pd.Timedelta,
) -> float:
    """Rain at a RiscBal gauge from the database, in a window of days.

    Sums the ``Rain10m`` records with quality ``0`` of the gauge (credentials
    from ``.env``). Missing records count as 0, like the AEMET branch.

    Parameters
    ----------
    station_code : str
        RiscBal gauge code (BAR04, GRO08).
    start_date, end_date : pandas.Timestamp
        Window limits at midnight; ``end_date`` is not included.
    day_offset : pandas.Timedelta
        Start of the pluviometric day: both limits are shifted by it, so the
        window measures the same days as AEMET (07:00 UTC to 07:00 UTC).

    Returns
    -------
    float
        Rain in mm (not rounded); 0.0 if there are no records.
    """
    # Same pluviometric days as AEMET: day D runs 07:00 UTC D -> 07:00 D+1,
    # so both limits are shifted by `day_offset`.
    start = (
        (start_date + day_offset).to_pydatetime().replace(tzinfo=timezone.utc)
    )
    end = (end_date + day_offset).to_pydatetime().replace(tzinfo=timezone.utc)

    with (
        psycopg.connect(
            f"dbname={os.getenv('DATABASE_NAME')} "
            f"user={os.getenv('DATABASE_USER')} "
            f"password={os.getenv('DATABASE_PASSWORD')} "
            f"host={os.getenv('DATABASE_HOST')} "
            f"port={os.getenv('DATABASE_PORT')}"
        ) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            """
                SELECT SUM(d.value)
                FROM data_historical_data d
                INNER JOIN instrumentation_sensors ins
                    ON ins.id = d.sensor_id
                INNER JOIN stations_stations s
                    ON ins.code_id = s.code
                    AND s.code = %s
                INNER JOIN variables_variables v
                    ON v.id = ins.variable_id
                    AND v.variable = 'Rain10m'
                WHERE d.logger_timestamp >= %s
                    AND d.logger_timestamp < %s
                    AND d.quality = '0'
            """,
            (station_code, start, end),
        )

        result = cur.fetchone()
        # SUM over no rows is NULL (None). Treat it as 0, like the AEMET
        # branch, instead of crashing later in round(None).
        if result is None or result[0] is None:
            return 0.0
        return float(result[0])


def window_rain(
    station_code: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rain_index: RainIndex,
    riscbal_stations: tuple[str, ...],
    day_offset: pd.Timedelta,
) -> float:
    """Rain at one gauge from start_date (inclusive) to end_date (exclusive).

    Parameters
    ----------
    station_code : str
        Gauge code: an AEMET gauge, or a RiscBal one (read from the
        database).
    start_date, end_date : pandas.Timestamp
        Window limits, at midnight; ``end_date`` is not included.
    rain_index : dict
        From ``index_rain_by_gauge``.
    riscbal_stations : tuple of str
        Gauges read from the RiscBal database instead of ``rain_index``.
    day_offset : pandas.Timedelta
        Start of the pluviometric day (07:00 UTC), for the RiscBal gauges.

    Returns
    -------
    float
        Rain in mm, rounded to 0.1 mm. 0 if the gauge has no data in the
        window (or none at all).
    """
    if station_code in riscbal_stations:
        rain = _get_riscbal_precipitation_acc(
            station_code, start_date, end_date, day_offset
        )
        return round(rain, RAIN_DECIMALS)
    if station_code not in rain_index:
        return round(0.0, RAIN_DECIMALS)
    dates, rain = rain_index[station_code]
    # Binary search on the sorted dates: the same days as filtering
    # start <= date < end, summed in the same order, so the result is
    # identical to summing the filtered table.
    first = np.searchsorted(dates, start_date.to_datetime64(), side="left")
    last = np.searchsorted(dates, end_date.to_datetime64(), side="left")
    return round(rain[first:last].sum(), RAIN_DECIMALS)


def add_rain_windows(
    samples: pd.DataFrame,
    rain_index: RainIndex,
    riscbal_stations: tuple[str, ...],
    day_offset: pd.Timedelta,
) -> pd.DataFrame:
    """Add the four rain windows to every sample.

    Parameters
    ----------
    samples : pandas.DataFrame
        One row per sample with ``code`` (the gauge to read) and ``date``.
    rain_index : dict
        From ``index_rain_by_gauge``.
    riscbal_stations : tuple of str
        Gauges read from the RiscBal database.
    day_offset : pandas.Timedelta
        Start of the pluviometric day (07:00 UTC), for the RiscBal gauges.

    Returns
    -------
    pandas.DataFrame
        A copy of ``samples`` with ``event_day_acc`` (rain on the event day)
        and ``last_week_acc``, ``last_month_acc``, ``last_three_months_acc``
        (rain over the 7, 30 and 90 days before it, excluding it).

    Notes
    -----
    Only the date of each event is known, not the time. Event-day rain may
    therefore include rain that fell after the rockfall (operationally, that
    day's rain would come from a forecast).
    """
    with_rain = samples.copy()
    windows = {column: [] for column in RAIN_COLUMNS}
    for code, date in zip(samples["code"], samples["date"], strict=True):
        event_day = pd.Timestamp(date).normalize()
        next_day = event_day + pd.Timedelta(days=1)
        starts = {
            "event_day_acc": event_day,
            "last_week_acc": event_day - pd.Timedelta(days=WEEK_DAYS),
            "last_month_acc": event_day - pd.Timedelta(days=MONTH_DAYS),
            "last_three_months_acc": event_day
            - pd.Timedelta(days=THREE_MONTHS_DAYS),
        }
        for column, start in starts.items():
            end = next_day if column == "event_day_acc" else event_day
            windows[column].append(
                window_rain(
                    code, start, end, rain_index, riscbal_stations, day_offset
                )
            )
    for column in RAIN_COLUMNS:
        with_rain[column] = windows[column]
    return with_rain


def add_rain(
    samples: pd.DataFrame,
    aemet: pd.DataFrame,
    days_by_gauge: dict[str, pd.DatetimeIndex],
    riscbal_stations: tuple[str, ...],
    day_offset: pd.Timedelta,
    output_path: Path,
) -> pd.DataFrame:
    """Add the data coverage and the four rain windows of every sample.

    Writes ``samples_with_accumulations.csv`` to the run's outputs folder.

    Parameters
    ----------
    samples : pandas.DataFrame
        All samples, with ``code`` (the gauge to read) and ``date``.
    aemet : pandas.DataFrame
        The AEMET daily rain.
    days_by_gauge : dict of str to pandas.DatetimeIndex
        Days each AEMET gauge measured.
    riscbal_stations : tuple of str
        Gauges read from the RiscBal database.
    day_offset : pandas.Timedelta
        Start of the pluviometric day (07:00 UTC), for the RiscBal gauges.
    output_path : pathlib.Path
        The run's outputs folder (``runs/<run_id>/outputs/``).

    Returns
    -------
    pandas.DataFrame
        The samples with ``rain_days_90d`` and the rain windows.
    """
    samples["rain_days_90d"] = rain_days_90d(samples, days_by_gauge)
    # Split once by station: the windows look up one station at a time.
    aemet_by_station = dict(tuple(aemet.groupby("code")))
    with_rain = add_rain_windows(
        samples,
        index_rain_by_gauge(aemet_by_station),
        riscbal_stations,
        day_offset,
    )
    with_rain.to_csv(
        output_path / "samples_with_accumulations.csv", index=False
    )
    return with_rain
