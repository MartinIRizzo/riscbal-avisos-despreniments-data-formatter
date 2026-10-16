import os
from datetime import datetime, timezone

import pandas as pd
import psycopg

RISCBAL_STATIONS = [
    "BAR04",
    "GRO08"
]

# AEMET daily precipitation is measured over the "pluviometric day": from
# 07:00 UTC on day D to 07:00 UTC on day D+1, and it is labelled with day D.
# RiscBal stations store 10-minute records, so their windows are shifted by
# the same 7 hours to measure the same "day" as AEMET.
PLUVIOMETRIC_DAY_OFFSET = pd.Timedelta(hours=7)


def get_prec_accumulations_by_date(row, aemet_df: pd.DataFrame) -> pd.DataFrame:
    station_code = row["code"]
    # Only the date of the event is known (no time). Normalising to midnight
    # guarantees that the windows below select whole AEMET days, even if a
    # date was ever stored with a time.
    event_day = row["date"].normalize()
    next_day = event_day + pd.Timedelta(days=1)

    # Event-day rain: the triggering rain, which is what the official
    # 90 mm / 24 h threshold refers to. Caveat: with daily data it may include
    # rain that fell after the rockfall (operationally, that day's rain would
    # come from a forecast).
    event_day_acc = round(_compute_accumulation(station_code, event_day, next_day, aemet_df), 1)

    # Antecedent rain: the days *before* the event, excluding the event day.
    last_week_acc = round(_compute_accumulation(station_code, event_day - pd.Timedelta(days=7), event_day, aemet_df), 1)
    last_month_acc = round(_compute_accumulation(station_code, event_day - pd.Timedelta(days=30), event_day, aemet_df), 1)
    last_three_months_acc = round(_compute_accumulation(station_code, event_day - pd.Timedelta(days=90), event_day, aemet_df), 1)

    row["event_day_acc"] = event_day_acc
    row["last_week_acc"] = last_week_acc
    row["last_month_acc"] = last_month_acc
    row["last_three_months_acc"] = last_three_months_acc

    return row

def _compute_accumulation(
    station_code: str, start_date: datetime, end_date: datetime, aemet_df: pd.DataFrame
):
    """Rain from start_date (inclusive) to end_date (exclusive), in days.

    Missing data counts as 0 in both branches: AEMET days absent from the
    file are simply not summed, and the database returns no rows. This is a
    known limitation (see TODO: formatter refactor).
    """
    if station_code in RISCBAL_STATIONS:
        return _get_riscbal_precipitation_acc(station_code, start_date, end_date)

    return aemet_df[
        (aemet_df["code"] == station_code)
        & (aemet_df["date"] >= start_date)
        & (aemet_df["date"] < end_date)
        ]["precipitation"].sum()


def _get_riscbal_precipitation_acc(
    station_code: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> float:
    # Same pluviometric days as AEMET: day D runs 07:00 UTC D -> 07:00 UTC D+1.
    start = (start_date + PLUVIOMETRIC_DAY_OFFSET).to_pydatetime().replace(tzinfo=timezone.utc)
    end = (end_date + PLUVIOMETRIC_DAY_OFFSET).to_pydatetime().replace(tzinfo=timezone.utc)

    with psycopg.connect(
        f"dbname={os.getenv('DATABASE_NAME')} "
        f"user={os.getenv('DATABASE_USER')} "
        f"password={os.getenv('DATABASE_PASSWORD')} "
        f"host={os.getenv('DATABASE_HOST')} "
        f"port={os.getenv('DATABASE_PORT')}"
    ) as conn:
        with conn.cursor() as cur:
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
                (station_code, start, end)
            )

            result = cur.fetchone()
            # SUM over no rows is NULL (None). Treat it as 0, like the AEMET
            # branch, instead of crashing later in round(None).
            if result is None or result[0] is None:
                return 0.0
            return float(result[0])
