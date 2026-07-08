import os
from datetime import datetime, timezone

import pandas as pd
import psycopg

RISCBAL_STATIONS = [
    "BAR04",
    "GRO08"
]

def get_prec_accumulations_by_date(row, aemet_df: pd.DataFrame) -> pd.DataFrame:
    station_code = row["code"]

    last_day_acc = round(_compute_accumulation(station_code, row["date"] - pd.Timedelta(days=1), row["date"], aemet_df), 1)
    last_week_acc = round(_compute_accumulation(station_code, row["date"] - pd.Timedelta(days=7), row["date"], aemet_df), 1)
    last_month_acc = round(_compute_accumulation(station_code, row["date"] - pd.Timedelta(days=30), row["date"], aemet_df), 1)
    last_three_months_acc = round(_compute_accumulation(station_code, row["date"] - pd.Timedelta(days=90), row["date"], aemet_df), 1)

    row["last_day_acc"] = last_day_acc
    row["last_week_acc"] = last_week_acc
    row["last_month_acc"] = last_month_acc
    row["last_three_months_acc"] = last_three_months_acc

    return row

def _compute_accumulation(
    station_code: str, start_date: datetime, end_date: datetime, aemet_df: pd.DataFrame
):
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
                (station_code,
                 start_date.to_pydatetime().replace(tzinfo=timezone.utc),
                 end_date.to_pydatetime().replace(tzinfo=timezone.utc))
            )

            result = cur.fetchone()
            return result[0] if result else 0