"""Check the AEMET OpenData API for the rain that is missing in our local file.

Diagnostic only: it does not change the pipeline or final_gdf.csv.

Some samples have less than 90% of their 90 antecedent days in
data/historical_aemet_data.csv (12 rockfalls have none at all). Missing days
count as 0 mm in the accumulations, so we can't tell "no rain" from "gauge
not recording". This script asks the official API, for the same gauge and the
same window, whether AEMET has those days, and compares the result with the
local file.

Run from the repo root (needs AEMET_API_KEY in .env; the key is free:
https://opendata.aemet.es/centrodedescargas/altaUsuario):

    uv run python -m scripts.aemet_api_check [--run <run_id>]

Inputs, from the outputs of a formatter run (default: the latest; see
run_selection.py): final_gdf.csv (which samples are under-measured),
samples_with_stations.csv (which gauge each sample uses, including the
manual reassignments) and formatted_historical_aemet_data.csv.gz. Outputs in
runs/<run_id>/analyses/aemet_api/:
    daily.csv              the daily values received (one row per gauge-day)
    coverage_check.csv     one row per sample: local vs API days and rain
    overlap_check.csv      days present in both sources, to check they agree
Shared by every run, in cache/aemet_api/:
    responses/             raw API responses (rerunning downloads nothing twice)
    prec_metadata.json     AEMET's own field definitions (what "prec" means)

How the API works: each request returns a small JSON with a "datos" URL; the
data itself is a second download from that URL. Daily requests are limited to
a window of about 6 months, and there is a per-minute rate limit (HTTP 429),
so requests are chunked, paced and retried.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from scripts.run_selection import (
    CACHE_PATH,
    add_run_argument,
    analysis_folder,
    run_folder,
)

API_CACHE_PATH = CACHE_PATH / "aemet_api"
RESPONSE_CACHE_PATH = API_CACHE_PATH / "responses"
AEMET_TABLE_FILE = "formatted_historical_aemet_data.csv.gz"

BASE_URL = "https://opendata.aemet.es/opendata/api"
DAILY_ENDPOINT = (
    "/valores/climatologicos/diarios/datos/"
    "fechaini/{start}T00:00:00UTC/fechafin/{end}T23:59:59UTC/estacion/{gauge}"
)

# Same threshold as the controls and the coverage analysis: 81 of 90 days.
MIN_DAYS_90D = 81
WINDOW_DAYS = 90
# Below the ~6-month limit, with margin.
MAX_CHUNK_DAYS = 150
# Pause between requests, and after a 429 (too many requests).
PAUSE_S = 1.5
RATE_LIMIT_PAUSE_S = 60
MAX_RETRIES = 5


def _get_json(url: str, api_key: str) -> object:
    """GET a URL and parse the JSON body, retrying on rate limits.

    Waits ``RATE_LIMIT_PAUSE_S`` on HTTP 429 (also when the API reports it
    inside a 200 answer) and retries network errors, up to ``MAX_RETRIES``
    attempts.

    Parameters
    ----------
    url : str
        Full request URL.
    api_key : str
        AEMET OpenData API key (sent as the ``api_key`` header).

    Returns
    -------
    object
        The parsed JSON (a dict for API answers, a list for data files).

    Raises
    ------
    SystemExit
        On any other HTTP error (AEMET's explanation is in the message).
    RuntimeError
        If every attempt failed.
    """
    for attempt in range(MAX_RETRIES):
        request = urllib.request.Request(url, headers={"api_key": api_key})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                # The data files are usually ISO-8859-15, not UTF-8.
                charset = response.headers.get_content_charset() or "latin-1"
                body = response.read().decode(charset)
        except urllib.error.HTTPError as error:
            if error.code == 429:
                print(f"  rate limit, waiting {RATE_LIMIT_PAUSE_S}s")
                time.sleep(RATE_LIMIT_PAUSE_S)
                continue
            # AEMET explains the error in the body (e.g. 401: expired key).
            detail = error.read().decode("latin-1", errors="replace")
            raise SystemExit(
                f"HTTP {error.code} from AEMET: {detail}"
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"  network error ({error}), retrying")
            time.sleep(PAUSE_S * (attempt + 2))
            continue
        payload = json.loads(body)
        # The API sometimes reports errors with HTTP 200 and an "estado" field.
        if isinstance(payload, dict) and payload.get("estado") == 429:
            print(f"  rate limit, waiting {RATE_LIMIT_PAUSE_S}s")
            time.sleep(RATE_LIMIT_PAUSE_S)
            continue
        return payload
    raise RuntimeError(f"Gave up after {MAX_RETRIES} attempts: {url}")


def fetch_daily(
    gauge: str, start: pd.Timestamp, end: pd.Timestamp, api_key: str
) -> list[dict]:
    """Daily climatological values of one gauge between two dates (cached).

    Saves AEMET's field definitions to ``prec_metadata.json`` the first time
    the API returns data.

    Parameters
    ----------
    gauge : str
        AEMET gauge code.
    start, end : pandas.Timestamp
        First and last day of the request (inclusive; at most ~6 months).
    api_key : str
        AEMET OpenData API key.

    Returns
    -------
    list of dict
        One record per day AEMET has, as the API returns it (empty if it has
        none).

    Raises
    ------
    RuntimeError
        If the API answers with a status other than 200 or 404.
    """
    cache_file = (
        RESPONSE_CACHE_PATH / f"{gauge}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    )
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    path = DAILY_ENDPOINT.format(
        start=f"{start:%Y-%m-%d}",
        end=f"{end:%Y-%m-%d}",
        gauge=urllib.parse.quote(gauge),
    )
    answer = _get_json(BASE_URL + path, api_key)
    time.sleep(PAUSE_S)

    status = answer.get("estado")
    if status == 404:
        # "No hay datos que satisfagan esos criterios": AEMET has nothing either.
        records = []
    elif status == 200:
        records = _get_json(answer["datos"], api_key)
        time.sleep(PAUSE_S)
        # Saved once: the field definitions (units, what "prec" covers).
        metadata_file = API_CACHE_PATH / "prec_metadata.json"
        if not metadata_file.exists():
            metadata = _get_json(answer["metadatos"], api_key)
            metadata_file.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=1)
            )
    else:
        raise RuntimeError(
            f"{gauge} {start:%Y-%m-%d}..{end:%Y-%m-%d}: {answer}"
        )

    cache_file.write_text(json.dumps(records, ensure_ascii=False))
    return records


def parse_prec(value: object) -> tuple[float, str]:
    """AEMET's "prec" string → (mm, flag).

    "Ip" (inapreciable, < 0.1 mm) → 0.0, like the -3 code in the local file.
    "Acum" (the day's rain is included in a later day's total) → NaN, flagged:
    the local formatter spreads such totals evenly, which we don't redo here.
    Decimals use a comma ("12,3").

    Parameters
    ----------
    value : object
        The ``prec`` field of one API record (a string, or missing).

    Returns
    -------
    mm : float
        Rain in mm, or NaN.
    flag : str
        ``missing`` (no value), ``trace`` (``Ip``, read as 0 mm),
        ``accumulated`` (``Acum``, NaN) or ``value``.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, "missing"
    text = str(value).strip()
    if text == "":
        return np.nan, "missing"
    if text == "Ip":
        return 0.0, "trace"
    if text.startswith("Acum"):
        return np.nan, "accumulated"
    return float(text.replace(",", ".")), "value"


def _chunks(days: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Group the needed days of one gauge into contiguous request ranges.

    The API serves at most about six months per daily request.

    Parameters
    ----------
    days : pandas.DatetimeIndex
        Days to request for one gauge.

    Returns
    -------
    list of tuple of pandas.Timestamp
        (first, last) day of each range; consecutive days, at most
        ``MAX_CHUNK_DAYS`` long.
    """
    ranges = []
    start = previous = None
    for day in days.sort_values().unique():
        if start is None:
            start = previous = day
        elif (day - previous).days > 1 or (day - start).days >= MAX_CHUNK_DAYS:
            ranges.append((start, previous))
            start = day
        previous = day
    if start is not None:
        ranges.append((start, previous))
    return ranges


def _local_daily(outputs: Path) -> pd.DataFrame:
    """The local AEMET file, already reshaped by the formatter.

    Returns
    -------
    pandas.DataFrame
        ``code``, ``date`` and ``precipitation`` (mm), one row per gauge-day
        with a value.
    """
    local = pd.read_csv(outputs / AEMET_TABLE_FILE)
    local["date"] = pd.to_datetime(local["date"], format="%Y-%m-%d")
    return local[["code", "date", "precipitation"]]


def _windows(api: pd.Series, event_day: pd.Timestamp) -> dict[str, float]:
    """The four rain windows from one daily series, as in prec_accumulation.py.

    Antecedent windows are the days *before* the event day (half-open). Days
    without a value are skipped (summed as 0), as in the pipeline; the counts
    tell how complete each sum is.

    Parameters
    ----------
    api : pandas.Series
        Daily rain in mm indexed by date (from the API).
    event_day : pandas.Timestamp
        The sample's event day.

    Returns
    -------
    dict of str to float
        ``event_day_acc``, ``last_week_acc``, ``last_month_acc`` and
        ``last_three_months_acc``, rounded to 0.1 mm.
    """

    def total(start: pd.Timestamp, end: pd.Timestamp) -> float:
        return round(api[(api.index >= start) & (api.index < end)].sum(), 1)

    next_day = event_day + pd.Timedelta(days=1)
    return {
        "event_day_acc": total(event_day, next_day),
        "last_week_acc": total(event_day - pd.Timedelta(days=7), event_day),
        "last_month_acc": total(event_day - pd.Timedelta(days=30), event_day),
        "last_three_months_acc": total(
            event_day - pd.Timedelta(days=WINDOW_DAYS), event_day
        ),
    }


def main() -> None:
    """Compare the under-measured samples' local rain with the AEMET API.

    Writes ``daily.csv``, ``coverage_check.csv`` and ``overlap_check.csv``
    to ``runs/<run_id>/analyses/aemet_api/`` (see the module docstring).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_argument(parser)
    run = run_folder(parser.parse_args().run)
    outputs = run / "outputs"
    analysis = analysis_folder(run, "aemet_api")
    load_dotenv()
    api_key = os.getenv("AEMET_API_KEY")
    if not api_key:
        raise SystemExit("AEMET_API_KEY is not set in .env")
    RESPONSE_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    final = pd.read_csv(outputs / "final_gdf.csv")
    gauges = pd.read_csv(outputs / "samples_with_stations.csv")[
        ["sample_id", "code"]
    ]
    samples = final[final["rain_days_90d"] < MIN_DAYS_90D].merge(
        gauges, on="sample_id", how="left"
    )
    samples["date"] = pd.to_datetime(samples["date"], format="%Y-%m-%d")
    print(
        f"{len(samples)} under-measured samples, {samples['code'].nunique()} gauges"
    )

    # Every gauge-day we need: the 90 antecedent days plus the event day.
    needed = []
    for _, row in samples.iterrows():
        days = pd.date_range(
            row["date"] - pd.Timedelta(days=WINDOW_DAYS), row["date"], freq="D"
        )
        needed.append(pd.DataFrame({"code": row["code"], "date": days}))
    needed = pd.concat(needed).drop_duplicates()

    received = []
    for gauge, group in needed.groupby("code"):
        for start, end in _chunks(pd.DatetimeIndex(group["date"])):
            print(f"{gauge}: {start:%Y-%m-%d} .. {end:%Y-%m-%d}")
            for record in fetch_daily(gauge, start, end, api_key):
                mm, flag = parse_prec(record.get("prec"))
                received.append(
                    {
                        "code": gauge,
                        "date": pd.to_datetime(
                            record["fecha"], format="%Y-%m-%d"
                        ),
                        "prec_raw": record.get("prec"),
                        "prec_mm": mm,
                        "flag": flag,
                    }
                )
    api = pd.DataFrame(
        received, columns=["code", "date", "prec_raw", "prec_mm", "flag"]
    )
    api = api.drop_duplicates(["code", "date"])
    api.to_csv(analysis / "daily.csv", index=False)

    local = _local_daily(outputs)

    # Do the two sources agree where both have a value? (checks units and the
    # local reshaping: tenths of mm, -3 and -4 codes)
    overlap = api.merge(local, on=["code", "date"], how="inner").dropna(
        subset=["prec_mm", "precipitation"]
    )
    overlap["difference_mm"] = (
        overlap["prec_mm"] - overlap["precipitation"]
    ).round(1)
    overlap.to_csv(analysis / "overlap_check.csv", index=False)
    if len(overlap):
        same = (overlap["difference_mm"].abs() <= 0.1).mean()
        print(
            f"\nOverlap: {len(overlap)} gauge-days in both sources, {same:.1%} agree within 0.1 mm"
        )

    rows = []
    for _, row in samples.iterrows():
        event_day = row["date"]
        start = event_day - pd.Timedelta(days=WINDOW_DAYS)
        series = api[
            (api["code"] == row["code"]) & api["prec_mm"].notna()
        ].set_index("date")["prec_mm"]
        antecedent = series[
            (series.index >= start) & (series.index < event_day)
        ]
        flags = api[
            (api["code"] == row["code"])
            & (api["date"] >= start)
            & (api["date"] <= event_day)
        ]
        api_windows = _windows(series, event_day)
        rows.append(
            {
                "sample_id": row["sample_id"],
                "sample_type": row["sample_type"],
                "gauge": row["code"],
                "date": f"{event_day:%Y-%m-%d}",
                "local_days_90d": int(row["rain_days_90d"]),
                "api_days_90d": len(antecedent),
                "api_accumulated_days": int(
                    (flags["flag"] == "accumulated").sum()
                ),
                "api_has_event_day": event_day in series.index,
                "api_complete": len(antecedent) >= MIN_DAYS_90D
                and event_day in series.index,
                **{f"local_{k}": row[k] for k in api_windows},
                **{f"api_{k}": v for k, v in api_windows.items()},
            }
        )
    check = pd.DataFrame(rows)
    check.to_csv(analysis / "coverage_check.csv", index=False)

    print(
        "\nPer sample type: how many the API completes (≥ 81 days + event day)"
    )
    print(
        check.groupby("sample_type")["api_complete"]
        .agg(["sum", "size"])
        .to_string()
    )
    print("\nDays found, local vs API:")
    print(
        check[
            [
                "sample_id",
                "gauge",
                "date",
                "local_days_90d",
                "api_days_90d",
                "api_has_event_day",
                "local_last_three_months_acc",
                "api_last_three_months_acc",
            ]
        ].to_string(index=False)
    )
    print(f"\nWritten to {analysis}/")


if __name__ == "__main__":
    main()
