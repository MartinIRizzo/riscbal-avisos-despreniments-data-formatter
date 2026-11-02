"""One-off: split the old `data/samples.csv` into the formatter's new inputs.

Run once from the repo root:  python -m scripts.split_samples

Why: samples.csv mixed three things that now come from different places.
The positives are rebuilt from the original inventory (samples.csv had their
day and month swapped, see inventory.py), so samples.csv is reduced to what
the inventory doesn't have:

  data/negatives.csv        the negative samples (random points), one row
                            each, with their location attributes, date and
                            rain-gauge station. From now on this file holds
                            negatives only, so it has no label column.
  data/recent_rockfalls.csv rockfalls after the inventory was compiled (the
                            Esporles events of 2024-2026), same columns.
  data/event_stations.csv   rain-gauge station of every rockfall location in
                            samples.csv. POINT_X/POINT_Y are the coordinates
                            of the EVENT, not of the station (those are in
                            stations.csv). The station is the slope-weighted
                            Thiessen polygon that contains the event, as
                            computed by a workmate in ArcGIS.

A station depends only on the place, not on the date, so the swapped dates
in samples.csv don't affect these stations.

The new files keep samples.csv's format (";" separator, "," decimal mark,
ISO dates), which is what the ArcGIS exports use. The script refuses to
overwrite them. samples.csv itself is left untouched; a read-only copy is
in data/backup/2026-10-17/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path("./data")
SAMPLES_FILE = DATA_PATH / "samples.csv"
NEGATIVES_FILE = DATA_PATH / "negatives.csv"
RECENT_EVENTS_FILE = DATA_PATH / "recent_rockfalls.csv"
EVENT_STATIONS_FILE = DATA_PATH / "event_stations.csv"

# Rockfalls in samples.csv that are not in the inventory.
RECENT_EVENTS_MUNICIPALITY = "Esporles"
RECENT_EVENTS_FROM_YEAR = 2024

CSV_FORMAT = {"sep": ";", "decimal": ","}


def main() -> None:
    """Split ``samples.csv`` into the three input files, once.

    Raises
    ------
    FileExistsError
        If any of the output files already exists (never overwritten).
    """
    outputs = [NEGATIVES_FILE, RECENT_EVENTS_FILE, EVENT_STATIONS_FILE]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite: {existing}. Delete them first to regenerate."
        )

    # utf-8-sig: samples.csv starts with a byte-order mark (Excel/ArcGIS export).
    samples = pd.read_csv(SAMPLES_FILE, encoding="utf-8-sig", **CSV_FORMAT)
    # Parsed only to select the recent events; the text in "Fecha" is copied as is.
    dates = pd.to_datetime(samples["Fecha"], format="%Y-%m-%d")
    positives = samples[samples["Tipus"] == "T"]

    negatives = samples[samples["Tipus"] == "F"].drop(columns=["Tipus"])

    recent = positives[
        (positives["Municipio"] == RECENT_EVENTS_MUNICIPALITY)
        & (dates[positives.index].dt.year >= RECENT_EVENTS_FROM_YEAR)
    ].drop(
        columns=["Tipus", "id_station"]
    )  # their station comes from event_stations.csv

    event_stations = positives[
        ["POINT_X", "POINT_Y", "id_station"]
    ].drop_duplicates()
    repeated = event_stations[
        event_stations.duplicated(subset=["POINT_X", "POINT_Y"], keep=False)
    ]
    if not repeated.empty:
        raise ValueError(
            f"Rockfall locations with more than one station:\n{repeated}"
        )

    negatives.to_csv(NEGATIVES_FILE, index=False, **CSV_FORMAT)
    recent.to_csv(RECENT_EVENTS_FILE, index=False, **CSV_FORMAT)
    event_stations.to_csv(EVENT_STATIONS_FILE, index=False, **CSV_FORMAT)

    print(
        f"{SAMPLES_FILE}: {len(samples)} rows ({len(positives)} rockfalls, {len(negatives)} negatives)"
    )
    print(f"  -> {NEGATIVES_FILE}: {len(negatives)} negatives")
    print(f"  -> {RECENT_EVENTS_FILE}: {len(recent)} recent rockfalls")
    print(
        f"  -> {EVENT_STATIONS_FILE}: {len(event_stations)} rockfall locations with their station"
    )


if __name__ == "__main__":
    main()
