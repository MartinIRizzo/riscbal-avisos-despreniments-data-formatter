"""Build the sample table: inventory rockfalls + recent rockfalls + negatives.

The table produced here has the same columns and meaning as the old
`data/samples.csv` (one row per sample: location attributes, date, label,
rain-gauge station). Each kind of sample comes from its own input file:

  positives  data/inventari_depurat_coordenades.csv  the original inventory
             data/recent_rockfalls.csv               events after it (Esporles 2024-2026)
  stations   data/event_stations.csv                 station of each rockfall location
  negatives  data/negatives.csv                      random points, negatives only

Why the inventory and not samples.csv: in samples.csv, every rockfall whose
day of month is <= 12 had its day and month swapped (e.g. inventory 8/1/2009
= 8 January -> samples.csv 2009-08-01 = 1 August). The conversion happened
before samples.csv was saved, outside this repository. Reading the inventory
directly, with an explicit day-first format, removes that error at the source.

The other three files were split once from samples.csv by
`split_samples.py`. negatives.csv holds negatives only, so a workmate can
regenerate the negatives without touching the rockfalls.

Stations: a slope-weighted Thiessen polygon per station, computed in ArcGIS
by a workmate. We don't have the polygon layer, only its result for each
point, so rockfalls get their station from event_stations.csv (matched by the
event's coordinates) and negatives.csv carries the station of each point.
With the layer, both would be replaced by a point-in-polygon join.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

INVENTORY_FILE = "inventari_depurat_coordenades.csv"
RECENT_EVENTS_FILE = "recent_rockfalls.csv"
EVENT_STATIONS_FILE = "event_stations.csv"
NEGATIVES_FILE = "negatives.csv"

# Inventory event type to keep: slides and rockfalls ("esllavissades /
# despreniments"). Excludes flows, cracks, collapses and subsidence.
EVENT_TYPE = "Esllavissades/Despreniments"

# The inventory writes dates as day/month/year. Stating the format explicitly
# is the whole point: without it, a parser may read 8/1/2009 as August 1st.
INVENTORY_DATE_FORMAT = "%d/%m/%Y %H:%M:%S"
# The other inputs, and the table built here, use ISO dates.
SAMPLES_DATE_FORMAT = "%Y-%m-%d"

# Study period. It also removes the inventory's placeholder dates for unknown
# events (1899, Excel's day zero) and the few historical events.
PERIOD_START = pd.Timestamp("2005-01-01")
PERIOD_END = pd.Timestamp("2015-12-31")

# Coordinates are matched after rounding to 0.1 m: the two files store the
# same points with different numbers of decimals.
COORDINATE_DECIMALS = 1

LOCATION_COLUMNS = ["Vertiente", "DURESA"] + [
    f"{prefix}_{stat}"
    for prefix in ("slope", "elev")
    for stat in ("MIN", "MAX", "RANGE", "MEAN", "STD", "MEDIAN", "PCT90")
]
# Rows missing any of these cannot be used (no place, no date, no attributes).
REQUIRED_COLUMNS = ["Fecha", "POINT_X", "POINT_Y", "Vertiente", "DURESA"]

# Columns of the table built here, in the old samples.csv order.
SAMPLE_COLUMNS = ["Municipio"] + LOCATION_COLUMNS + ["Fecha", "POINT_X", "POINT_Y", "Tipus", "id_station"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _read_sample_file(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read one of the samples.csv-style inputs and parse its ISO dates."""
    # utf-8-sig accepts files with or without the byte-order mark that
    # Excel/ArcGIS exports start with.
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    df = _strip_text(df)
    if "Fecha" in df.columns:
        df["date"] = pd.to_datetime(df["Fecha"], format=SAMPLES_DATE_FORMAT)
    return df


def _strip_text(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].str.strip()
    return df


def _coordinate_key(df: pd.DataFrame) -> pd.Series:
    x = df["POINT_X"].round(COORDINATE_DECIMALS).astype(str)
    y = df["POINT_Y"].round(COORDINATE_DECIMALS).astype(str)
    return x + "_" + y


def _normalise_hardness(df: pd.DataFrame) -> pd.DataFrame:
    # "Moderately" and "Moderately hard rocks" are the same lithology class.
    df["DURESA"] = df["DURESA"].replace({"Moderately": "Moderately hard rocks"})
    return df


def _report(step: str, n: int) -> None:
    print(f"  {step:<62s} {n:>5d}")


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------

def load_inventory_events(data_path: Path) -> pd.DataFrame:
    """Slides and rockfalls from the inventory, dated within the study period."""
    inventory = pd.read_csv(
        data_path / INVENTORY_FILE, sep=";", decimal=",", encoding="utf-8-sig"
    )
    inventory = _strip_text(inventory)
    # Line number in the CSV file (header is line 1), to trace each sample back.
    inventory["inventory_line"] = inventory.index + 2
    _report("inventory events", len(inventory))

    events = inventory[inventory["Tipo"] == EVENT_TYPE].copy()
    _report(f"type '{EVENT_TYPE}'", len(events))

    # errors="coerce": a missing or malformed date becomes NaT and is dropped
    # below as a missing required value, instead of being guessed.
    events["date"] = pd.to_datetime(events["Fecha"], format=INVENTORY_DATE_FORMAT, errors="coerce")
    events = events[events["date"].between(PERIOD_START, PERIOD_END)]
    _report(f"dated within {PERIOD_START.date()} .. {PERIOD_END.date()}", len(events))

    before = len(events)
    events = events.dropna(subset=REQUIRED_COLUMNS)
    _report("after dropping missing required values", len(events))
    if len(events) < before:
        print(f"    (dropped {before - len(events)} with missing {REQUIRED_COLUMNS})")

    events = events.drop_duplicates(subset=["POINT_X", "POINT_Y", "date"])
    _report("after dropping duplicates (same place and date)", len(events))
    return events


def assign_stations(events: pd.DataFrame, event_stations: pd.DataFrame, label: str) -> pd.DataFrame:
    """Take each rockfall's rain-gauge station from event_stations.csv, by coordinates.

    The stations come from slope-weighted Thiessen polygons built in ArcGIS.
    They can't be recomputed from stations.csv (the nearest station agrees
    only for ~65% of samples, as expected for a slope-weighted partition),
    so they are reused as is. Events without a station are excluded.
    """
    station_by_location = event_stations.groupby(_coordinate_key(event_stations))["id_station"].unique()
    conflicting = station_by_location[station_by_location.apply(len) > 1]
    if not conflicting.empty:
        raise ValueError(f"Locations with more than one station in {EVENT_STATIONS_FILE}: {conflicting.to_dict()}")

    events = events.copy()
    events["id_station"] = _coordinate_key(events).map(station_by_location.apply(lambda ids: ids[0]))
    unmatched = events[events["id_station"].isna()]
    if not unmatched.empty:
        print(f"    WARNING: {len(unmatched)} {label} have no station in {EVENT_STATIONS_FILE}, excluded:")
        print(unmatched[["Municipio", "Fecha", "POINT_X", "POINT_Y"]].to_string(index=False))
    events = events.dropna(subset=["id_station"])
    events["id_station"] = events["id_station"].astype(int)
    _report(f"{label} with a station from {EVENT_STATIONS_FILE}", len(events))
    return events


def build_samples(data_path: Path, results_path: Path) -> pd.DataFrame:
    """Samples in samples.csv format: inventory rockfalls + recent rockfalls + negatives.

    Writes `samples_from_inventory.csv` (the table) and
    `inventory_selection_report.csv` (where each rockfall comes from).
    """
    print("Building samples:")
    event_stations = _read_sample_file(data_path / EVENT_STATIONS_FILE, ["POINT_X", "POINT_Y", "id_station"])
    sample_columns = [column for column in SAMPLE_COLUMNS if column not in ("Tipus", "id_station")]

    events = assign_stations(load_inventory_events(data_path), event_stations, "inventory events")
    events["source"] = "inventory"

    recent = _read_sample_file(data_path / RECENT_EVENTS_FILE, sample_columns)
    _report(f"recent rockfalls in {RECENT_EVENTS_FILE}", len(recent))
    recent = assign_stations(recent, event_stations, "recent rockfalls")
    recent["source"] = RECENT_EVENTS_FILE

    negatives = _read_sample_file(data_path / NEGATIVES_FILE, sample_columns + ["id_station"])
    negatives["Tipus"] = "F"
    negatives["source"] = NEGATIVES_FILE
    _report(f"negatives in {NEGATIVES_FILE}", len(negatives))

    rockfalls = pd.concat([events, recent], ignore_index=True)
    rockfalls["Tipus"] = "T"
    combined = pd.concat([rockfalls, negatives], ignore_index=True)
    combined = _normalise_hardness(combined)
    combined["Fecha"] = combined["date"].dt.strftime(SAMPLES_DATE_FORMAT)

    table = combined[SAMPLE_COLUMNS]
    _report("total samples", len(table))
    print(f"  positives: {(table['Tipus'] == 'T').sum()}, negatives: {(table['Tipus'] == 'F').sum()}")

    results_path.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_path / "samples_from_inventory.csv", index=False)
    _selection_report(events, recent).to_csv(results_path / "inventory_selection_report.csv", index=False)
    return table


def _selection_report(events: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    """One row per rockfall: origin, inventory line and classification, station.

    (The comparison with the old, swapped samples.csv dates was a one-off
    check; see progress/2026-10-16-samples-from-inventory.md.)
    """
    report = pd.DataFrame({
        "source": events["source"],
        "inventory_line": events["inventory_line"],
        "classification": events["classifica"],
        "data_source": events["Fuente_dat"],
        "municipality": events["Municipio"],
        "date": events["date"].dt.date,
        "id_station": events["id_station"],
    })
    recent_rows = pd.DataFrame({
        "source": recent["source"],
        "municipality": recent["Municipio"],
        "date": recent["date"].dt.date,
        "id_station": recent["id_station"],
    })
    return pd.concat([report, recent_rows], ignore_index=True)
