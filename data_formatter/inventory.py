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
`scripts/split_samples.py`. negatives.csv holds negatives only, so a workmate can
regenerate the negatives without touching the rockfalls.

Stations: a slope-weighted Thiessen polygon per station, computed in ArcGIS
by a workmate. We don't have the polygon layer, only its result for each
point, so rockfalls get their station from event_stations.csv (matched by the
event's coordinates) and negatives.csv carries the station of each point.
With the layer, both would be replaced by a point-in-polygon join.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

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

# Columns of the table built here, in the old samples.csv order, plus two
# columns to trace each sample:
#   sample_id    inv_<inventory line> (inventory rockfalls), recent_<row> (row
#                of recent_rockfalls.csv), neg_<row> (row of negatives.csv);
#                rows counted from 1 after the header. The temporal controls
#                (controls.py) get <rockfall sample_id>_c<nn>.
#   sample_type  rockfall | random_point (| temporal_control, in controls.py)
SAMPLE_COLUMNS = (
    ["Municipio"]
    + LOCATION_COLUMNS
    + [
        "Fecha",
        "POINT_X",
        "POINT_Y",
        "Tipus",
        "id_station",
        "sample_id",
        "sample_type",
    ]
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _read_sample_file(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read one of the samples.csv-style inputs and parse its ISO dates.

    Parameters
    ----------
    path : pathlib.Path
        The CSV file (``;``-separated, decimal comma).
    columns : list of str
        Columns the file must have.

    Returns
    -------
    pandas.DataFrame
        The file with text stripped and, if it has ``Fecha``, a parsed
        ``date``.

    Raises
    ------
    ValueError
        If any of ``columns`` is missing.
    """
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
    """Strip leading and trailing spaces from every text column.

    Parameters
    ----------
    df : pandas.DataFrame
        Any table; changed in place.

    Returns
    -------
    pandas.DataFrame
        The same table.
    """
    for column in df.columns:
        if pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].str.strip()
    return df


def _coordinate_key(df: pd.DataFrame) -> pd.Series:
    """A text key per row from its coordinates, to match places across files.

    Coordinates are rounded to ``COORDINATE_DECIMALS`` so that the same
    point written with a different number of decimals still matches.

    Parameters
    ----------
    df : pandas.DataFrame
        With ``POINT_X`` and ``POINT_Y``.

    Returns
    -------
    pandas.Series
        ``"<x>_<y>"`` per row.
    """
    x = df["POINT_X"].round(COORDINATE_DECIMALS).astype(str)
    y = df["POINT_Y"].round(COORDINATE_DECIMALS).astype(str)
    return x + "_" + y


def _normalise_hardness(df: pd.DataFrame) -> pd.DataFrame:
    """Write the hardness class "Moderately" as "Moderately hard rocks".

    Both labels are the same lithology class in the inputs.

    Parameters
    ----------
    df : pandas.DataFrame
        With ``DURESA``; changed in place.

    Returns
    -------
    pandas.DataFrame
        The same table.
    """
    df["DURESA"] = df["DURESA"].replace(
        {"Moderately": "Moderately hard rocks"}
    )
    return df


def _report(step: str, n: int) -> None:
    """Log one step of the sample selection and how many rows are left.

    Parameters
    ----------
    step : str
        What the step did.
    n : int
        Rows after the step.
    """
    logger.info("  %-62s %5d", step, n)


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------


def load_inventory_events(data_path: Path) -> pd.DataFrame:
    """Slides and rockfalls from the inventory, dated within the study period.

    Keeps events of type ``EVENT_TYPE`` with an exact date (parsed with
    ``INVENTORY_DATE_FORMAT``) in ``PERIOD_START`` .. ``PERIOD_END``, with
    every ``REQUIRED_COLUMNS`` value present, and drops duplicates (same
    place and date). Each step's count is logged.

    Parameters
    ----------
    data_path : pathlib.Path
        Input folder (holds ``INVENTORY_FILE``).

    Returns
    -------
    pandas.DataFrame
        The selected events, with a parsed ``date`` and ``inventory_line``
        (line number in the CSV file, header = line 1).
    """
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
    events["date"] = pd.to_datetime(
        events["Fecha"], format=INVENTORY_DATE_FORMAT, errors="coerce"
    )
    events = events[events["date"].between(PERIOD_START, PERIOD_END)]
    _report(
        f"dated within {PERIOD_START.date()} .. {PERIOD_END.date()}",
        len(events),
    )

    before = len(events)
    events = events.dropna(subset=REQUIRED_COLUMNS)
    _report("after dropping missing required values", len(events))
    if len(events) < before:
        logger.info(
            "    (dropped %d with missing %s)",
            before - len(events),
            REQUIRED_COLUMNS,
        )

    events = events.drop_duplicates(subset=["POINT_X", "POINT_Y", "date"])
    _report("after dropping duplicates (same place and date)", len(events))
    return events


def assign_stations(
    events: pd.DataFrame, event_stations: pd.DataFrame, label: str
) -> pd.DataFrame:
    """Take each rockfall's rain gauge from event_stations.csv, by place.

    The stations come from slope-weighted Thiessen polygons built in ArcGIS.
    They can't be recomputed from stations.csv (the nearest station agrees
    only for ~65% of samples, as expected for a slope-weighted partition),
    so they are reused as is. Events without a station are excluded and
    logged.

    Parameters
    ----------
    events : pandas.DataFrame
        Rockfalls with ``POINT_X`` and ``POINT_Y``.
    event_stations : pandas.DataFrame
        ``event_stations.csv``: ``POINT_X``, ``POINT_Y``, ``id_station``.
    label : str
        What the events are, for the log messages.

    Returns
    -------
    pandas.DataFrame
        The events that have a station, with ``id_station`` (int).

    Raises
    ------
    ValueError
        If a location has more than one station in ``event_stations``.
    """
    station_by_location = event_stations.groupby(
        _coordinate_key(event_stations)
    )["id_station"].unique()
    conflicting = station_by_location[station_by_location.apply(len) > 1]
    if not conflicting.empty:
        raise ValueError(
            f"Locations with more than one station in {EVENT_STATIONS_FILE}: {conflicting.to_dict()}"
        )

    events = events.copy()
    events["id_station"] = _coordinate_key(events).map(
        station_by_location.apply(lambda ids: ids[0])
    )
    unmatched = events[events["id_station"].isna()]
    if not unmatched.empty:
        logger.warning(
            "%d %s have no station in %s, excluded:\n%s",
            len(unmatched),
            label,
            EVENT_STATIONS_FILE,
            unmatched[["Municipio", "Fecha", "POINT_X", "POINT_Y"]].to_string(
                index=False
            ),
        )
    events = events.dropna(subset=["id_station"])
    events["id_station"] = events["id_station"].astype(int)
    _report(f"{label} with a station from {EVENT_STATIONS_FILE}", len(events))
    return events


def build_samples(data_path: Path, output_path: Path) -> pd.DataFrame:
    """Samples in samples.csv format: inventory + recent rockfalls + negatives.

    Writes ``samples_from_inventory.csv`` (the table) and
    ``inventory_selection_report.csv`` (where each rockfall comes from) to
    the run's outputs folder.

    Parameters
    ----------
    data_path : pathlib.Path
        Input folder (inventory, ``recent_rockfalls.csv``,
        ``negatives.csv``, ``event_stations.csv``).
    output_path : pathlib.Path
        The run's outputs folder (``runs/<run_id>/outputs/``).

    Returns
    -------
    pandas.DataFrame
        One row per sample with the ``SAMPLE_COLUMNS``: ``sample_id``
        (``inv_<line>``, ``recent_<n>``, ``neg_<n>``), ``sample_type``
        (rockfall or random_point), ``Tipus`` (T/F), ``Fecha`` and the
        attributes.
    """
    logger.info("Building samples:")
    event_stations = _read_sample_file(
        data_path / EVENT_STATIONS_FILE, ["POINT_X", "POINT_Y", "id_station"]
    )
    sample_columns = [
        column
        for column in SAMPLE_COLUMNS
        if column not in ("Tipus", "id_station", "sample_id", "sample_type")
    ]

    events = assign_stations(
        load_inventory_events(data_path), event_stations, "inventory events"
    )
    events["source"] = "inventory"
    events["sample_id"] = "inv_" + events["inventory_line"].astype(str)

    recent = _read_sample_file(data_path / RECENT_EVENTS_FILE, sample_columns)
    recent["sample_id"] = [
        f"recent_{row}" for row in range(1, len(recent) + 1)
    ]
    _report(f"recent rockfalls in {RECENT_EVENTS_FILE}", len(recent))
    recent = assign_stations(recent, event_stations, "recent rockfalls")
    recent["source"] = RECENT_EVENTS_FILE

    negatives = _read_sample_file(
        data_path / NEGATIVES_FILE, sample_columns + ["id_station"]
    )
    negatives["Tipus"] = "F"
    negatives["sample_id"] = [
        f"neg_{row}" for row in range(1, len(negatives) + 1)
    ]
    negatives["sample_type"] = "random_point"
    negatives["source"] = NEGATIVES_FILE
    _report(f"negatives in {NEGATIVES_FILE}", len(negatives))

    rockfalls = pd.concat([events, recent], ignore_index=True)
    rockfalls["Tipus"] = "T"
    rockfalls["sample_type"] = "rockfall"
    combined = pd.concat([rockfalls, negatives], ignore_index=True)
    combined = _normalise_hardness(combined)
    combined["Fecha"] = combined["date"].dt.strftime(SAMPLES_DATE_FORMAT)

    table = combined[SAMPLE_COLUMNS]
    _report("total samples", len(table))
    logger.info(
        "  positives: %d, negatives: %d",
        (table["Tipus"] == "T").sum(),
        (table["Tipus"] == "F").sum(),
    )

    output_path.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path / "samples_from_inventory.csv", index=False)
    _selection_report(events, recent).to_csv(
        output_path / "inventory_selection_report.csv", index=False
    )
    return table


def _selection_report(
    events: pd.DataFrame, recent: pd.DataFrame
) -> pd.DataFrame:
    """Where each rockfall comes from (``inventory_selection_report.csv``).

    The comparison with the old, swapped samples.csv dates was a one-off
    check; see progress/2026-10-16-samples-from-inventory.md.

    Parameters
    ----------
    events : pandas.DataFrame
        The selected inventory rockfalls (from ``assign_stations``).
    recent : pandas.DataFrame
        The recent rockfalls (``recent_rockfalls.csv``).

    Returns
    -------
    pandas.DataFrame
        One row per rockfall: ``sample_id``, ``source``, ``inventory_line``,
        ``classification``, ``data_source``, ``municipality``, ``date``,
        ``id_station`` (inventory columns empty for the recent ones).
    """
    report = pd.DataFrame(
        {
            "sample_id": events["sample_id"],
            "source": events["source"],
            "inventory_line": events["inventory_line"],
            "classification": events["classifica"],
            "data_source": events["Fuente_dat"],
            "municipality": events["Municipio"],
            "date": events["date"].dt.date,
            "id_station": events["id_station"],
        }
    )
    recent_rows = pd.DataFrame(
        {
            "sample_id": recent["sample_id"],
            "source": recent["source"],
            "municipality": recent["Municipio"],
            "date": recent["date"].dt.date,
            "id_station": recent["id_station"],
        }
    )
    return pd.concat([report, recent_rows], ignore_index=True)
