"""Rejoin the inventory's record columns to their events, by identifier.

``inventari_depurat_coordenades.csv`` is two tables side by side:

  event columns   Año ... elev_PCT90, OBJECTID, POINT_X, POINT_Y
                  where the event is (municipality, hillside, type,
                  material, hardness, terrain statistics, coordinates)
  record columns  FID_1, Id, localizaci ... Temporada
                  the inventory record: the date (Fecha), the description,
                  the data source, and a second copy of year, month,
                  municipality, hillside, type, material and damage

In lines ~810-934 of the file the record columns are not on their event's
line: they were reordered as a block, and 6 event lines were deleted. The
identifiers still tell which record belongs to which event: ``OBJECTID``
stays on the event's line and ``FID_1`` moves with the record, and a record
belongs to the event whose ``OBJECTID`` is its ``FID_1`` + 1 (the
relation every row of the original layer keeps; in the rows of this file
that were not moved, OBJECTID = FID_1 + 1 too).

``realign_records`` applies that join before the rockfalls are selected, so
each event gets its own date. It then checks every joined pair: the fields
written in both tables must agree wherever both are filled (the type by
family, since the record often has a finer one, e.g. "Deslizamiento
rotacional" for "Esllavissades/Despreniments"). A contradiction stops the
run: it would mean the join is wrong for this file (e.g. a new version of
it). On the current file there is none, over 927 pairs.

What the join can't check: the day of the date (only the year and month are
written in both tables). What it doesn't fix: the terrain statistics
(``slope_*``, ``elev_*``) are also misplaced in this file (one line off, for
most of it), but they have no identifier; they aren't features (export.py
drops them), so they are left as they are.

Every run writes ``inventory_realignment.csv`` to the outputs folder: each
line whose record changes (with the date and description before and after)
and each record left without an event.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

REPORT_FILE = "inventory_realignment.csv"

# Identifiers: the event's stays on its line, the record's moves with it.
EVENT_ID = "OBJECTID"
RECORD_ID = "FID_1"
# A record belongs to the event with EVENT_ID == RECORD_ID + ID_OFFSET.
ID_OFFSET = 1

# The record columns: from FIRST to LAST in the file, except EVENT_ID.
FIRST_RECORD_COLUMN = "FID_1"
LAST_RECORD_COLUMN = "Temporada"

# Fields written in both tables: (event column, record column).
SHARED_FIELDS = [
    ("Año", "Año_1"),
    ("Mes", "Mes_1"),
    ("Municipio", "Municipio_1"),
    ("Vertiente", "Vertiente_1"),
    ("Tipo", "Tipo_1"),
    ("Material", "Material_1"),
    ("Daños", "Daños_1"),
]

# Type families, to compare the event's type (Catalan) with the record's
# (Spanish, often finer or mixed: "Flujo y desprendimiento" is both a flow
# and a rockfall). Two types agree if they share a family. Keys are
# lower-case fragments; misspellings found in the file are included.
TYPE_FAMILIES = {
    "esllav": "slide_or_rockfall",
    "despren": "slide_or_rockfall",
    "desliz": "slide_or_rockfall",
    "slump": "slide_or_rockfall",
    "fluxe": "flow",
    "fluj": "flow",
    "fluy": "flow",
    "griet": "crack",
    "gireta": "crack",
    "col·l": "collapse",
    "colap": "collapse",
    "enfons": "subsidence",
    "hundim": "subsidence",
}


@dataclass(frozen=True)
class RealignmentSettings:
    """``inventory_realignment`` in config.yaml.

    Attributes
    ----------
    enabled : bool
        Feature flag. Off reads the file as it is (the exports before
        2026-11-16).
    """

    enabled: bool


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def record_columns(columns: pd.Index) -> list[str]:
    """The record columns of the inventory, in file order.

    Parameters
    ----------
    columns : pandas.Index
        The inventory's columns.

    Returns
    -------
    list of str
        ``FIRST_RECORD_COLUMN`` .. ``LAST_RECORD_COLUMN``, without
        ``EVENT_ID``.
    """
    names = list(columns)
    block = names[
        names.index(FIRST_RECORD_COLUMN) : names.index(LAST_RECORD_COLUMN) + 1
    ]
    return [name for name in block if name != EVENT_ID]


def _normal(value: object) -> str:
    """A field as comparable text: "" if empty, integers without ".0"."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip().lower()


def _families(value: str) -> set[str]:
    """The type families a type belongs to (see ``TYPE_FAMILIES``)."""
    return {family for key, family in TYPE_FAMILIES.items() if key in value}


def contradictions(inventory: pd.DataFrame) -> pd.DataFrame:
    """Lines whose event and record columns disagree on a shared field.

    Parameters
    ----------
    inventory : pandas.DataFrame
        The inventory, with ``inventory_line``.

    Returns
    -------
    pandas.DataFrame
        One row per contradicting field: ``inventory_line``, ``field``,
        ``event_value``, ``record_value``. Empty if none.
    """
    rows = []
    for event_column, record_column in SHARED_FIELDS:
        for line, mine, theirs in zip(
            inventory["inventory_line"],
            inventory[event_column].map(_normal),
            inventory[record_column].map(_normal),
            strict=True,
        ):
            if not mine or not theirs:
                continue
            if event_column == "Tipo":
                differ = not _families(mine) & _families(theirs)
            else:
                differ = mine != theirs
            if differ:
                rows.append((line, event_column, mine, theirs))
    return pd.DataFrame(
        rows, columns=["inventory_line", "field", "event_value", "record_value"]
    )


# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------


def realign_records(
    inventory: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    """Give each event line the record whose identifier points to it.

    Line i gets the record columns of the line whose ``RECORD_ID`` equals
    line i's ``EVENT_ID`` - ``ID_OFFSET``; lines without one (no
    ``EVENT_ID``, or no record points to them) get empty record columns,
    so they have no date and are never selected. Writes ``REPORT_FILE``.

    Parameters
    ----------
    inventory : pandas.DataFrame
        The inventory as read, with ``inventory_line``.
    output_path : pathlib.Path
        The run's outputs folder.

    Returns
    -------
    pandas.DataFrame
        The inventory with its record columns rejoined (same rows, same
        order, same ``inventory_line``).

    Raises
    ------
    ValueError
        If ``RECORD_ID`` repeats, or if a rejoined line contradicts itself
        on a shared field.
    """
    columns = record_columns(inventory.columns)
    record_ids = inventory[RECORD_ID].dropna()
    if record_ids.duplicated().any():
        repeated = sorted(record_ids[record_ids.duplicated()].unique())
        raise ValueError(f"{RECORD_ID} repeats in the inventory: {repeated}")

    row_of_record = pd.Series(
        inventory.index[inventory[RECORD_ID].notna()],
        index=record_ids.astype(int).to_numpy(),
    )
    wanted = (inventory[EVENT_ID] - ID_OFFSET).round()
    source = wanted.map(row_of_record)  # NaN: no record for this line

    realigned = inventory.copy()
    has_record = source.notna()
    realigned[columns] = np.nan
    realigned[columns] = realigned[columns].astype(object)
    realigned.loc[has_record, columns] = (
        inventory.loc[source[has_record].astype(int), columns].to_numpy()
    )

    found = contradictions(realigned)
    if not found.empty:
        raise ValueError(
            "The identifier join gives contradicting lines; the inventory "
            f"is not the expected file:\n{found.to_string(index=False)}"
        )

    report = _report(inventory, realigned, source)
    output_path.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path / REPORT_FILE, index=False)
    moved = (report["status"] == "record moved").sum()
    lost = (report["status"] == "record lost").sum()
    orphans = (report["status"] == "record without event").sum()
    logger.info(
        "  Inventory realigned by identifier (%s = %s + %d): %d lines get "
        "another line's record, %d lose theirs, %d records have no event; "
        "no contradictions in %d joined lines (%s)",
        EVENT_ID, RECORD_ID, ID_OFFSET, moved, lost, orphans,
        int(has_record.sum()), REPORT_FILE,
    )
    return realigned


def _report(
    before: pd.DataFrame, after: pd.DataFrame, source: pd.Series
) -> pd.DataFrame:
    """Every line whose record changes, and every record left without event.

    Parameters
    ----------
    before, after : pandas.DataFrame
        The inventory as read and as realigned.
    source : pandas.Series
        Per line, the row its record comes from (NaN if none).

    Returns
    -------
    pandas.DataFrame
        ``status`` (``record moved``: the line gets another line's record;
        ``record lost``: the line had a record and now has none; ``record
        without event``: no line takes this record), ``inventory_line``,
        ``record_from_line``, and ``Fecha``, ``localizaci`` and ``Tipo``
        before and after.
    """
    own = pd.Series(before.index, index=before.index)
    had_record = before[RECORD_ID].notna()
    moved = source.notna() & (source != own)
    lost = had_record & source.isna()
    line_of = before["inventory_line"]

    lines = before.loc[moved | lost]
    report = pd.DataFrame(
        {
            "status": np.where(moved[lines.index], "record moved", "record lost"),
            "inventory_line": lines["inventory_line"],
            "record_from_line": source[lines.index].map(line_of),
            "Tipo": lines["Tipo"],
            "Fecha_before": lines["Fecha"],
            "Fecha_after": after.loc[lines.index, "Fecha"],
            "localizaci_before": lines["localizaci"],
            "localizaci_after": after.loc[lines.index, "localizaci"],
        }
    )
    taken = set(source.dropna().astype(int))
    unused = before.loc[had_record & ~before.index.isin(taken)]
    orphans = pd.DataFrame(
        {
            "status": "record without event",
            "record_from_line": unused["inventory_line"],
            "Fecha_before": unused["Fecha"],
            "localizaci_before": unused["localizaci"],
        }
    )
    out = pd.concat([report, orphans], ignore_index=True)
    for column in ("inventory_line", "record_from_line"):
        out[column] = out[column].astype("Int64")
    return out
