"""How many rows each column block of the inventory is shifted, row by row.

Diagnostic only (no database, doesn't change the pipeline).
``inventari_depurat_coordenades.csv`` is several tables pasted side by side,
and in parts of the file they don't describe the same event on the same row
(the fix is data_formatter/inventory_alignment.py, which uses test 6). This script measures the shift of each block of
columns against the coordinates, row by row, with independent evidence.
Run from the repo root::

    uv run python -m scripts.inventory_shift_study [--window 110]

Column blocks (by position in the file):

    attributes  Año ... GEO_SIMPL_code   (municipality, hillside, type,
                                          material, damage, hardness, geology)
    terrain     slope_* and elev_*       (statistics of slope and elevation
                                          around the event)
    point ids   FID_1, Id
    record      OBJECTID ... Temporada   (the inventory record: date,
                                          description, second copy of year,
                                          month, municipality, type, ...)
    coordinates POINT_X, POINT_Y

The coordinates are the reference: a shift of +k for a block means that the
block's values that belong to the event at row i are found at row i + k.

Tests (each one uses different evidence, so they check each other):

1. terrain vs coordinates: the DEM elevation and slope at the row's point
   must fall inside the elev/slope MIN..MAX range of the row that holds its
   terrain statistics (with a tolerance).
2. attributes vs coordinates: the point must be closer to the centre of its
   own municipality than to the centre of the municipality written k rows
   away (centres = median point of each municipality).
3. record vs attributes: the fields that appear in both blocks (year, month,
   municipality, hillside, type, material, damage) must agree.
4. identifiers: OBJECTID - 1 - row and FID_1 - row along the file. They
   are only descriptive: an identifier with a gap is not by itself a shift.
5. the workmate's file (``Inventari_TFM_Martin_FP_cOMPLET.csv``, read only)
   repeats test 1 on it, and checks whether its rows pair the attributes
   with the same records as ours (if so, the misalignment came before it).
6. identifier join: OBJECTID stays on the event's line and FID_1 moves with
   the record, so record FID_1 = f belongs to the line with OBJECTID = f + 1.
   The join is compared with test 3's pairing and checked for
   contradictions.

Inputs: data/inventari_depurat_coordenades.csv, the DEM and slope rasters
of ``cost_distance`` in config.yaml, the workmate's file (skipped if
missing), and the latest run's final_gdf.csv (to mark the exported rows).
The result doesn't depend on a formatter run, so it writes outside runs/,
in ``analyses/inventory_shift_study/`` (repo root, gitignored):

    summary.json         settings and the headline numbers
    shift_shares.csv     per test, part of the file and shift k: share of rows
                         consistent with k
    row_shifts.csv       every line: its best shift in each test (blank if no
                         unique answer) and the evidence behind it
    segments.csv         runs of consecutive lines with the same shift, per
                         test
    rockfalls.csv        the rockfalls the formatter selects from the file,
                         with their shifts and whether they are exported
    orphan_records.csv   records that no line claims once the shifts are
                         applied (events that lost their point)
    realigned_rockfalls.csv  what-if: the rockfalls selected if each line
                         took the record its shift points to, vs now

How to read it: the share at k = 0 should be the highest wherever a block is
aligned. A clear peak at another k is a shift; no peak at all means the
block's rows don't belong to the points near it (scrambled or empty).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from data_formatter import inventory
from data_formatter.config import DEFAULT_CONFIG_FILE, Config
from data_formatter.inventory_alignment import TYPE_FAMILIES
from scripts.run_selection import add_run_argument, run_folder

OUTPUT_PATH = Path("analyses") / "inventory_shift_study"
WORKMATE_FILE = Path(
    "../00_TFM_Martin_Rizzo/TFM_Martin_Rizzo/Inventari_TFM_Martin_FP_cOMPLET.csv"
)

# Column blocks, by name of their first and last column in the file.
BLOCKS = {
    "attributes": ("Año", "GEO_SIMPL_code"),
    "terrain": ("slope_MIN", "elev_PCT90"),
    "point_ids": ("FID_1", "Id"),
    "record": ("OBJECTID", "Temporada"),
    "coordinates": ("POINT_X", "POINT_Y"),
}

# Test 1 tolerances. The terrain statistics come from a footprint around the
# event, probably on another DEM; ours is the 20 m DEM, so a point can fall
# a little outside the range even when the row is right.
ELEVATION_TOLERANCE_M = 30.0
SLOPE_TOLERANCE_DEG = 5.0

# Test 3: fields written in both blocks (attributes column, record column).
SHARED_FIELDS = [
    ("Año", "Año_1"),
    ("Mes", "Mes_1"),
    ("Municipio", "Municipio_1"),
    ("Vertiente", "Vertiente_1"),
    ("Tipo", "Tipo_1"),
    ("Material", "Material_1"),
    ("Daños", "Daños_1"),
]
# The attributes write the type in Catalan, the record in Spanish.
TYPE_TRANSLATION = {
    "esllavissades/despreniments": "slide_or_rockfall",
    "desprendimiento": "slide_or_rockfall",
    "deslizamiento": "slide_or_rockfall",
    "fluxe": "flow",
    "flujo": "flow",
    "grieta": "crack",
    "col·lapse": "collapse",
    "colapso": "collapse",
}
# A record row matches an attributes row if at least this many shared fields
# are filled in both and all of them agree.
MIN_FIELDS_COMPARED = 3

# Segmentation: cost of a change of shift, in lines of evidence. With the
# default window, 6 gives the same segments except lines 820-827 (8 lines:
# -10 with 3, merged into the -7 segment with 6) and two weak terrain bits
# past line 926, so the result isn't sensitive to it.
CHANGE_PENALTY = 3.0
# A segment with fewer tested lines than this is reported as weak: with
# ±110 shifts, a few wide terrain ranges can fit some far shift by chance.
MIN_SEGMENT_LINES = 10

# Lines matched by more record lines than this are listed as "many".
MAX_LISTED_MATCHES = 5

# Parts of the file reported separately (first line of each part). They
# come from the identifier steps of test 4 and the end of the record block;
# test results are also given per line, so the parts are only for reading.
PARTS = {"lines 2-827": 2, "lines 828-934": 828, "lines 935-end": 935}


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _read_raw(path: Path) -> pd.DataFrame:
    """Read an inventory CSV as text (no type guessing), with ``line``.

    Parameters
    ----------
    path : pathlib.Path
        A ``;``-separated file.

    Returns
    -------
    pandas.DataFrame
        All values as stripped strings ("" when empty), plus ``line``, the
        line number in the file (header = line 1), as in ``sample_id``.
    """
    df = pd.read_csv(
        path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    df = df.apply(lambda column: column.str.strip())
    df["line"] = np.arange(len(df)) + 2
    return df


def _number(column: pd.Series) -> np.ndarray:
    """Text with decimal comma (or point) to float; empty → NaN."""
    return pd.to_numeric(
        column.str.replace(",", ".", regex=False), errors="coerce"
    ).to_numpy(float)


def _sample(raster_path: Path, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Raster value at each point (NaN outside the raster or on NoData).

    The inventory is in EPSG:32631 and the rasters in EPSG:25831; both are
    UTM 31N and differ by well under a metre here, so no reprojection.
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    values = np.full(len(x), np.nan)
    with rasterio.open(raster_path) as raster:
        sampled = np.array(
            [v[0] for v in raster.sample(zip(x[valid], y[valid], strict=True))], float
        )
        if raster.nodata is not None:
            sampled[sampled == raster.nodata] = np.nan
    values[valid] = sampled
    return values


def _block_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """The columns of each block of ``BLOCKS``, in file order."""
    columns = list(df.columns)
    return {
        name: columns[columns.index(first) : columns.index(last) + 1]
        for name, (first, last) in BLOCKS.items()
    }


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def terrain_fits(df: pd.DataFrame, dem: Path, slope: Path) -> dict:
    """Test 1: which row's terrain statistics fit the DEM at each point.

    Parameters
    ----------
    df : pandas.DataFrame
        From ``_read_raw``.
    dem, slope : pathlib.Path
        Elevation (m) and slope (degrees) rasters.

    Returns
    -------
    dict
        ``fits(i, k)``: True if the DEM and slope at row i's point fall
        inside row i + k's ranges; ``tested``: rows with a point inside the
        rasters and terrain statistics of their own; ``dem``, ``slope``: the
        sampled values.
    """
    x, y = _number(df["POINT_X"]), _number(df["POINT_Y"])
    z, s = _sample(dem, x, y), _sample(slope, x, y)
    elev_min, elev_max = _number(df["elev_MIN"]), _number(df["elev_MAX"])
    slope_min, slope_max = _number(df["slope_MIN"]), _number(df["slope_MAX"])
    n = len(df)

    def fits(i: int, k: int) -> bool:
        j = i + k
        if not 0 <= j < n or np.isnan(elev_min[j]):
            return False
        inside_elevation = (
            elev_min[j] - ELEVATION_TOLERANCE_M
            <= z[i]
            <= elev_max[j] + ELEVATION_TOLERANCE_M
        )
        # Slope: checked only where both are known (the slope raster has
        # gaps at the coast).
        inside_slope = np.isnan(s[i]) or (
            slope_min[j] - SLOPE_TOLERANCE_DEG
            <= s[i]
            <= slope_max[j] + SLOPE_TOLERANCE_DEG
        )
        return bool(inside_elevation and inside_slope)

    tested = ~np.isnan(z) & ~np.isnan(elev_min)
    return {"fits": fits, "tested": tested, "dem": z, "slope": s}


def municipality_fits(df: pd.DataFrame) -> dict:
    """Test 2: is each point nearest to the municipality written k rows away?

    The centre of a municipality is the median point of the rows that name
    it. A row is informative only if the municipalities within the window
    are not all the same, so ``nearest`` is computed per row over a set of
    shifts and is NaN where the answer can't discriminate.

    Returns
    -------
    dict
        ``distance(i, k)``: km from row i's point to the centre of row
        i + k's municipality (NaN if unknown).
    """
    x, y = _number(df["POINT_X"]), _number(df["POINT_Y"])
    municipality = df["Municipio"].to_numpy()
    centres = (
        pd.DataFrame({"m": municipality, "x": x, "y": y})
        .query("m != ''")
        .groupby("m")[["x", "y"]]
        .median()
    )
    n = len(df)

    def distance(i: int, k: int) -> float:
        j = i + k
        if not 0 <= j < n or municipality[j] not in centres.index:
            return np.nan
        cx, cy = centres.loc[municipality[j]]
        return float(np.hypot(x[i] - cx, y[i] - cy) / 1000)

    return {"distance": distance, "municipality": municipality}


def _encode_shared_fields(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Shared fields of test 3 as integer codes, one array per block.

    Returns
    -------
    tuple of numpy.ndarray
        ``(attributes, record)``, each of shape (rows, fields); 0 = empty,
        equal codes = equal (normalised) values across the two blocks.
    """
    codes: dict[str, int] = {}

    def normalise(value: str, field: str) -> str:
        value = value.strip().lower()
        if field.startswith("Mes") and value.isdigit():
            return str(int(value))
        if field.startswith("Tipo"):
            return TYPE_TRANSLATION.get(value, value)
        return value

    def encode(column: pd.Series, field: str) -> np.ndarray:
        out = np.zeros(len(column), int)
        for i, value in enumerate(column):
            value = normalise(value, field)
            if value:
                out[i] = codes.setdefault(f"{field[:3]}|{value}", len(codes) + 1)
        return out

    attributes = np.column_stack(
        [encode(df[a], a) for a, _ in SHARED_FIELDS]
    )
    record = np.column_stack([encode(df[b], a) for a, b in SHARED_FIELDS])
    return attributes, record


def record_matches(df: pd.DataFrame) -> dict:
    """Test 3: which record rows agree with each attributes row.

    Returns
    -------
    dict
        ``compared`` and ``agree``: (rows × rows) arrays, the number of
        shared fields filled in both (attributes row i, record row j) and
        how many of them are equal; ``match``: compared ≥
        MIN_FIELDS_COMPARED and agree == compared.
    """
    attributes, record = _encode_shared_fields(df)
    both = (attributes[:, None, :] > 0) & (record[None, :, :] > 0)
    equal = both & (attributes[:, None, :] == record[None, :, :])
    compared, agree = both.sum(axis=2), equal.sum(axis=2)
    match = (compared >= MIN_FIELDS_COMPARED) & (agree == compared)
    return {"compared": compared, "agree": agree, "match": match}


# --------------------------------------------------------------------------
# Consistency matrices, per-line shifts and summaries
# --------------------------------------------------------------------------
#
# Each test gives a matrix C (lines × shifts): C[i, k] is how consistent
# shift k is with line i (1 = consistent, 0 = not, a fraction for the ties of
# test 2), NaN where the line can't be tested. A line alone is often
# consistent with several shifts (wide elevation ranges, common months or
# municipalities), so the shift of a part of the file is read from many
# lines together: the column means of C over a part (shift_shares.csv) or
# over a moving window of lines (the local shift of row_shifts.csv and the
# runs of segments.csv).


def consistency_matrices(
    df: pd.DataFrame, terrain: dict, municipality: dict, record: dict,
    shifts: list[int],
) -> dict[str, np.ndarray]:
    """One consistency matrix per test (see the comment above).

    - terrain: 1 if line i's point fits line i + k's terrain ranges;
    - attributes: among the shifts whose municipality centre is nearest to
      line i's point, 1 / (number of tied shifts); NaN if the window has a
      single municipality (nothing to tell apart);
    - record: 1 if attributes line i matches record line i + k; NaN if line
      i has fewer than MIN_FIELDS_COMPARED shared fields filled against
      every record line (it can't match anything).
    """
    n, m = len(df), len(shifts)
    terrain_c = np.full((n, m), np.nan)
    attributes_c = np.full((n, m), np.nan)
    record_c = np.full((n, m), np.nan)
    filled = record["compared"].max(axis=1) >= MIN_FIELDS_COMPARED
    for i in range(n):
        if terrain["tested"][i]:
            terrain_c[i] = [terrain["fits"](i, k) for k in shifts]

        d = np.array([municipality["distance"](i, k) for k in shifts])
        names = {
            municipality["municipality"][i + k]
            for k, v in zip(shifts, d, strict=True) if not np.isnan(v)
        }
        if len(names) > 1:
            winners = d == np.nanmin(d)
            attributes_c[i] = np.where(np.isnan(d), np.nan, winners / winners.sum())

        if filled[i]:
            record_c[i] = [
                record["match"][i, i + k] if 0 <= i + k < n else np.nan
                for k in shifts
            ]
    return {"terrain": terrain_c, "attributes": attributes_c, "record": record_c}


def _part_of(line: int) -> str:
    """The part of ``PARTS`` that a line belongs to."""
    return [name for name, first in PARTS.items() if line >= first][-1]


def shift_shares(
    df: pd.DataFrame, matrices: dict[str, np.ndarray], shifts: list[int]
) -> pd.DataFrame:
    """Mean consistency of each shift, per test and part of the file.

    ``rows`` is the number of lines of the part that the test could use.
    """
    parts = np.array([_part_of(line) for line in df["line"]])
    out = []
    for test, c in matrices.items():
        for part in PARTS:
            rows = c[parts == part]
            used = ~np.all(np.isnan(rows), axis=1)
            for j, k in enumerate(shifts):
                share = np.nanmean(rows[used, j]) if used.any() else np.nan
                out.append((test, part, k, int(used.sum()), share))
    return pd.DataFrame(out, columns=["test", "part", "shift", "rows", "share"])


def segment_shifts(c: np.ndarray, shifts: list[int], penalty: float) -> np.ndarray:
    """Piecewise-constant shift that best explains a consistency matrix.

    Chooses one shift per line, k_1 .. k_n, to maximise

        sum_i C[i, k_i]  -  penalty × (number of lines where k changes),

    solved exactly by dynamic programming (the Viterbi algorithm over the
    shifts as states). NaN counts as 0 for every shift, so lines that can't
    be tested, or that fit every shift, don't pull towards any shift; a
    change is only made where more than ``penalty`` lines' worth of
    consistency supports it. This is the usual penalised change-point
    formulation, used here as a diagnostic, not as a method of the thesis.

    Parameters
    ----------
    c : numpy.ndarray
        A consistency matrix (lines × shifts).
    shifts : list of int
        The shifts of its columns.
    penalty : float
        Cost of each change of shift, in lines.

    Returns
    -------
    numpy.ndarray
        The shift of each line.
    """
    score = np.nan_to_num(c, nan=0.0)
    n, m = score.shape
    best = score[0].copy()
    came_from = np.zeros((n, m), int)
    for i in range(1, n):
        # Either keep the previous line's shift or jump from the best one.
        jump = best.max() - penalty
        came_from[i] = np.where(best >= jump, np.arange(m), best.argmax())
        best = np.maximum(best, jump) + score[i]
    path = np.zeros(n, int)
    path[-1] = best.argmax()
    for i in range(n - 1, 0, -1):
        path[i - 1] = came_from[i, path[i]]
    return np.asarray(shifts)[path]


def row_shifts(
    df: pd.DataFrame, matrices: dict[str, np.ndarray], shifts: list[int],
    terrain: dict, record: dict, penalty: float,
) -> pd.DataFrame:
    """Every line: the segmented shift of each test and the evidence.

    Columns per test ``<test>``: ``<test>_shift`` (from
    ``segment_shifts``), ``<test>_fit`` (C at that shift: is the line
    consistent with it?), ``<test>_fit_at_0`` and
    ``<test>_consistent_shifts``: the shifts consistent with the line alone.
    Plus ``terrain_present`` / ``record_present`` (the line has values in
    that block), the DEM value, the terrain range, the identifier offsets
    (test 4, descriptive) and a few columns to read the line.
    """
    n = len(df)
    zero = shifts.index(0)
    out = pd.DataFrame({"line": df["line"].to_numpy()})
    out["terrain_present"] = (df["elev_MIN"] != "").to_numpy()
    out["record_present"] = (df["OBJECTID"] != "").to_numpy()
    for test, c in matrices.items():
        best = segment_shifts(c, shifts, penalty)
        columns = np.searchsorted(shifts, best)
        out[f"{test}_shift"] = best
        out[f"{test}_fit"] = c[np.arange(n), columns]
        out[f"{test}_fit_at_0"] = c[:, zero]
        out[f"{test}_consistent_shifts"] = [
            " ".join(str(k) for k, v in zip(shifts, row, strict=True) if v > 0)
            for row in c
        ]
    # The record of the event can be anywhere in the file: global matches.
    out["record_matches_anywhere"] = [
        " ".join(str(j - i) for j in np.flatnonzero(record["match"][i]))
        if (record["match"][i].sum() <= MAX_LISTED_MATCHES) else "many"
        for i in range(n)
    ]
    out["dem_m"] = terrain["dem"].round(1)
    out["slope_deg"] = terrain["slope"].round(1)
    out["elev_MIN"], out["elev_MAX"] = df["elev_MIN"], df["elev_MAX"]
    index = np.arange(n)
    out["objectid_minus_row"] = _number(df["OBJECTID"]) - 1 - index
    out["fid_minus_row"] = _number(df["FID_1"]) - index
    for column in ["Municipio", "Municipio_1", "Tipo", "Fecha", "localizaci"]:
        out[column] = df[column].to_numpy()
    return out


def segments(shifts: pd.DataFrame, test: str) -> pd.DataFrame:
    """Runs of consecutive lines with the same segmented shift, for one test.

    A run is also split where the block itself starts or stops having
    values (the terrain block ends at line 940, the record block at 934),
    so a run past that point says "nothing to align", not a shift.
    ``fit`` and ``fit_at_0``: mean consistency of the run's tested lines at
    its shift and at 0.
    """
    block = "record" if test in ("record", "attributes") else "terrain"
    present = shifts[f"{block}_present"] if test != "attributes" else True
    key = shifts[f"{test}_shift"].astype(str) + "|" + pd.Series(present, index=shifts.index).astype(str)
    run = key.ne(key.shift()).cumsum()
    tested = shifts[f"{test}_fit"].notna()
    grouped = shifts.assign(_tested=tested, _present=present).groupby(run).agg(
        first_line=("line", "first"),
        last_line=("line", "last"),
        shift=(f"{test}_shift", "first"),
        block_has_values=("_present", "first"),
        lines=("line", "size"),
        tested_lines=("_tested", "sum"),
        fit=(f"{test}_fit", "mean"),
        fit_at_0=(f"{test}_fit_at_0", "mean"),
    )
    grouped.insert(0, "test", test)
    return grouped.reset_index(drop=True).round({"fit": 3, "fit_at_0": 3})


def workmate_check(ours: pd.DataFrame, dem: Path, slope: Path) -> dict:
    """Test 5: the same terrain test on the workmate's file, and its pairing.

    Returns
    -------
    dict
        Terrain shares at k = -1..2 for its rockfall rows and for its
        random points, and how many of its rockfalls pair the same
        attributes with the same record (by OBJECTID) as our file.
    """
    if not WORKMATE_FILE.exists():
        return {"skipped": f"{WORKMATE_FILE} not found"}
    theirs = _read_raw(WORKMATE_FILE)
    terrain = terrain_fits(theirs, dem, slope)
    rockfall = theirs["Tipo"].to_numpy() != ""
    result: dict = {"rows": len(theirs), "rockfall_rows": int(rockfall.sum())}
    # Its rockfalls and random points are two layers stacked one after the
    # other, so each is tested on its own.
    for label, mask in [("rockfalls", rockfall), ("random_points", ~rockfall)]:
        tested = np.flatnonzero(mask & terrain["tested"])
        result[f"terrain_share_{label}"] = {
            k: round(float(np.mean([terrain["fits"](i, k) for i in tested])), 3)
            for k in (-1, 0, 1, 2)
        }

    # Same record (OBJECTID) → same attributes in both files?
    attribute_columns = ["Municipio", "Vertiente", "Material", "Daños", "DURESA"]
    ours_by_id = ours[ours["OBJECTID"] != ""].copy()
    ours_by_id["oid"] = _number(ours_by_id["OBJECTID"])
    theirs_by_id = theirs[theirs["OBJECTID"] != ""].copy()
    theirs_by_id["oid"] = _number(theirs_by_id["OBJECTID"])
    both = theirs_by_id.merge(ours_by_id, on="oid", suffixes=("_w", "_o"))
    same = np.all(
        [both[f"{c}_w"] == both[f"{c}_o"] for c in attribute_columns], axis=0
    )
    result["rockfalls_found_by_objectid"] = len(both)
    result["same_attributes_as_ours"] = int(same.sum())
    result["their_attribute_record_mismatches"] = int(
        (theirs_by_id["Municipio"] != theirs_by_id["Municipio_1"]).sum()
    )
    return result


def orphan_records(
    df: pd.DataFrame, shifts: pd.DataFrame, record: dict
) -> pd.DataFrame:
    """Records that no line claims once the record shifts are applied.

    Line i's record is taken to be at line i + record_shift. A record line
    that no line points to has lost its attributes and coordinates (its
    event's point is elsewhere in the file, or missing). ``matches_anywhere``
    lists the attributes lines that agree with it on the shared fields.
    """
    present = np.flatnonzero(df["OBJECTID"].to_numpy() != "")
    claimed = set(np.arange(len(df)) + shifts["record_shift"].to_numpy())
    rows = [j for j in present if j not in claimed]
    out = df.iloc[rows][
        ["line", "OBJECTID", "Fecha", "Año_1", "Mes_1", "Municipio_1",
         "Tipo_1", "localizaci"]
    ].copy()
    dates = pd.to_datetime(
        out["Fecha"], format=inventory.INVENTORY_DATE_FORMAT, errors="coerce"
    )
    out["exact_date_in_period"] = dates.between(
        inventory.PERIOD_START, inventory.PERIOD_END
    ).to_numpy()
    out["matches_anywhere"] = [
        " ".join(str(int(df["line"].iloc[i])) for i in np.flatnonzero(record["match"][:, j]))
        if record["match"][:, j].sum() <= MAX_LISTED_MATCHES else "many"
        for j in rows
    ]
    return out


def _contradicting_fields(df: pd.DataFrame, i: int, j: int) -> list[str]:
    """Shared fields where attributes line i and record line j disagree.

    Only fields filled in both count. The type is compared by family
    (``TYPE_FAMILIES``): two types agree if they share one.
    """

    def families(value: str) -> set[str]:
        value = value.lower()
        return {family for key, family in TYPE_FAMILIES.items() if key in value}

    fields = []
    for attribute, record_column in SHARED_FIELDS:
        mine, theirs = df[attribute].iloc[i], df[record_column].iloc[j]
        if not mine or not theirs:
            continue
        if attribute == "Tipo":
            differ = not families(mine) & families(theirs)
        elif attribute == "Mes":
            differ = mine.isdigit() and theirs.isdigit() and int(mine) != int(theirs)
        else:
            differ = mine.lower() != theirs.lower()
        if differ:
            fields.append(attribute)
    return fields


def fid_join_check(
    df: pd.DataFrame, shifts: pd.DataFrame, record: dict
) -> dict:
    """Test 6: does an identifier join rebuild the event-record pairing?

    ``OBJECTID`` stays on the event's line, while ``FID_1`` moves with the
    rest of the record: a record with ``FID_1`` = f belongs to the event line
    whose ``OBJECTID`` = f + 1 (the workmate's file has OBJECTID = FID_1 + 1
    on every row). The join uses no value read from the content. It is
    compared with the pairing found from the content (test 3), and every
    pair is checked for contradictions in the shared fields.

    Returns
    -------
    dict
        Events paired; share paired with the same record as test 3 (per
        part of the file); contradicting pairs per shared field (type
        compared by family: slide/rockfall, flow, crack, collapse,
        subsidence; mixed types like "flujo y desprendimiento" count as
        both); the record lines no event takes; and the rows deleted from
        the event lines, read from the steps of OBJECTID - 1 - row.
    """
    n = len(df)
    line = df["line"].to_numpy()
    fid = _number(df["FID_1"])
    objectid = _number(df["OBJECTID"])
    record_line_of = {int(f): j for j, f in enumerate(fid) if not np.isnan(f)}
    target = np.full(n, -1)
    for i in np.flatnonzero(~np.isnan(objectid)):
        target[i] = record_line_of.get(int(objectid[i]) - 1, -1)
    paired = np.flatnonzero(target >= 0)
    study = np.arange(n) + shifts["record_shift"].to_numpy()

    contradictions = dict.fromkeys([a for a, _ in SHARED_FIELDS], 0)
    contradicting_lines = []
    for i in paired:
        fields = _contradicting_fields(df, i, target[i])
        for field in fields:
            contradictions[field] += 1
        if fields:
            contradicting_lines.append(int(line[i]))

    offset = objectid - 1 - np.arange(n)
    known = ~np.isnan(offset)
    steps = np.flatnonzero(known[1:] & known[:-1] & (np.diff(offset) != 0)) + 1
    result: dict = {
        "events_paired": len(paired),
        "same_record_as_content_study": {},
        "contradicting_pairs_per_field": contradictions,
        "contradicting_event_lines": contradicting_lines,
        "deleted_event_rows": {
            int(line[k]): int(offset[k] - offset[k - 1]) for k in steps
        },
        "record_lines_with_no_event": sorted(
            int(line[j]) for f, j in record_line_of.items()
            if j not in set(target[paired].tolist())
        ),
    }
    for first, last in [(2, 809), (810, 827), (828, 881), (882, 886), (887, 934)]:
        events = paired[(line[paired] >= first) & (line[paired] <= last)]
        result["same_record_as_content_study"][f"events {first}-{last}"] = round(
            float(np.mean(target[events] == study[events])), 3
        )
    return result


def selected_rockfalls(
    df: pd.DataFrame, shifts: pd.DataFrame, exported: set[str]
) -> pd.DataFrame:
    """The rockfalls the formatter takes from the file, with their shifts.

    Same selection as ``inventory.select_events`` (type, exact date in the
    study period, required values present), on the raw text.
    """
    dates = pd.to_datetime(
        df["Fecha"], format=inventory.INVENTORY_DATE_FORMAT, errors="coerce"
    )
    keep = (
        (df["Tipo"] == inventory.EVENT_TYPE)
        & dates.between(inventory.PERIOD_START, inventory.PERIOD_END)
        & (df[inventory.REQUIRED_COLUMNS] != "").all(axis=1)
    )
    out = shifts[keep.to_numpy()].copy()
    out.insert(0, "sample_id", "inv_" + out["line"].astype(str))
    out.insert(1, "exported", out["sample_id"].isin(exported))
    return out


def realigned_rockfalls(
    df: pd.DataFrame, shifts: pd.DataFrame, exported: set[str]
) -> pd.DataFrame:
    """The rockfall selection if each line took the record its shift points to.

    For every line i (attributes and coordinates), the date is read from
    line i + record_shift. Rows are the union of the current selection and
    the realigned one, with ``change``: ``same`` (selected in both, same
    date), ``new_date``, ``dropped`` (no exact date in the period once
    realigned) or ``added`` (only selected once realigned). What-if only:
    the formatter doesn't do this, and the segments it relies on must be
    confirmed first.
    """
    n = len(df)
    target = np.arange(n) + shifts["record_shift"].to_numpy()
    valid = (target >= 0) & (target < n)
    realigned_fecha = np.where(
        valid, df["Fecha"].to_numpy()[np.clip(target, 0, n - 1)], ""
    )

    def selected(fecha: np.ndarray) -> np.ndarray:
        dates = pd.to_datetime(
            pd.Series(fecha), format=inventory.INVENTORY_DATE_FORMAT, errors="coerce"
        )
        location = [c for c in inventory.REQUIRED_COLUMNS if c != "Fecha"]
        return (
            (df["Tipo"] == inventory.EVENT_TYPE).to_numpy()
            & dates.between(inventory.PERIOD_START, inventory.PERIOD_END).to_numpy()
            & (df[location] != "").all(axis=1).to_numpy()
        )

    now = selected(df["Fecha"].to_numpy())
    then = selected(realigned_fecha)
    keep = now | then
    out = pd.DataFrame({
        "sample_id": "inv_" + df["line"].astype(str),
        "exported": ("inv_" + df["line"].astype(str)).isin(exported),
        "record_shift": shifts["record_shift"],
        "date_now": np.where(now, df["Fecha"], ""),
        "date_realigned": np.where(then, realigned_fecha, ""),
        "Municipio": df["Municipio"],
        "localizaci_realigned": np.where(
            valid, df["localizaci"].to_numpy()[np.clip(target, 0, n - 1)], ""
        ),
    })[keep]
    out["change"] = np.select(
        [
            out["date_now"].ne("") & out["date_now"].eq(out["date_realigned"]),
            out["date_now"].ne("") & out["date_realigned"].ne(""),
            out["date_now"].ne(""),
        ],
        ["same", "new_date", "dropped"],
        default="added",
    )
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    """Run the five tests and write the outputs (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--window", type=int, default=110,
        help="shifts tried: -window .. +window (default 110: the largest "
        "shift found is 100)",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_FILE)
    add_run_argument(parser)
    args = parser.parse_args()

    config = Config.from_yaml(Path(args.config))
    data_path = config.data_path
    dem = data_path / config.cost_distance.dem
    slope = data_path / config.cost_distance.cost
    df = _read_raw(data_path / inventory.INVENTORY_FILE)
    blocks = _block_columns(df)

    terrain = terrain_fits(df, dem, slope)
    municipality = municipality_fits(df)
    record = record_matches(df)
    shift_values = list(range(-args.window, args.window + 1))
    matrices = consistency_matrices(df, terrain, municipality, record, shift_values)
    shares = shift_shares(df, matrices, shift_values)
    shifts = row_shifts(df, matrices, shift_values, terrain, record, CHANGE_PENALTY)
    runs = pd.concat([segments(shifts, test) for test in matrices], ignore_index=True)

    final = run_folder(args.run) / "outputs" / "final_gdf.csv"
    exported = set(pd.read_csv(final, usecols=["sample_id"])["sample_id"])
    rockfalls = selected_rockfalls(df, shifts, exported)
    orphans = orphan_records(df, shifts, record)
    present_records = set(df.loc[df["OBJECTID"] != "", "line"])
    realigned = realigned_rockfalls(df, shifts, exported)

    def peak(test: str, part: str) -> dict:
        rows = shares.query("test == @test and part == @part")
        top = rows.sort_values("share", ascending=False).head(3)
        return {
            "rows": int(rows["rows"].iloc[0]),
            "share_at_0": round(float(rows.query("shift == 0")["share"].iloc[0]), 3),
            "top_shifts": {int(k): round(float(v), 3) for k, v in zip(top["shift"], top["share"], strict=True)},
        }

    summary = {
        "inventory": str(data_path / inventory.INVENTORY_FILE),
        "lines": [int(df["line"].min()), int(df["line"].max())],
        "blocks": {name: [cols[0], cols[-1], len(cols)] for name, cols in blocks.items()},
        "last_line_with_record": int(df.loc[df["OBJECTID"] != "", "line"].max()),
        "last_line_with_terrain": int(df.loc[df["elev_MIN"] != "", "line"].max()),
        "settings": {
            "window": args.window,
            "elevation_tolerance_m": ELEVATION_TOLERANCE_M,
            "slope_tolerance_deg": SLOPE_TOLERANCE_DEG,
            "min_fields_compared": MIN_FIELDS_COMPARED,
            "change_penalty": CHANGE_PENALTY,
        },
        "peaks": {
            test: {part: peak(test, part) for part in PARTS}
            for test in ["terrain", "attributes", "record"]
        },
        "workmate_file": workmate_check(df, dem, slope),
        "fid_join": fid_join_check(df, shifts, record),
        "selected_rockfalls": len(rockfalls),
        "exported_rockfalls": int(rockfalls["exported"].sum()),
        "exported_rockfalls_by_shift": {
            test: {
                str(k): int(v)
                for k, v in rockfalls.loc[rockfalls["exported"], f"{test}_shift"]
                .value_counts().items()
            }
            for test in ("terrain", "record")
        },
        "records_claimed_by_several_lines": int(
            (lambda t: t[t.isin(present_records)].duplicated(keep=False).sum())(
                shifts["line"] + shifts["record_shift"]
            )
        ),
        "lines_whose_record_line_is_empty": int(
            (~(shifts["line"] + shifts["record_shift"]).isin(present_records)
             & (df["Municipio"] != "").to_numpy()).sum()
        ),
        "realigned_selection": {
            str(k): int(v) for k, v in realigned["change"].value_counts().items()
        },
        "orphan_records": len(orphans),
        "orphan_records_dated_in_period": int(orphans["exact_date_in_period"].sum()),
        "segments": {
            test: [
                f"{r.first_line}-{r.last_line}: "
                + (f"{int(r.shift):+d} (fit {r.fit:.2f}, at 0 {r.fit_at_0:.2f}, "
                   f"{r.tested_lines} tested"
                   + (", weak)" if r.tested_lines < MIN_SEGMENT_LINES else ")")
                   if r.block_has_values else "block empty")
                for r in runs.query("test == @test").itertuples()
            ]
            # Test 2 can't place a change of shift (neighbouring lines often
            # share a municipality); it is read per part, in "peaks".
            for test in ("terrain", "record")
        },
    }

    out = OUTPUT_PATH
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    shares.to_csv(out / "shift_shares.csv", index=False)
    shifts.to_csv(out / "row_shifts.csv", index=False)
    runs.to_csv(out / "segments.csv", index=False)
    rockfalls.to_csv(out / "rockfalls.csv", index=False)
    orphans.to_csv(out / "orphan_records.csv", index=False)
    realigned.to_csv(out / "realigned_rockfalls.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
