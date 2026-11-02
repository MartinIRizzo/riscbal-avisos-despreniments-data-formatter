"""Offline gauge replacement study: -4 days as not measured, correlation check.

Diagnostic only (no database, doesn't change the pipeline). It reproduces the
study of 2026-11-02 (vault note "Gauge replacement study (2026-11-02)",
decision D-021): which samples aren't covered by their own gauge once the
days of -4 runs count as not measured, which replacement gauge passes the
correlation rule, and which samples would be excluded. The twin step here
is the one of that study (any gauge <= 1 km from the own gauge); the final
rule (same code +/- a suffix) is implemented in the formatter.

Run from the repo root::

    uv run python -m scripts.gauge_study [--run <run_id>]

Inputs: data/historical_aemet_data.csv, data/stations.csv and the
``final_gdf.csv`` of a formatter run (default: the latest; see
run_selection.py). Each sample's own gauge comes from its ``id_station``,
so the fallback gauges of that run don't matter. Outputs, in
``runs/<run_id>/analyses/gauge_study/``: twins_B022_B001.csv,
replacement_study.csv.

Definitions:

- real daily reading: a day with a value that is not part of a -4 run (the
  -4 days plus the positive day that closes the run);
- covered window: >= 81 of the 90 days before the event are real readings,
  and the event day itself is one;
- correlation of two gauges: Pearson r of daily rain over the days both have
  a real reading in 2005-2015, with the number of common days; Spearman r
  and the ratio of totals for information.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from data_formatter.aemet_formatter import format_aemet_data
from scripts.run_selection import add_run_argument, analysis_folder, run_folder

DATA_PATH = Path("data")
PERIOD = ("2005-01-01", "2015-12-31")
WINDOW_DAYS = 90
MIN_COVERED_DAYS = 81
MAX_DISTANCE_KM = 5.0
TWIN_DISTANCE_KM = 1.0
NEIGHBOUR_SEARCH_KM = 12.0
MIN_R = 0.9
MIN_COMMON_DAYS = 365
# Fewer common days than this: no correlation computed at all.
MIN_DAYS_FOR_ANY_R = 30
METRES_PER_KM = 1000
# Sorts candidates without an assessable r below every real r.
NO_R_SORT_KEY = -9
GEOGRAPHIC_CRS = "EPSG:4326"
UTM_CRS = "EPSG:32631"
GAUGES_WITHOUT_TWIN_CHECK = ["B022", "B001"]
# Hand-made reassignments of the formatter (config.yaml), RiscBal gauges.
RISCBAL_REASSIGNMENTS = [
    ("Esporles", "2024-03-25", "2025-03-25", "GRO08"),
    ("Esporles", "2026-01-20", "2026-01-21", "BAR04"),
]
RISCBAL_STATIONS = ["GRO08", "BAR04"]


@dataclass(frozen=True)
class Correlation:
    """Agreement of two gauges' daily rain over their common real readings.

    Attributes
    ----------
    common_days : int
        Days both gauges have a real reading in the period.
    pearson : float
        Pearson r (NaN if too few common days).
    spearman : float
        Spearman r (NaN if too few common days).
    ratio_of_totals : float
        Total rain of the second gauge divided by that of the first.
    """

    common_days: int
    pearson: float
    spearman: float
    ratio_of_totals: float


@dataclass(frozen=True)
class Candidate:
    """A possible replacement gauge for one sample.

    Attributes
    ----------
    code : str
        Gauge code.
    distance_km : float
        Straight-line distance to the sample.
    correlation : Correlation
        Agreement with the sample's own gauge.
    """

    code: str
    distance_km: float
    correlation: Correlation

    @property
    def passes(self) -> bool:
        """Whether the correlation rule accepts this gauge.

        Returns
        -------
        bool
            At least ``MIN_COMMON_DAYS`` common days and r >= ``MIN_R``.
        """
        return (
            self.correlation.common_days >= MIN_COMMON_DAYS
            and self.correlation.pearson >= MIN_R
        )


def real_readings(aemet: pd.DataFrame) -> dict[str, pd.Series]:
    """Daily rain of every gauge, real readings only.

    Parameters
    ----------
    aemet : pandas.DataFrame
        From ``format_aemet_data``.

    Returns
    -------
    dict of str to pandas.Series
        For each gauge code, rain in mm indexed by date, without the days
        of -4 runs.
    """
    real = aemet[~aemet["accumulated"]]
    return {
        code: group.set_index("date")["precipitation"]
        for code, group in real.groupby("code")
    }


def gauge_coordinates(
    aemet: pd.DataFrame, stations: pd.DataFrame
) -> pd.DataFrame:
    """Name and UTM coordinates of every gauge in the AEMET file.

    Parameters
    ----------
    aemet : pandas.DataFrame
        From ``format_aemet_data`` (decimal ``lon`` and ``lat``).
    stations : pandas.DataFrame
        ``stations.csv``.

    Returns
    -------
    pandas.DataFrame
        Indexed by gauge code: ``name``, ``x``, ``y`` (UTM 31N metres) and
        ``in_stations_csv``.
    """
    first_rows = aemet.drop_duplicates("code").set_index("code")
    to_utm = Transformer.from_crs(GEOGRAPHIC_CRS, UTM_CRS, always_xy=True)
    gauges = pd.DataFrame(
        {
            "name": first_rows["NOMBRE"]
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        }
    )
    gauges["x"], gauges["y"] = to_utm.transform(
        first_rows["lon"].to_numpy(), first_rows["lat"].to_numpy()
    )
    gauges["in_stations_csv"] = gauges.index.isin(stations["code"])
    return gauges


def coverage(
    readings: dict[str, pd.Series], code: str, date: pd.Timestamp
) -> tuple[bool, int]:
    """Whether a gauge covers a sample's window, and its antecedent days.

    Parameters
    ----------
    readings : dict of str to pandas.Series
        From ``real_readings``.
    code : str
        Gauge code.
    date : pandas.Timestamp
        Event day of the sample.

    Returns
    -------
    covered : bool
        At least ``MIN_COVERED_DAYS`` real readings in the 90 days before,
        and a real reading on the event day.
    antecedent_days : int
        Real readings in the 90 days before the event day.
    """
    series = readings.get(code)
    if series is None:
        return False, 0
    window = pd.date_range(
        date - pd.Timedelta(days=WINDOW_DAYS), date - pd.Timedelta(days=1)
    )
    antecedent_days = int(series.index.isin(window).sum())
    covered = antecedent_days >= MIN_COVERED_DAYS and date in series.index
    return covered, antecedent_days


def correlate(
    readings: dict[str, pd.Series], first_code: str, second_code: str
) -> Correlation:
    """Correlation of two gauges' daily rain, 2005-2015.

    Parameters
    ----------
    readings : dict of str to pandas.Series
        From ``real_readings``.
    first_code, second_code : str
        Gauge codes; the ratio of totals is second / first.

    Returns
    -------
    Correlation
        r is NaN with fewer than ``MIN_DAYS_FOR_ANY_R`` common days.
    """
    first, second = readings.get(first_code), readings.get(second_code)
    if first is None or second is None:
        return Correlation(0, np.nan, np.nan, np.nan)
    common = pd.concat([first, second], axis=1, join="inner").loc[
        PERIOD[0] : PERIOD[1]
    ]
    if len(common) < MIN_DAYS_FOR_ANY_R:
        return Correlation(len(common), np.nan, np.nan, np.nan)
    first_rain, second_rain = common.iloc[:, 0], common.iloc[:, 1]
    first_total = first_rain.sum()
    return Correlation(
        common_days=len(common),
        pearson=first_rain.corr(second_rain),
        spearman=first_rain.rank().corr(second_rain.rank()),
        ratio_of_totals=second_rain.sum() / first_total
        if first_total
        else np.nan,
    )


def neighbour_table(
    readings: dict[str, pd.Series], gauges: pd.DataFrame
) -> pd.DataFrame:
    """Neighbours of the gauges in ``GAUGES_WITHOUT_TWIN_CHECK``.

    Parameters
    ----------
    readings : dict of str to pandas.Series
        From ``real_readings``.
    gauges : pandas.DataFrame
        From ``gauge_coordinates``.

    Returns
    -------
    pandas.DataFrame
        One row per neighbour within ``NEIGHBOUR_SEARCH_KM``: distance,
        record, common days and correlation.
    """
    rows = []
    for target in GAUGES_WITHOUT_TWIN_CHECK:
        distance_km = (
            np.hypot(
                gauges["x"] - gauges.loc[target, "x"],
                gauges["y"] - gauges.loc[target, "y"],
            )
            / METRES_PER_KM
        )
        near = distance_km[
            (distance_km <= NEIGHBOUR_SEARCH_KM)
            & (distance_km.index != target)
        ]
        for code in near.sort_values().index:
            correlation = correlate(readings, target, code)
            series = readings.get(code)
            rows.append(
                {
                    "gauge": target,
                    "neighbour": code,
                    "name": gauges.loc[code, "name"],
                    "distance_km": round(distance_km[code], 2),
                    "in_stations_csv": gauges.loc[code, "in_stations_csv"],
                    "first": series.index.min().date()
                    if series is not None
                    else None,
                    "last": series.index.max().date()
                    if series is not None
                    else None,
                    "common_days_2005_2015": correlation.common_days,
                    "pearson_r": round(correlation.pearson, 3),
                    "spearman_r": round(correlation.spearman, 3),
                    "ratio_of_totals": round(correlation.ratio_of_totals, 2),
                }
            )
    return pd.DataFrame(rows)


def samples_with_gauges(
    stations: pd.DataFrame, final_gdf: Path
) -> pd.DataFrame:
    """The AEMET samples of the export, with their own gauge code.

    Parameters
    ----------
    stations : pandas.DataFrame
        ``stations.csv`` (``id`` -> ``code``).
    final_gdf : pathlib.Path
        The run's ``final_gdf.csv``.

    Returns
    -------
    pandas.DataFrame
        The rows of the export whose own gauge is in the AEMET file, with
        ``code`` (from ``id_station``, not the fallback gauge).
    """
    samples = pd.read_csv(final_gdf, parse_dates=["date"])
    samples["code"] = samples["id_station"].map(
        stations.set_index("id")["code"]
    )
    for municipality, after, until, gauge in RISCBAL_REASSIGNMENTS:
        samples.loc[
            (samples["municipality"] == municipality)
            & (samples["date"] > after)
            & (samples["date"] <= until),
            "code",
        ] = gauge
    return samples[~samples["code"].isin(RISCBAL_STATIONS)].copy()


def _distance_km(
    station_xy: pd.DataFrame, code: str, x: float, y: float
) -> float:
    """Straight-line distance from a stations-list gauge to a point.

    Parameters
    ----------
    station_xy : pandas.DataFrame
        ``stations.csv`` coordinates indexed by code.
    code : str
        Gauge code.
    x, y : float
        The point, UTM 31N metres.

    Returns
    -------
    float
        Distance in km.
    """
    return (
        np.hypot(
            station_xy.loc[code, "utm_x"] - x,
            station_xy.loc[code, "utm_y"] - y,
        )
        / METRES_PER_KM
    )


def _round_or_none(value: float | None, decimals: int) -> float | None:
    """Round a value, turning None and NaN into None (an empty CSV cell).

    Parameters
    ----------
    value : float or None
        The value.
    decimals : int
        Decimals to keep.

    Returns
    -------
    float or None
        The rounded value, or None.
    """
    if value is None or np.isnan(value):
        return None
    return round(value, decimals)


def _nearby_candidates(
    sample: pd.Series,
    readings: dict[str, pd.Series],
    station_xy: pd.DataFrame,
) -> list[Candidate]:
    """Stations-list gauges <= MAX_DISTANCE_KM that cover the window.

    Parameters
    ----------
    sample : pandas.Series
        One row of ``samples_with_gauges``.
    readings : dict of str to pandas.Series
        From ``real_readings``.
    station_xy : pandas.DataFrame
        ``stations.csv`` coordinates indexed by code.

    Returns
    -------
    list of Candidate
        Sorted by distance to the sample, nearest first.
    """
    candidates = []
    for code in station_xy.index:
        if code == sample["code"] or code not in readings:
            continue
        distance = _distance_km(
            station_xy, code, sample["POINT_X"], sample["POINT_Y"]
        )
        if (
            distance <= MAX_DISTANCE_KM
            and coverage(readings, code, sample["date"])[0]
        ):
            candidates.append(
                Candidate(
                    code, distance, correlate(readings, sample["code"], code)
                )
            )
    return sorted(candidates, key=lambda candidate: candidate.distance_km)


def _twin(
    sample: pd.Series,
    readings: dict[str, pd.Series],
    station_xy: pd.DataFrame,
) -> Candidate | None:
    """A twin of the sample's own gauge that covers the window and passes.

    Twin, as in the 2026-11-02 study: a stations-list gauge <= 1 km from
    the own gauge, at any distance from the sample.

    Parameters
    ----------
    sample : pandas.Series
        One row of ``samples_with_gauges``.
    readings : dict of str to pandas.Series
        From ``real_readings``.
    station_xy : pandas.DataFrame
        ``stations.csv`` coordinates indexed by code.

    Returns
    -------
    Candidate or None
        The last such gauge in the stations list, or None.
    """
    if sample["code"] not in station_xy.index:
        return None
    own_x, own_y = station_xy.loc[sample["code"], ["utm_x", "utm_y"]]
    twin = None
    for code in station_xy.index:
        if code == sample["code"] or code not in readings:
            continue
        if _distance_km(station_xy, code, own_x, own_y) > TWIN_DISTANCE_KM:
            continue
        candidate = Candidate(
            code,
            _distance_km(
                station_xy, code, sample["POINT_X"], sample["POINT_Y"]
            ),
            correlate(readings, sample["code"], code),
        )
        if coverage(readings, code, sample["date"])[0] and candidate.passes:
            twin = candidate
    return twin


def _outcome(
    twin: Candidate | None,
    chosen: Candidate | None,
    candidates: list[Candidate],
) -> str:
    """Why the sample is replaced or excluded, as written in the table.

    Parameters
    ----------
    twin : Candidate or None
        From ``_twin``.
    chosen : Candidate or None
        The replacement gauge, if any.
    candidates : list of Candidate
        From ``_nearby_candidates``.

    Returns
    -------
    str
        ``replaced: ...`` or ``excluded: ...``.
    """
    if twin:
        return "replaced: twin"
    if chosen:
        return f"replaced: nearest with r >= {MIN_R}"
    if not candidates:
        return f"excluded: no covering gauge within {MAX_DISTANCE_KM:g} km"
    if all(
        candidate.correlation.common_days < MIN_COMMON_DAYS
        for candidate in candidates
    ):
        return (
            "excluded: correlation not assessable "
            f"(< {MIN_COMMON_DAYS} common days)"
        )
    return f"excluded: correlation below {MIN_R}"


def _best_r(candidates: list[Candidate]) -> Candidate | None:
    """The candidate with the highest assessable r, if any.

    Parameters
    ----------
    candidates : list of Candidate
        From ``_nearby_candidates``.

    Returns
    -------
    Candidate or None
        None if there are no candidates.
    """
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            candidate.correlation.pearson
            if candidate.correlation.common_days >= MIN_COMMON_DAYS
            and not np.isnan(candidate.correlation.pearson)
            else NO_R_SORT_KEY
        ),
    )


def _describe(
    prefix: str, candidate: Candidate | None, fields: list[str]
) -> dict[str, object]:
    """Columns describing one candidate (all None if there is none).

    Parameters
    ----------
    prefix : str
        Column prefix, e.g. ``nearest`` gives ``nearest``, ``nearest_km``.
    candidate : Candidate or None
        The gauge to describe.
    fields : list of str
        Which of ``km``, ``r``, ``spearman``, ``common_days``, ``ratio``.

    Returns
    -------
    dict
        Column name -> value: the code under ``prefix`` itself, then one
        column per field.
    """
    values = {"km": None, "r": None, "spearman": None}
    values |= {"common_days": None, "ratio": None}
    if candidate is not None:
        values = {
            "km": round(candidate.distance_km, 2),
            "r": _round_or_none(candidate.correlation.pearson, 3),
            "spearman": _round_or_none(candidate.correlation.spearman, 3),
            "common_days": candidate.correlation.common_days,
            "ratio": _round_or_none(candidate.correlation.ratio_of_totals, 2),
        }
    columns = {prefix: candidate.code if candidate else None}
    columns |= {f"{prefix}_{field}": values[field] for field in fields}
    return columns


def study_sample(
    sample: pd.Series,
    readings: dict[str, pd.Series],
    accumulated_days: dict[str, set[pd.Timestamp]],
    station_xy: pd.DataFrame,
) -> dict[str, object] | None:
    """Replacement gauge, or exclusion, for one sample.

    Parameters
    ----------
    sample : pandas.Series
        One row of ``samples_with_gauges``.
    readings : dict of str to pandas.Series
        From ``real_readings``.
    accumulated_days : dict of str to set of pandas.Timestamp
        Days of -4 runs per gauge.
    station_xy : pandas.DataFrame
        ``stations.csv`` coordinates indexed by code.

    Returns
    -------
    dict or None
        One row of the study, or None if the own gauge covers the window.
    """
    covered, own_days = coverage(readings, sample["code"], sample["date"])
    if covered:
        return None
    candidates = _nearby_candidates(sample, readings, station_xy)
    twin = _twin(sample, readings, station_xy)
    passing = [candidate for candidate in candidates if candidate.passes]
    chosen = twin or (passing[0] if passing else None)
    own_series = readings.get(sample["code"])
    best = _best_r(candidates)
    return {
        "sample_id": sample["sample_id"],
        "sample_type": sample["sample_type"],
        "date": sample["date"].date(),
        "own_gauge": sample["code"],
        "own_days_90": own_days,
        "event_day_in_minus4_run": sample["date"]
        in accumulated_days.get(sample["code"], set()),
        "own_event_day_real": bool(
            own_series is not None and sample["date"] in own_series.index
        ),
        "n_candidates_5km": len(candidates),
        **_describe(
            "nearest",
            candidates[0] if candidates else None,
            ["km", "r", "common_days"],
        ),
        "best_r_gauge": best.code if best else None,
        "best_r": _round_or_none(best.correlation.pearson, 3)
        if best
        else None,
        "best_r_km": round(best.distance_km, 2) if best else None,
        **_describe(
            "chosen", chosen, ["km", "r", "spearman", "common_days", "ratio"]
        ),
        "outcome": _outcome(twin, chosen, candidates),
    }


def main() -> None:
    """Run the study and write its two tables (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_argument(parser)
    run = run_folder(parser.parse_args().run)
    output = analysis_folder(run, "gauge_study")
    raw = pd.read_csv(DATA_PATH / "historical_aemet_data.csv", sep=";")
    aemet = format_aemet_data(raw)
    print(
        "days with a value:",
        len(aemet),
        "| part of a -4 run:",
        int(aemet["accumulated"].sum()),
    )
    readings = real_readings(aemet)
    accumulated_days = {
        code: set(group["date"])
        for code, group in aemet[aemet["accumulated"]].groupby("code")
    }
    stations = pd.read_csv(DATA_PATH / "stations.csv")
    station_xy = stations.set_index("code")[["utm_x", "utm_y"]]

    neighbours = neighbour_table(readings, gauge_coordinates(aemet, stations))
    neighbours.to_csv(output / "twins_B022_B001.csv", index=False)
    print(neighbours.to_string(index=False))

    rows = []
    samples = samples_with_gauges(stations, run / "outputs" / "final_gdf.csv")
    for _, sample in samples.iterrows():
        row = study_sample(sample, readings, accumulated_days, station_xy)
        if row is not None:
            rows.append(row)
    study = pd.DataFrame(rows)
    study.to_csv(output / "replacement_study.csv", index=False)
    print(
        pd.crosstab(
            study["sample_type"], study["outcome"], margins=True
        ).to_string()
    )


if __name__ == "__main__":
    main()
