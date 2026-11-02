"""Fallback gauge for samples whose own gauge didn't measure their window.

Problem: a sample's rain comes from the gauge of its slope-weighted Thiessen
polygon. When that gauge has gaps, the missing days count as 0 mm, and the
rain windows are too low. Some rockfalls have no data at all, while the
temporal controls are >= 90% measured by rule, so the tree models learned
"no rain data -> rockfall" (see docs/08-limitations.md, "Coverage asymmetry").

Idea: Thiessen polygons recomputed, for each sample, with only the gauges
that measured its window. A Thiessen polygon is the area closer to one gauge
than to any other, so "the polygon it falls in, among the gauges with data"
is the same as "the nearest gauge with data". No polygons have to be drawn:
it is enough to rank the gauges by distance.

Rule, applied the same way to every sample (rockfall, control, random point):
  1. If the sample's own gauge covers its window, keep it. Covering means at
     least `min_coverage` of the 90 days before the event measured and, if
     `require_event_day`, the event day measured too.
  2. Otherwise use the nearest gauge (from stations.csv) that covers it,
     within `max_distance_km`.
  3. If there is none, keep the own gauge (the sample stays under-measured;
     its rain_days_90d shows it).

Caveats:
  - Distance is straight-line (UTM metres), a stand-in for the slope-weighted
    cost distance of the original polygons, which we don't have yet. Swap it
    in `_distances_km` when the cost-distance table is available.
  - Only the gauge used for the rain changes. The CV group (id_station, the
    original polygon) never changes, so a site and its controls always stay
    in the same LeaveOneGroupOut fold.
  - Temporal controls are drawn before this step, at the site's own gauge,
    with >= 90% coverage by rule, so in practice only rockfalls and random
    points are reassigned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_formatter.controls import ANTECEDENT_DAYS, antecedent_coverage

logger = logging.getLogger(__name__)

# Values of the `rain_station_status` column.
OWN = "own"  # own gauge covers the window
REASSIGNED = "reassigned"  # nearest gauge with data used instead
NO_GAUGE = "no_gauge_in_range"  # nobody covers it: own gauge kept
NOT_CHECKED = "not_checked"  # gauge outside the AEMET file (RiscBal)


@dataclass(frozen=True)
class FallbackSettings:
    """When a sample's own gauge is replaced (``fallback`` in config.yaml).

    Attributes
    ----------
    enabled : bool
        Feature flag, checked by the orchestrator (main.py), not here.
    min_coverage : float
        Share of the 90 antecedent days that must be measured.
    require_event_day : bool
        The event day itself must be measured (it is the triggering rain).
    max_distance_km : float or None
        Furthest gauge allowed as a fallback; None means no limit.
    """

    enabled: bool
    min_coverage: float
    require_event_day: bool
    max_distance_km: float | None


def _covers(
    days: pd.DatetimeIndex, dates: pd.DatetimeIndex, settings: FallbackSettings
) -> np.ndarray:
    """Whether a gauge with these measured days covers each date's window.

    Parameters
    ----------
    days : pandas.DatetimeIndex
        Days the gauge measured.
    dates : pandas.DatetimeIndex
        Event dates of the samples.
    settings : FallbackSettings
        Coverage share and whether the event day must be measured.

    Returns
    -------
    numpy.ndarray
        One boolean per date.
    """
    enough = (
        antecedent_coverage(days, dates)
        >= settings.min_coverage * ANTECEDENT_DAYS
    )
    if settings.require_event_day:
        enough &= dates.isin(days)
    return enough


def _distances_km(samples: pd.DataFrame, stations: pd.DataFrame) -> np.ndarray:
    """Straight-line distance from each sample to each gauge.

    Parameters
    ----------
    samples : pandas.DataFrame
        With ``POINT_X`` and ``POINT_Y`` (UTM 31N, metres).
    stations : pandas.DataFrame
        With ``utm_x`` and ``utm_y`` (UTM 31N, metres).

    Returns
    -------
    numpy.ndarray
        Distances in km, shape (samples, gauges).
    """
    dx = (
        samples["POINT_X"].to_numpy()[:, None]
        - stations["utm_x"].to_numpy()[None, :]
    )
    dy = (
        samples["POINT_Y"].to_numpy()[:, None]
        - stations["utm_y"].to_numpy()[None, :]
    )
    return np.hypot(dx, dy) / 1000


def assign_fallback_gauges(
    samples: pd.DataFrame,
    stations: pd.DataFrame,
    days_by_gauge: dict[str, pd.DatetimeIndex],
    settings: FallbackSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the fallback rule of the module docstring to every sample.

    Parameters
    ----------
    samples : pandas.DataFrame
        With ``date``, ``code`` (its gauge), ``POINT_X``, ``POINT_Y``,
        ``sample_id`` and ``sample_type``.
    stations : pandas.DataFrame
        ``stations.csv`` (``code``, ``utm_x``, ``utm_y``): the candidate
        gauges.
    days_by_gauge : dict of str to pandas.DatetimeIndex
        Days each AEMET gauge measured.
    settings : FallbackSettings
        Coverage rule and maximum distance.

    Returns
    -------
    samples : pandas.DataFrame
        A copy with ``code`` updated to the gauge used, plus
        ``rain_station_original``, ``rain_station_status`` (own,
        reassigned, no_gauge_in_range, not_checked) and
        ``rain_station_distance_km``.
    report : pandas.DataFrame
        One row per sample its own gauge didn't cover: id, type, date,
        original gauge and its coverage, outcome, new gauge and distance.
    """
    samples = samples.copy()
    dates = pd.DatetimeIndex(samples["date"]).normalize()
    samples["rain_station_original"] = samples["code"]
    samples["rain_station_status"] = NOT_CHECKED
    samples["rain_station_distance_km"] = np.nan

    # 1. Does the own gauge cover the window?
    own_covers = pd.Series(False, index=samples.index)
    for code, group in samples.groupby("code"):
        if code not in days_by_gauge:
            continue  # RiscBal gauge: coverage unknown here, left as it is
        position = samples.index.get_indexer(group.index)
        own_covers.loc[group.index] = _covers(
            days_by_gauge[code], dates[position], settings
        )
        samples.loc[group.index, "rain_station_status"] = OWN
    uncovered = (samples["rain_station_status"] == OWN) & ~own_covers
    samples.loc[uncovered, "rain_station_status"] = NO_GAUGE

    # 2. For the uncovered samples: the nearest gauge that covers the window.
    todo = samples[uncovered]
    if not todo.empty:
        todo_dates = dates[samples.index.get_indexer(todo.index)]
        candidates = stations[
            stations["code"].isin(days_by_gauge)
        ].reset_index(drop=True)
        # covers[i, j]: candidate gauge j covers sample i's window.
        covers = np.column_stack(
            [
                _covers(days_by_gauge[code], todo_dates, settings)
                for code in candidates["code"]
            ]
        )
        distance = _distances_km(todo, candidates)
        allowed = covers & (
            candidates["code"].to_numpy()[None, :]
            != todo["code"].to_numpy()[:, None]
        )
        if settings.max_distance_km is not None:
            allowed &= distance <= settings.max_distance_km
        # Distance to disallowed gauges set to infinity, so argmin skips them.
        best = np.where(allowed, distance, np.inf).argmin(axis=1)
        found = allowed[np.arange(len(todo)), best]

        hit = todo.index[found]
        samples.loc[hit, "code"] = candidates["code"].to_numpy()[best[found]]
        samples.loc[hit, "rain_station_status"] = REASSIGNED
        samples.loc[hit, "rain_station_distance_km"] = distance[
            found, best[found]
        ].round(2)

    # 3. Report: one row per sample its own gauge didn't cover.
    report_rows = samples.loc[uncovered]
    report = pd.DataFrame(
        {
            "sample_id": report_rows["sample_id"],
            "sample_type": report_rows["sample_type"],
            "date": dates[samples.index.get_indexer(report_rows.index)].date,
            "original_code": report_rows["rain_station_original"],
            "original_rain_days_90d": [
                int(
                    antecedent_coverage(
                        days_by_gauge[code], pd.DatetimeIndex([day])
                    )[0]
                )
                for code, day in zip(
                    report_rows["rain_station_original"],
                    dates[samples.index.get_indexer(report_rows.index)],
                    strict=True,
                )
            ],
            "status": report_rows["rain_station_status"],
            "new_code": report_rows["code"].where(
                report_rows["rain_station_status"] == REASSIGNED
            ),
            "distance_km": report_rows["rain_station_distance_km"],
        }
    )

    counts = (
        pd.crosstab(report["sample_type"], report["status"])
        if not report.empty
        else report
    )
    logger.info(
        "Gauge fallback: %d of %d samples not covered by their own gauge "
        "(>= %.0f%% of %d days%s):\n%s",
        len(report),
        len(samples),
        100 * settings.min_coverage,
        ANTECEDENT_DAYS,
        " + event day" if settings.require_event_day else "",
        counts.to_string() if not report.empty else "  none",
    )
    return samples, report


def use_fallback_gauges(
    samples: pd.DataFrame,
    stations: pd.DataFrame,
    days_by_gauge: dict[str, pd.DatetimeIndex],
    settings: FallbackSettings,
    output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read under-measured samples' rain from a fallback gauge.

    Runs after the temporal controls, so they are drawn at their site's own
    gauge. From here on, ``code`` is the gauge the rain is read from
    (written out as ``rain_station``). Writes ``gauge_fallback.csv`` to the
    run's outputs folder.

    Parameters
    ----------
    samples : pandas.DataFrame
        All samples.
    stations : pandas.DataFrame
        ``stations.csv``, the candidate gauges.
    days_by_gauge : dict of str to pandas.DatetimeIndex
        Days each AEMET gauge measured.
    settings : FallbackSettings
        Coverage rule and maximum distance.
    output_path : pathlib.Path
        The run's outputs folder (``runs/<run_id>/outputs/``).

    Returns
    -------
    samples : pandas.DataFrame
        The samples, with the gauge used and the ``rain_station_*`` columns.
    report : pandas.DataFrame
        One row per sample not covered by its own gauge.
    """
    samples, report = assign_fallback_gauges(
        samples, stations, days_by_gauge, settings
    )
    report.to_csv(output_path / "gauge_fallback.csv", index=False)
    return samples, report
