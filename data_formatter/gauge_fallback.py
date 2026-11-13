"""Replacement gauge, or exclusion, for samples their own gauge didn't measure.

Problem: a sample's rain comes from the gauge of its slope-weighted Thiessen
polygon. When that gauge has gaps, the missing days count as 0 mm, and the
rain windows are too low. Some rockfalls had no data at all, while the
temporal controls are >= 90% measured by rule, so the tree models learned
"no rain data -> rockfall" (vault: "Missing rain data and gauge fallback").

Idea (vault D-012, D-029): the slope-weighted Thiessen polygons recomputed,
for each sample, with only the gauges that measured its window, and the
gauge of the recomputed polygon the sample falls in. A polygon is the area
a gauge reaches more cheaply than any other, and a gauge's cost to a place
doesn't depend on the other gauges, so "the polygon it falls in, among the
gauges with data" is the same as "the cheapest gauge with data". No
polygons have to be redrawn: one table of costs from every site to every
gauge (cost_distance.sample_gauge_costs) is enough.

But distance alone doesn't guarantee the other gauge saw the same rain
(B001-B780: r = 0.11 at 11 km), so a replacement also has to correlate with
the own gauge (vault D-021). Rule, applied the same way to every sample
(rockfall, control, random point) at an AEMET gauge:

  1. If the own gauge covers the window, keep it (status ``own``). Covering
     means at least ``min_coverage`` of the 90 days before the event are
     real readings and, if ``require_event_day``, the event day is one.
     Days of -4 runs are not real readings (vault D-015; controls.py).
  2. Otherwise a **twin** of the own gauge (same code with a suffix added or
     removed, e.g. B158 / B158X) that covers the window and passes the
     correlation test, and is <= ``twin_max_distance_km`` from the own gauge
     or has r above ``min_r`` (status ``twin``). A twin takes over the own
     gauge's polygon, so its distance to the sample doesn't matter.
  3. Otherwise **reallocation** (vault D-029): the cheapest stations.csv
     gauge by cost distance that covers the window, with no distance limit
     (status ``reallocated``). It must pass the correlation test; if it
     fails, the sample is excluded: taking the next gauge would use one
     whose polygon the sample isn't in.
  4. Otherwise the sample is **excluded**, with the reason. If it is a
     rockfall, its temporal controls are excluded too (vault D-023).

With cost distance off (``cost_distance.enabled: false``), step 3 is the
rule before D-029: the nearest gauge by straight-line distance, within
``max_distance_km``, that covers the window and passes the correlation test
(status ``nearest``). With cost distance on, that rule is still evaluated
and reported (``old_rule_*`` columns of gauge_fallback.csv), so one run
shows which replacements and exclusions the reallocation changed.

Correlation test: Pearson r of the daily rain of the two gauges over the
days both have a real reading in ``correlation_period``; at least
``min_common_days`` of them and r >= ``min_r``. With 365 days, the 95%
interval of r around 0.90 is about 0.88-0.92.

Every exclusion is reported (``exclusions.csv``, in every run, empty if
none): which sample, why, and the gauges that were considered.

Caveats:
  - Twins are chosen by straight-line distance: they are the same site, so
    km between gauges is the right measure. Straight-line distances are
    also reported for every gauge chosen (``rain_station_distance_km``).
  - Only the gauge used for the rain changes. The CV group (id_station, the
    original polygon) never changes, so a site and its controls always stay
    in the same LeaveOneGroupOut fold.
  - Temporal controls are drawn before this step, at the site's own gauge,
    on covered days only, so in practice only rockfalls and random points
    get a replacement or are excluded for their own sake.
  - The RiscBal gauges aren't in the AEMET file: their samples are not
    checked (status ``not_checked``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data_formatter.controls import ANTECEDENT_DAYS, antecedent_coverage

logger = logging.getLogger(__name__)

# Values of the `rain_station_status` column.
OWN = "own"  # own gauge covers the window
TWIN = "twin"  # a twin of the own gauge used instead
NEAREST = "nearest"  # the nearest gauge that passes (cost distance off)
REALLOCATED = "reallocated"  # the cheapest covering gauge (vault D-029)
NOT_CHECKED = "not_checked"  # gauge outside the AEMET file (RiscBal)

# Reason given in exclusions.csv for the controls of an excluded rockfall.
ROCKFALL_EXCLUDED = "rockfall excluded"

# Fewer common days than this: r is not computed at all (reported as n/a).
MIN_DAYS_FOR_ANY_R = 30
# Cost distances are reported to this many decimals (units: metres x degrees).
COST_DECIMALS = 1
# A twin's code is the own code with letters added or removed at the end.
TWIN_SUFFIX = re.compile(r"[A-Z]+$")

EXCLUSION_COLUMNS = [
    "sample_id",
    "sample_type",
    "source_sample_id",
    "date",
    "own_gauge",
    "own_real_days_90d",
    "event_day_in_minus4_run",
    "reason",
    "nearest",
    "nearest_km",
    "nearest_r",
    "nearest_common_days",
    "best_r_gauge",
    "best_r",
    "best_r_km",
    "reallocated",
    "reallocated_km",
    "reallocated_cost",
    "reallocated_r",
    "reallocated_common_days",
    "old_rule_gauge",
]
# Columns of EXCLUSION_COLUMNS that are counts (integers with empty cells).
EXCLUSION_COUNT_COLUMNS = [
    "own_real_days_90d",
    "nearest_common_days",
    "reallocated_common_days",
]


@dataclass(frozen=True)
class FallbackSettings:
    """When and how a sample's own gauge is replaced (``fallback``).

    Attributes
    ----------
    enabled : bool
        Feature flag, checked by the orchestrator (main.py), not here.
    min_coverage : float
        Share of the 90 antecedent days that must be real readings.
    require_event_day : bool
        The event day itself must be a real reading (it is the triggering
        rain).
    max_distance_km : float or None
        Rule before D-029 (cost distance off, and the ``old_rule_*`` report
        columns): furthest gauge from the sample allowed as a replacement
        (the twin step aside); None means no limit.
    twin_max_distance_km : float
        A twin must be this close to the own gauge, unless r > ``min_r``.
    min_r : float
        Lowest Pearson r with the own gauge accepted for a replacement.
    min_common_days : int
        Fewest common real days for r to count.
    correlation_period : tuple of str
        First and last day (ISO dates) of the correlation.
    """

    enabled: bool
    min_coverage: float
    require_event_day: bool
    max_distance_km: float | None
    twin_max_distance_km: float
    min_r: float
    min_common_days: int
    correlation_period: tuple[str, str]


@dataclass(frozen=True)
class Correlation:
    """Agreement of two gauges' daily rain over their common real readings.

    Attributes
    ----------
    common_days : int
        Days both gauges have a real reading in the correlation period.
    pearson : float
        Pearson r; NaN with fewer than ``MIN_DAYS_FOR_ANY_R`` common days.
    """

    common_days: int
    pearson: float

    def passes(self, settings: FallbackSettings) -> bool:
        """Whether the correlation test accepts this pair of gauges.

        Parameters
        ----------
        settings : FallbackSettings
            ``min_common_days`` and ``min_r``.

        Returns
        -------
        bool
            Enough common days and r >= ``min_r`` (False if r is NaN).
        """
        return (
            self.common_days >= settings.min_common_days
            and self.pearson >= settings.min_r
        )


@dataclass(frozen=True)
class Candidate:
    """A possible replacement gauge for one sample.

    Attributes
    ----------
    code : str
        Gauge code.
    distance_km : float
        Straight-line distance to the sample.
    cost : float
        Cost distance from the sample (NaN with cost distance off).
    correlation : Correlation
        Agreement with the sample's own gauge.
    """

    code: str
    distance_km: float
    cost: float
    correlation: Correlation


@dataclass
class Correlations:
    """Correlations between gauges, computed once per pair.

    The same pairs come up for many samples (e.g. every B022 sample asks
    about B684A), and each correlation aligns two daily series, so they are
    cached.

    Attributes
    ----------
    readings : dict of str to pandas.Series
        Real daily rain of every gauge (``real_readings``).
    settings : FallbackSettings
        The correlation period.
    cache : dict
        ``(first, second) -> Correlation`` (r is symmetric, so one order is
        stored).
    """

    readings: dict[str, pd.Series]
    settings: FallbackSettings
    cache: dict[tuple[str, str], Correlation] = field(default_factory=dict)

    def get(self, first: str, second: str) -> Correlation:
        """Correlation of two gauges' daily rain.

        Parameters
        ----------
        first, second : str
            Gauge codes.

        Returns
        -------
        Correlation
            ``Correlation(0, nan)`` if either gauge has no real readings.
        """
        key = (min(first, second), max(first, second))
        if key not in self.cache:
            self.cache[key] = self._compute(*key)
        return self.cache[key]

    def _compute(self, first: str, second: str) -> Correlation:
        """Correlation of two gauges, without the cache.

        Parameters
        ----------
        first, second : str
            Gauge codes.

        Returns
        -------
        Correlation
            Pearson r over the common real days of the period; NaN if one
            gauge's rain is constant on those days (r is undefined: it
            divides by a standard deviation of 0). E.g. B268, whose record
            starts in December 2015: 31 dry days in the period.
        """
        a, b = self.readings.get(first), self.readings.get(second)
        if a is None or b is None:
            return Correlation(0, np.nan)
        start, end = self.settings.correlation_period
        common = pd.concat([a, b], axis=1, join="inner").loc[start:end]
        if len(common) < MIN_DAYS_FOR_ANY_R or (common.std() == 0).any():
            return Correlation(len(common), np.nan)
        return Correlation(
            len(common), common.iloc[:, 0].corr(common.iloc[:, 1])
        )


def real_readings(aemet: pd.DataFrame) -> dict[str, pd.Series]:
    """Daily rain of every AEMET gauge, real readings only.

    Parameters
    ----------
    aemet : pandas.DataFrame
        The reshaped AEMET table (``code``, ``date``, ``precipitation``,
        ``accumulated``).

    Returns
    -------
    dict of str to pandas.Series
        For each gauge code, rain in mm indexed by date, without the days of
        -4 runs.
    """
    real = aemet[~aemet["accumulated"]]
    return {
        code: group.set_index("date")["precipitation"]
        for code, group in real.groupby("code")
    }


def is_twin(first: str, second: str) -> bool:
    """Whether two gauge codes are twins: the same code ± a letter suffix.

    E.g. B158 / B158X, B087 / B087X, B684 / B684A. B174 / B186 are not
    twins, however close (vault D-021).

    Parameters
    ----------
    first, second : str
        Gauge codes.

    Returns
    -------
    bool
        Different codes with the same code once trailing letters are
        removed.
    """
    return first != second and TWIN_SUFFIX.sub("", first) == TWIN_SUFFIX.sub(
        "", second
    )


def _covers(
    days: pd.DatetimeIndex, dates: pd.DatetimeIndex, settings: FallbackSettings
) -> np.ndarray:
    """Whether a gauge with these measured days covers each date's window.

    Parameters
    ----------
    days : pandas.DatetimeIndex
        Days the gauge measured (real readings).
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


def _twin(
    own: str,
    distance_km: pd.Series,
    cost: pd.Series,
    covering: pd.Series,
    station_xy: pd.DataFrame,
    correlations: Correlations,
    settings: FallbackSettings,
) -> Candidate | None:
    """Step 2 of the rule: a twin of the own gauge that may replace it.

    Parameters
    ----------
    own : str
        The sample's own gauge.
    distance_km : pandas.Series
        Distance from the sample to every candidate gauge, by code.
    cost : pandas.Series
        Cost distance from the sample to every candidate gauge, by code
        (reported only).
    covering : pandas.Series
        Whether each candidate gauge covers the sample's window, by code.
    station_xy : pandas.DataFrame
        ``stations.csv`` coordinates indexed by code.
    correlations : Correlations
        Correlation cache.
    settings : FallbackSettings
        Twin distance and correlation test.

    Returns
    -------
    Candidate or None
        The passing twin nearest to the sample, or None.
    """
    if own not in station_xy.index:
        return None
    own_x, own_y = station_xy.loc[own, ["utm_x", "utm_y"]]
    twins = []
    for code in covering.index[covering]:
        if not is_twin(own, code):
            continue
        correlation = correlations.get(own, code)
        gauge_gap_km = (
            np.hypot(
                station_xy.loc[code, "utm_x"] - own_x,
                station_xy.loc[code, "utm_y"] - own_y,
            )
            / 1000
        )
        close_or_better = (
            gauge_gap_km <= settings.twin_max_distance_km
            or correlation.pearson > settings.min_r
        )
        if correlation.passes(settings) and close_or_better:
            twins.append(
                Candidate(code, distance_km[code], cost[code], correlation)
            )
    return min(twins, key=lambda twin: twin.distance_km, default=None)


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


def _describe(prefix: str, candidate: Candidate | None) -> dict[str, object]:
    """Report columns of one candidate gauge (all None if there is none).

    Parameters
    ----------
    prefix : str
        Column prefix, e.g. ``nearest`` gives ``nearest``, ``nearest_km``,
        ``nearest_cost``, ``nearest_r`` and ``nearest_common_days``.
    candidate : Candidate or None
        The gauge to describe.

    Returns
    -------
    dict
        Column name -> value.
    """
    if candidate is None:
        return {
            prefix: None,
            f"{prefix}_km": None,
            f"{prefix}_cost": None,
            f"{prefix}_r": None,
            f"{prefix}_common_days": None,
        }
    return {
        prefix: candidate.code,
        f"{prefix}_km": round(candidate.distance_km, 2),
        f"{prefix}_cost": _round_or_none(candidate.cost, COST_DECIMALS),
        f"{prefix}_r": _round_or_none(candidate.correlation.pearson, 3),
        f"{prefix}_common_days": candidate.correlation.common_days,
    }


def _exclusion_reason(
    candidates: list[Candidate], settings: FallbackSettings
) -> str:
    """Why a sample without a replacement is excluded.

    Parameters
    ----------
    candidates : list of Candidate
        The covering gauges within ``max_distance_km`` (step 3).
    settings : FallbackSettings
        Distance and correlation thresholds, for the wording.

    Returns
    -------
    str
        The reason, as written in ``exclusions.csv``.
    """
    if not candidates:
        return (
            f"no covering gauge within {settings.max_distance_km:g} km"
            if settings.max_distance_km is not None
            else "no covering gauge"
        )
    if all(
        candidate.correlation.common_days < settings.min_common_days
        for candidate in candidates
    ):
        return (
            "correlation not assessable "
            f"(< {settings.min_common_days} common days)"
        )
    return f"correlation below {settings.min_r:g}"


def _reallocated(
    own: str,
    covering: pd.Series,
    distance_km: pd.Series,
    cost: pd.Series,
    correlations: Correlations,
) -> Candidate | None:
    """Step 3 (vault D-029): the gauge of the recomputed polygon.

    Parameters
    ----------
    own : str
        The sample's own gauge (its correlation partner).
    covering : pandas.Series
        Whether each candidate gauge covers the sample's window, by code
        (the own gauge already False).
    distance_km : pandas.Series
        Straight-line distance from the sample, by code: breaks exact ties.
    cost : pandas.Series
        Cost distance from the sample, by code (inf: no land path).
    correlations : Correlations
        Correlation cache.

    Returns
    -------
    Candidate or None
        The cheapest covering, reachable gauge (whether it passes the
        correlation test is the caller's decision), or None.
    """
    reachable = covering & np.isfinite(cost)
    if not reachable.any():
        return None
    # sorted() is stable: on a full tie (co-located gauges, e.g. B013 /
    # B013X) the first in stations.csv order wins, i.e. the lower id, as in
    # cost_distance.allocate and ArcGIS.
    code = min(
        reachable.index[reachable],
        key=lambda code: (cost[code], distance_km[code]),
    )
    return Candidate(
        code, distance_km[code], cost[code], correlations.get(own, code)
    )


def _reallocation_reason(
    reallocated: Candidate | None,
    covering: pd.Series,
    settings: FallbackSettings,
) -> str:
    """Why a sample whose reallocated gauge can't be used is excluded.

    The reason is a category, so exclusions group by it; the gauge, its r
    and common days are in the ``reallocated*`` columns.

    Parameters
    ----------
    reallocated : Candidate or None
        From ``_reallocated``.
    covering : pandas.Series
        Whether each candidate gauge covers the sample's window, by code.
    settings : FallbackSettings
        Correlation thresholds, for the wording.

    Returns
    -------
    str
        The reason, as written in ``exclusions.csv``.
    """
    if reallocated is None:
        return (
            "no covering gauge reachable (no land path)"
            if covering.any()
            else "no gauge covers the window"
        )
    correlation = reallocated.correlation
    if correlation.common_days < settings.min_common_days:
        return (
            "reallocated gauge: correlation not assessable "
            f"(< {settings.min_common_days} common days)"
        )
    if np.isnan(correlation.pearson):
        return (
            "reallocated gauge: correlation undefined "
            "(constant rain on the common days)"
        )
    return f"reallocated gauge: r below {settings.min_r:g}"


def _best_r(
    candidates: list[Candidate], settings: FallbackSettings
) -> Candidate | None:
    """The candidate with the highest assessable r, for the report.

    Parameters
    ----------
    candidates : list of Candidate
        The covering gauges within ``max_distance_km``.
    settings : FallbackSettings
        ``min_common_days``.

    Returns
    -------
    Candidate or None
        None if no candidate has an assessable r.
    """
    assessable = [
        candidate
        for candidate in candidates
        if candidate.correlation.common_days >= settings.min_common_days
        and not np.isnan(candidate.correlation.pearson)
    ]
    return max(
        assessable,
        key=lambda candidate: candidate.correlation.pearson,
        default=None,
    )


def assign_replacement_gauges(
    samples: pd.DataFrame,
    stations: pd.DataFrame,
    aemet: pd.DataFrame,
    days_by_gauge: dict[str, pd.DatetimeIndex],
    settings: FallbackSettings,
    cost: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the rule of the module docstring to every sample.

    Parameters
    ----------
    samples : pandas.DataFrame
        With ``date``, ``code`` (its gauge), ``POINT_X``, ``POINT_Y``,
        ``sample_id``, ``sample_type`` and ``source_sample_id``.
    stations : pandas.DataFrame
        ``stations.csv`` (``code``, ``utm_x``, ``utm_y``): the candidate
        gauges.
    aemet : pandas.DataFrame
        The reshaped AEMET table (for the real readings and the -4 runs).
    days_by_gauge : dict of str to pandas.DatetimeIndex
        Days each AEMET gauge measured (real readings).
    settings : FallbackSettings
        Coverage rule, distances and correlation test.
    cost : pandas.DataFrame or None
        Cost distance from every sample to every gauge
        (``cost_distance.sample_gauge_costs``): step 3 is the reallocation.
        None (cost distance off): step 3 is the rule before D-029.

    Returns
    -------
    samples : pandas.DataFrame
        A copy of all samples (excluded ones included) with ``code``
        updated to the gauge used, plus ``rain_station_original``,
        ``rain_station_status`` (own, twin, reallocated or nearest,
        not_checked; excluded samples keep ``own`` here),
        ``rain_station_distance_km`` and ``rain_station_cost``.
    report : pandas.DataFrame
        One row per sample its own gauge didn't cover: own gauge and its
        real days, the nearest covering gauge within ``max_distance_km``,
        the reallocated gauge, the gauge chosen and the outcome, and what
        the rule before D-029 would have done (``old_rule_gauge``,
        ``old_rule_excluded``, ``rule_changed``); ``excluded`` marks the
        samples to drop.
    """
    samples = samples.copy()
    dates = pd.DatetimeIndex(samples["date"]).normalize()
    samples["rain_station_original"] = samples["code"]
    samples["rain_station_status"] = NOT_CHECKED
    samples["rain_station_distance_km"] = np.nan
    samples["rain_station_cost"] = np.nan

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
    todo = samples[uncovered]
    if todo.empty:
        logger.info("Gauge replacement: every own gauge covers its window")
        return samples, pd.DataFrame(columns=["excluded"])

    # Candidate gauges: stations.csv gauges with AEMET data.
    candidates = stations[stations["code"].isin(days_by_gauge)].reset_index(
        drop=True
    )
    codes = candidates["code"].to_numpy()
    todo_dates = dates[samples.index.get_indexer(todo.index)]
    # covers[i, j]: candidate gauge j covers sample i's window.
    covers = np.column_stack(
        [_covers(days_by_gauge[code], todo_dates, settings) for code in codes]
    )
    distance = _distances_km(todo, candidates)
    station_xy = stations.set_index("code")[["utm_x", "utm_y"]]
    correlations = Correlations(real_readings(aemet), settings)
    minus4_days = {
        code: set(group["date"])
        for code, group in aemet[aemet["accumulated"]].groupby("code")
    }

    rows = []
    for i, (index, sample) in enumerate(todo.iterrows()):
        own, date = sample["code"], todo_dates[i]
        sample_km = pd.Series(distance[i], index=codes)
        sample_cost = (
            cost.loc[index, codes]
            if cost is not None
            else pd.Series(np.nan, index=codes)
        )
        covering = pd.Series(covers[i] & (codes != own), index=codes)

        # Rule before D-029 (step 3 with cost distance off; reported in
        # both cases): covering gauges near the sample, nearest first.
        near = covering & (
            sample_km <= settings.max_distance_km
            if settings.max_distance_km is not None
            else True
        )
        nearby = sorted(
            (
                Candidate(
                    code,
                    sample_km[code],
                    sample_cost[code],
                    correlations.get(own, code),
                )
                for code in near.index[near]
            ),
            key=lambda candidate: candidate.distance_km,
        )
        # 2. A twin first.
        twin = _twin(
            own,
            sample_km,
            sample_cost,
            covering,
            station_xy,
            correlations,
            settings,
        )
        passing = [c for c in nearby if c.correlation.passes(settings)]
        old_choice = twin or (passing[0] if passing else None)

        # 3. The reallocated gauge if it passes (cost distance on); else
        # the rule before D-029.
        reallocated = None
        if cost is None:
            chosen = old_choice
            replaced_status = NEAREST
            replaced_outcome = (
                f"replaced: nearest with r >= {settings.min_r:g}"
            )
            reason = _exclusion_reason(nearby, settings)
        else:
            reallocated = _reallocated(
                own, covering, sample_km, sample_cost, correlations
            )
            usable = (
                reallocated is not None
                and reallocated.correlation.passes(settings)
            )
            chosen = twin or (reallocated if usable else None)
            replaced_status = REALLOCATED
            replaced_outcome = (
                f"replaced: reallocated with r >= {settings.min_r:g}"
            )
            reason = _reallocation_reason(reallocated, covering, settings)

        if twin is not None:
            outcome = "replaced: twin"
        elif chosen is not None:
            outcome = replaced_outcome
        else:
            outcome = f"excluded: {reason}"
        if chosen is not None:
            samples.loc[index, "code"] = chosen.code
            samples.loc[index, "rain_station_status"] = (
                TWIN if twin is not None else replaced_status
            )
            samples.loc[index, "rain_station_distance_km"] = round(
                chosen.distance_km, 2
            )
            samples.loc[index, "rain_station_cost"] = chosen.cost

        best = _best_r(nearby, settings)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "sample_type": sample["sample_type"],
                "source_sample_id": sample["source_sample_id"],
                "date": date.date(),
                "own_gauge": own,
                "own_real_days_90d": int(
                    antecedent_coverage(
                        days_by_gauge[own], pd.DatetimeIndex([date])
                    )[0]
                ),
                "event_day_in_minus4_run": date in minus4_days.get(own, set()),
                "own_event_day_real": date in days_by_gauge[own],
                "n_candidates_within_max_km": len(nearby),
                **_describe("nearest", nearby[0] if nearby else None),
                "best_r_gauge": best.code if best else None,
                "best_r": _round_or_none(best.correlation.pearson, 3)
                if best
                else None,
                "best_r_km": round(best.distance_km, 2) if best else None,
                **_describe("reallocated", reallocated),
                **_describe("chosen", chosen),
                "outcome": outcome,
                "excluded": chosen is None,
                "old_rule_gauge": old_choice.code if old_choice else None,
                "old_rule_excluded": old_choice is None,
                "rule_changed": (chosen.code if chosen else None)
                != (old_choice.code if old_choice else None),
            }
        )
    report = pd.DataFrame(rows)

    logger.info(
        "Gauge replacement: %d of %d samples not covered by their own gauge "
        "(>= %.0f%% of %d days%s, real readings only):\n%s",
        len(report),
        len(samples),
        100 * settings.min_coverage,
        ANTECEDENT_DAYS,
        " + event day" if settings.require_event_day else "",
        pd.crosstab(report["sample_type"], report["outcome"]).to_string(),
    )
    if cost is not None:
        logger.info(
            "Reallocation vs the rule before D-029 (nearest within %s km): "
            "%d samples with another gauge or outcome (rule_changed in "
            "gauge_fallback.csv)",
            settings.max_distance_km,
            int(report["rule_changed"].sum()),
        )
    return samples, report


def drop_excluded(
    samples: pd.DataFrame, report: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove the excluded samples, and the controls of excluded rockfalls.

    Parameters
    ----------
    samples : pandas.DataFrame
        All samples, from ``assign_replacement_gauges``.
    report : pandas.DataFrame
        From ``assign_replacement_gauges``.

    Returns
    -------
    kept : pandas.DataFrame
        The samples that stay in the dataset.
    exclusions : pandas.DataFrame
        One row per removed sample (``EXCLUSION_COLUMNS``): its own reason,
        or ``rockfall excluded`` for the controls of an excluded rockfall
        (vault D-023).
    """
    if report.empty:
        return samples, pd.DataFrame(columns=EXCLUSION_COLUMNS)
    own_reason = report[report["excluded"]].copy()
    own_reason["reason"] = own_reason["outcome"].str.removeprefix("excluded: ")

    rockfalls = own_reason.loc[
        own_reason["sample_type"] == "rockfall", "sample_id"
    ]
    orphan = (
        (samples["sample_type"] == "temporal_control")
        & samples["source_sample_id"].isin(rockfalls)
        & ~samples["sample_id"].isin(own_reason["sample_id"])
    )
    controls = samples[orphan]
    orphans = pd.DataFrame(
        {
            "sample_id": controls["sample_id"],
            "sample_type": controls["sample_type"],
            "source_sample_id": controls["source_sample_id"],
            "date": pd.DatetimeIndex(controls["date"]).date,
            "own_gauge": controls["rain_station_original"],
            "reason": ROCKFALL_EXCLUDED,
        }
    )
    exclusions = pd.concat(
        [own_reason.reindex(columns=EXCLUSION_COLUMNS), orphans],
        ignore_index=True,
    ).reindex(columns=EXCLUSION_COLUMNS)
    # Counts stay integers with empty cells (the controls' rows), not 65.0.
    exclusions = exclusions.astype(
        dict.fromkeys(EXCLUSION_COUNT_COLUMNS, "Int64")
    )

    kept = samples[~samples["sample_id"].isin(exclusions["sample_id"])]
    logger.info(
        "Excluded: %d samples (see exclusions.csv):\n%s",
        len(exclusions),
        pd.crosstab(
            exclusions["sample_type"], exclusions["reason"]
        ).to_string()
        if not exclusions.empty
        else "  none",
    )
    return kept.reset_index(drop=True), exclusions


def use_fallback_gauges(
    samples: pd.DataFrame,
    stations: pd.DataFrame,
    aemet: pd.DataFrame,
    days_by_gauge: dict[str, pd.DatetimeIndex],
    settings: FallbackSettings,
    output_path: Path,
    cost: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace or exclude the samples their own gauge didn't measure.

    Runs after the temporal controls, so they are drawn at their site's own
    gauge. From here on, ``code`` is the gauge the rain is read from
    (written out as ``rain_station``). Writes ``gauge_fallback.csv`` and
    ``exclusions.csv`` to the run's outputs folder.

    Parameters
    ----------
    samples : pandas.DataFrame
        All samples.
    stations : pandas.DataFrame
        ``stations.csv``, the candidate gauges.
    aemet : pandas.DataFrame
        The reshaped AEMET table.
    days_by_gauge : dict of str to pandas.DatetimeIndex
        Days each AEMET gauge measured (real readings).
    settings : FallbackSettings
        Coverage rule, distances and correlation test.
    output_path : pathlib.Path
        The run's outputs folder (``runs/<run_id>/outputs/``).
    cost : pandas.DataFrame or None
        Cost distance from every sample to every gauge; None with cost
        distance off (see ``assign_replacement_gauges``).

    Returns
    -------
    samples : pandas.DataFrame
        The samples kept, with the gauge used and the ``rain_station_*``
        columns.
    report : pandas.DataFrame
        One row per sample not covered by its own gauge.
    exclusions : pandas.DataFrame
        One row per sample removed, with the reason.
    """
    samples, report = assign_replacement_gauges(
        samples, stations, aemet, days_by_gauge, settings, cost
    )
    kept, exclusions = drop_excluded(samples, report)
    report.to_csv(output_path / "gauge_fallback.csv", index=False)
    exclusions.to_csv(output_path / "exclusions.csv", index=False)
    return kept, report, exclusions


def no_exclusions(output_path: Path) -> pd.DataFrame:
    """Write an empty ``exclusions.csv`` (fallback switched off).

    The file is in every run, so "nothing was excluded" is stated, not
    assumed from a missing file.

    Parameters
    ----------
    output_path : pathlib.Path
        The run's outputs folder.

    Returns
    -------
    pandas.DataFrame
        The empty exclusions table.
    """
    exclusions = pd.DataFrame(columns=EXCLUSION_COLUMNS)
    exclusions.to_csv(output_path / "exclusions.csv", index=False)
    return exclusions
