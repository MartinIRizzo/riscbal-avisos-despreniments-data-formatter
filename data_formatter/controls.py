"""Temporal controls: non-rockfall days at each rockfall's own site (Phase B1).

For every rockfall, K other days are drawn at random from its rain gauge's
record, and each becomes a negative sample at the *same place*. This is the
idea of the case-crossover design (Maclure, 1991): each case is compared with
itself at other times.

Why, in this project:
  - Terrain: a control has the hillside, hardness, slope and susceptibility of
    its rockfall. Terrain then can't separate a rockfall from its controls,
    and the negatives include dangerous places (steep, north-facing), not
    only the random points, which mostly fell on gentle terrain.
  - Rain: the days are drawn WITHOUT looking at the rain, so wet days appear
    in their natural share (~17% of days have >= 1 mm, ~2% >= 20 mm). The
    model sees "it rained and nothing fell" as often as it really happens.
    This is the difference with the old P95 shift, which picked negatives
    *because* they were rainy and so distorted the learned rain effect.

Only rockfalls at AEMET gauges get controls. The 3 Esporles events at the
RiscBal gauges (BAR04, GRO08) get none for now (TODO: their 10-minute
records have duplicated timestamps to sort out first).

A candidate day D for a rockfall must:
  1. be a day the gauge measured (it has an AEMET value), within `period`;
  2. have at least `min_coverage` of the 90 days before it measured, so the
     antecedent windows aren't "missing counted as 0 mm";
  3. be more than `exclusion_days` away from every recorded rockfall at the
     same gauge, so a control isn't the rockfall day itself, a day of the
     same storm, or a neighbouring event's day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANTECEDENT_DAYS = 90  # the longest rain window (last_three_months_acc)


@dataclass(frozen=True)
class ControlSettings:
    """How temporal controls are drawn (``temporal_controls`` in config.yaml).

    Attributes
    ----------
    enabled : bool
        Feature flag, checked by the orchestrator (main.py), not here.
    per_rockfall : int
        Controls drawn per rockfall (fewer if there aren't enough
        candidate days).
    exclusion_days : int
        No control within this many days of any rockfall at the same gauge.
    min_coverage : float
        Share of the 90 antecedent days that must be measured.
    period : tuple of str
        First and last day (ISO dates) the controls are drawn from: the
        rockfalls' period.
    seed : int
        Seed of the random generator.
    """

    enabled: bool
    per_rockfall: int
    exclusion_days: int
    min_coverage: float
    period: tuple[str, str]
    seed: int


def measured_days(
    aemet_by_station: dict[str, pd.DataFrame],
) -> dict[str, pd.DatetimeIndex]:
    """Days each AEMET gauge measured.

    The formatted AEMET table has one row per station and day with a value;
    a missing day has no row.

    Parameters
    ----------
    aemet_by_station : dict of str to pandas.DataFrame
        The formatted AEMET table split by gauge code.

    Returns
    -------
    dict of str to pandas.DatetimeIndex
        For each gauge code, the sorted unique days with a value.
    """
    return {
        code: pd.DatetimeIndex(df["date"]).normalize().unique().sort_values()
        for code, df in aemet_by_station.items()
    }


def _measured_calendar(days: pd.DatetimeIndex) -> pd.Series:
    """1/0 per calendar day (1 = measured), from 90 days before the first.

    Parameters
    ----------
    days : pandas.DatetimeIndex
        Days a gauge measured (not empty).

    Returns
    -------
    pandas.Series
        Indexed by every calendar day from 90 days before the first measured
        day to the last one; 1 where the gauge measured, 0 elsewhere.
    """
    calendar = pd.date_range(
        days.min() - pd.Timedelta(days=ANTECEDENT_DAYS), days.max(), freq="D"
    )
    return pd.Series(calendar.isin(days).astype(int), index=calendar)


def antecedent_coverage(
    days: pd.DatetimeIndex, dates: pd.DatetimeIndex
) -> np.ndarray:
    """Number of measured days in the 90 days before each date.

    Parameters
    ----------
    days : pandas.DatetimeIndex
        Days the gauge measured.
    dates : pandas.DatetimeIndex
        Event dates (at midnight). The date itself is not counted.

    Returns
    -------
    numpy.ndarray
        One integer (0..90) per date; 0 for dates outside the gauge's record.
    """
    if len(days) == 0:
        return np.zeros(len(dates), dtype=int)
    measured = _measured_calendar(days)
    # Rolling sum over the 90 days *before* D: window ending at D-1.
    before = measured.rolling(ANTECEDENT_DAYS, min_periods=1).sum().shift(1)
    return before.reindex(dates).fillna(0).astype(int).to_numpy()


def candidate_days(
    days: pd.DatetimeIndex,
    rockfall_dates_at_gauge: pd.DatetimeIndex,
    settings: ControlSettings,
) -> pd.DatetimeIndex:
    """Days allowed as controls at one gauge.

    Rules 1-3 of the module docstring: measured, within the period, enough
    antecedent coverage, and far enough from every rockfall at the gauge.

    Parameters
    ----------
    days : pandas.DatetimeIndex
        Days the gauge measured.
    rockfall_dates_at_gauge : pandas.DatetimeIndex
        Dates of every rockfall linked to the gauge.
    settings : ControlSettings
        Period, coverage and exclusion rules.

    Returns
    -------
    pandas.DatetimeIndex
        The candidate days, sorted.
    """
    candidates = days[
        (days >= settings.period[0]) & (days <= settings.period[1])
    ]

    coverage = antecedent_coverage(days, candidates)
    candidates = candidates[
        coverage >= settings.min_coverage * ANTECEDENT_DAYS
    ]

    if len(rockfall_dates_at_gauge) and len(candidates):
        # Distance in days from each candidate to the nearest rockfall.
        distance = np.abs(
            (
                candidates.to_numpy()[:, None]
                - rockfall_dates_at_gauge.to_numpy()[None, :]
            )
            / np.timedelta64(1, "D")
        ).min(axis=1)
        candidates = candidates[distance > settings.exclusion_days]
    return candidates


def draw_temporal_controls(
    samples: pd.DataFrame,
    days_by_gauge: dict[str, pd.DatetimeIndex],
    skip_codes: list[str],
    settings: ControlSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Control rows for the rockfalls in ``samples``.

    Each control copies every column of its rockfall, then sets ``date``,
    ``is_rockfall`` = F, ``sample_type`` = temporal_control,
    ``source_sample_id`` and its own ``sample_id``
    (``<rockfall id>_c<nn>``). Rockfalls are processed in ``sample_id``
    order with one generator, so the draws are reproducible for a fixed
    seed and input.

    Parameters
    ----------
    samples : pandas.DataFrame
        One row per sample with at least ``date``, ``is_rockfall`` (T/F),
        ``code`` (the gauge), ``sample_id`` and ``sample_type``.
    days_by_gauge : dict of str to pandas.DatetimeIndex
        Days each AEMET gauge measured (``measured_days``).
    skip_codes : list of str
        Gauges whose rockfalls get no controls (the RiscBal gauges).
    settings : ControlSettings
        Controls per rockfall and the candidate-day rules.

    Returns
    -------
    controls : pandas.DataFrame
        The control rows, with the columns of ``samples`` plus
        ``source_sample_id``.
    summary : pandas.DataFrame
        One row per rockfall: ``sample_id``, ``code``, ``date``,
        ``n_candidates`` and ``n_controls``.
    """
    rng = np.random.default_rng(settings.seed)
    rockfalls = samples[samples["sample_type"] == "rockfall"].sort_values(
        "sample_id"
    )
    skipped = rockfalls[rockfalls["code"].isin(skip_codes)]
    if not skipped.empty:
        logger.info(
            "  %d rockfalls at gauges without controls (%s): %s",
            len(skipped),
            ", ".join(sorted(set(skipped["code"]))),
            ", ".join(skipped["sample_id"]),
        )
    rockfalls = rockfalls[~rockfalls["code"].isin(skip_codes)]
    rockfall_dates = {
        code: pd.DatetimeIndex(group["date"]).normalize()
        for code, group in rockfalls.groupby("code")
    }

    controls, summary = [], []
    for _, rockfall in rockfalls.iterrows():
        code = rockfall["code"]
        candidates = candidate_days(
            days_by_gauge.get(code, pd.DatetimeIndex([])),
            rockfall_dates[code],
            settings,
        )
        n = min(settings.per_rockfall, len(candidates))
        chosen = (
            np.sort(rng.choice(candidates.to_numpy(), size=n, replace=False))
            if n
            else []
        )
        summary.append(
            {
                "sample_id": rockfall["sample_id"],
                "code": code,
                "date": rockfall["date"].date(),
                "n_candidates": len(candidates),
                "n_controls": n,
            }
        )
        for number, day in enumerate(chosen, start=1):
            control = rockfall.copy()
            control["date"] = pd.Timestamp(day)
            control["is_rockfall"] = "F"
            control["sample_type"] = "temporal_control"
            control["source_sample_id"] = rockfall["sample_id"]
            control["sample_id"] = f"{rockfall['sample_id']}_c{number:02d}"
            controls.append(control)

    summary = pd.DataFrame(summary)
    short = summary[summary["n_controls"] < settings.per_rockfall]
    if not short.empty:
        logger.warning(
            "%d rockfalls have fewer than %d candidate days:\n%s",
            len(short),
            settings.per_rockfall,
            short.to_string(index=False),
        )
    columns = list(dict.fromkeys(list(samples.columns) + ["source_sample_id"]))
    return pd.DataFrame(controls, columns=columns), summary


def rain_days_90d(
    samples: pd.DataFrame, days_by_gauge: dict[str, pd.DatetimeIndex]
) -> pd.Series:
    """Measured days in each sample's 90-day antecedent window.

    Metadata, not a feature: a low value means the antecedent sums are partly
    "missing counted as 0 mm" (e.g. the rows with 0 mm in every window).

    Parameters
    ----------
    samples : pandas.DataFrame
        All samples, with ``code`` (the gauge) and ``date``.
    days_by_gauge : dict of str to pandas.DatetimeIndex
        Days each AEMET gauge measured.

    Returns
    -------
    pandas.Series
        0..90 per sample, aligned with ``samples``. NaN for gauges not in
        ``days_by_gauge`` (the RiscBal gauges): unknown, not zero.
    """
    result = pd.Series(np.nan, index=samples.index, dtype=float)
    for code, group in samples.groupby("code"):
        if code not in days_by_gauge:
            continue
        dates = pd.DatetimeIndex(group["date"]).normalize()
        result.loc[group.index] = antecedent_coverage(
            days_by_gauge[code], dates
        )
    return result


def with_own_source_ids(samples: pd.DataFrame) -> pd.DataFrame:
    """Set ``source_sample_id``: every sample is, so far, its own source.

    Controls added later point to their rockfall instead. The column exists
    whether or not controls are drawn: the modelling repo expects it.

    Parameters
    ----------
    samples : pandas.DataFrame
        Rockfalls and random points, with ``sample_id``.

    Returns
    -------
    pandas.DataFrame
        The same samples with ``source_sample_id`` = ``sample_id``.
    """
    samples["source_sample_id"] = samples["sample_id"]
    return samples


def add_temporal_controls(
    samples: pd.DataFrame,
    days_by_gauge: dict[str, pd.DatetimeIndex],
    skip_codes: list[str],
    settings: ControlSettings,
    output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw the controls of every rockfall and append them to the samples.

    Writes ``temporal_controls_summary.csv`` to the run's outputs folder.

    Parameters
    ----------
    samples : pandas.DataFrame
        Rockfalls and random points, with ``source_sample_id``.
    days_by_gauge : dict of str to pandas.DatetimeIndex
        Days each AEMET gauge measured (``measured_days``).
    skip_codes : list of str
        Gauges whose rockfalls get no controls (the RiscBal gauges).
    settings : ControlSettings
        Controls per rockfall, exclusion, coverage, period and seed.
    output_path : pathlib.Path
        The run's outputs folder (``runs/<run_id>/outputs/``).

    Returns
    -------
    samples : pandas.DataFrame
        The samples plus the controls.
    summary : pandas.DataFrame
        Candidates and controls per rockfall.
    """
    logger.info("Drawing temporal controls:")
    controls, summary = draw_temporal_controls(
        samples, days_by_gauge, skip_codes, settings
    )
    summary.to_csv(output_path / "temporal_controls_summary.csv", index=False)
    return pd.concat([samples, controls], ignore_index=True), summary
