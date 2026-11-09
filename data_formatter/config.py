"""Settings of a formatter run, read from ``config.yaml``.

``Config.from_yaml`` builds one frozen ``Config`` from the file. The values
and what they mean are documented in ``config.yaml`` itself; this module only
gives them types. Each optional step (negative shift, temporal controls,
fallback gauges, cost distance) has an ``enabled`` flag in its own section. A key missing
from the file, or one that isn't expected, raises a ``TypeError`` naming
it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from data_formatter.controls import ControlSettings
from data_formatter.cost_distance import CostDistanceSettings
from data_formatter.gauge_fallback import FallbackSettings

DEFAULT_CONFIG_FILE = Path("config.yaml")


@dataclass(frozen=True)
class GaugeReassignment:
    """Samples of a municipality dated in (after, until] use this gauge.

    Attributes
    ----------
    municipality : str
        Municipality of the samples (``municipality`` column).
    after : str
        ISO date; samples dated after it (exclusive).
    until : str
        ISO date; samples dated up to it (inclusive).
    gauge : str
        Gauge code they use instead of their AEMET station.
    """

    municipality: str
    after: str
    until: str
    gauge: str


@dataclass(frozen=True)
class NegativeShiftSettings:
    """The original negative sampling (``negative_shift`` in config.yaml).

    Attributes
    ----------
    enabled : bool
        Feature flag, checked by the orchestrator (main.py).
    share : float
        Share of the random points moved to a high-rainfall day.
    percentile : float
        A high-rainfall day has rain >= this quantile of its station.
    period : tuple of str
        First and last day (ISO dates) the new days are drawn from.
    seed : int
        Seed of the random generator.
    """

    enabled: bool
    share: float
    percentile: float
    period: tuple[str, str]
    seed: int


@dataclass(frozen=True)
class Config:
    """All the settings of one formatter run.

    Attributes
    ----------
    data_path : pathlib.Path
        Input folder.
    runs_path : pathlib.Path
        Folder of the run histories; each run writes all its outputs to
        ``runs_path/<run_id>/outputs/``.
    riscbal_stations : tuple of str
        Gauges read from the RiscBal database instead of the AEMET file.
    pluviometric_day_offset : pandas.Timedelta
        Start of the pluviometric day (07:00 UTC), for the RiscBal gauges.
    gauge_reassignments : tuple of GaugeReassignment
        Hand-made gauge changes (Esporles).
    negative_shift : NegativeShiftSettings
        The original negative sampling.
    controls : ControlSettings
        Temporal controls.
    fallback : FallbackSettings
        Replacement gauges and exclusions.
    cost_distance : CostDistanceSettings
        Slope-weighted cost distance to the gauges.
    """

    data_path: Path
    runs_path: Path
    riscbal_stations: tuple[str, ...]
    pluviometric_day_offset: pd.Timedelta
    gauge_reassignments: tuple[GaugeReassignment, ...]
    negative_shift: NegativeShiftSettings
    controls: ControlSettings
    fallback: FallbackSettings
    cost_distance: CostDistanceSettings

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        """Read the settings of a run from a YAML file.

        Parameters
        ----------
        path : pathlib.Path
            The YAML file (normally ``config.yaml`` at the repo root).

        Returns
        -------
        Config
            The settings, with periods as tuples and the pluviometric day
            offset as a ``pandas.Timedelta``.
        """
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        gauges = raw["gauges"]
        shift = raw["negative_shift"]
        controls = raw["temporal_controls"]
        fallback = raw["fallback"]
        cost_distance = raw["cost_distance"]
        return cls(
            data_path=Path(raw["paths"]["data"]),
            runs_path=Path(raw["paths"]["runs"]),
            riscbal_stations=tuple(gauges["riscbal_stations"]),
            pluviometric_day_offset=pd.Timedelta(
                hours=gauges["pluviometric_day_offset_hours"]
            ),
            gauge_reassignments=tuple(
                GaugeReassignment(**item) for item in gauges["reassignments"]
            ),
            negative_shift=NegativeShiftSettings(
                **{**shift, "period": tuple(shift["period"])}
            ),
            controls=ControlSettings(
                **{**controls, "period": tuple(controls["period"])}
            ),
            fallback=FallbackSettings(
                **{
                    **fallback,
                    "correlation_period": tuple(
                        fallback["correlation_period"]
                    ),
                }
            ),
            cost_distance=CostDistanceSettings(
                **{
                    **cost_distance,
                    "dem": Path(cost_distance["dem"]),
                    "cost": Path(cost_distance["cost"]),
                }
            ),
        )
