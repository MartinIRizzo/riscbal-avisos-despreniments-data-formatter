"""The original negative sampling: move random points to high-rainfall days.

``share`` of the random points get a random day with rain >= their station's
``percentile``, within ``period``. Off since Phase B0 (2026-10-17): the moved
negatives were much wetter than the rockfalls and every model learned "more
rain -> less risk" (vault: D-008). Kept behind the feature flag
``negative_shift.enabled`` in ``config.yaml``.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data_formatter.config import NegativeShiftSettings

logger = logging.getLogger(__name__)

SHIFTED_COLUMNS = [
    "POINT_X",
    "POINT_Y",
    "code",
    "original_date",
    "shifted_date",
]


def shift_negatives(
    samples: pd.DataFrame,
    aemet: pd.DataFrame,
    settings: NegativeShiftSettings,
    output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move some random points to high-rainfall days at their station.

    Writes ``shifted_negatives.csv`` to the run's outputs folder.

    Parameters
    ----------
    samples : pandas.DataFrame
        The samples; only ``random_point`` rows are moved.
    aemet : pandas.DataFrame
        The AEMET daily rain.
    settings : NegativeShiftSettings
        Share, percentile, period and seed.
    output_path : pathlib.Path
        The run's outputs folder (``runs/<run_id>/outputs/``).

    Returns
    -------
    samples : pandas.DataFrame
        The samples, with the moved dates.
    shifted : pandas.DataFrame
        One row per negative actually moved: where, which station, old and
        new date.
    """
    percentiles = aemet.groupby("code")["precipitation"].quantile(
        settings.percentile
    )
    in_period = aemet[
        (aemet["date"] >= settings.period[0])
        & (aemet["date"] <= settings.period[1])
    ].copy()
    high_rain_days = in_period[
        in_period["precipitation"] >= in_period["code"].map(percentiles)
    ]
    high_rain_days_by_station = (
        high_rain_days.groupby("code")["date"]
        .apply(lambda dates: dates.tolist())
        .to_dict()
    )

    rng = np.random.default_rng(settings.seed)
    # Only the random points: rockfalls and controls are never moved.
    random_points = samples.loc[samples["sample_type"] == "random_point"].index
    n_replace = int(len(random_points) * settings.share)
    to_move = rng.choice(random_points, size=n_replace, replace=False)
    shifted_rows = []
    for index in to_move:
        station_code = samples.at[index, "code"]
        available_dates = high_rain_days_by_station.get(station_code, [])
        if available_dates:
            old_date = samples.at[index, "date"]
            samples.at[index, "date"] = rng.choice(available_dates)
            shifted_rows.append(
                {
                    "POINT_X": samples.at[index, "POINT_X"],
                    "POINT_Y": samples.at[index, "POINT_Y"],
                    "code": station_code,
                    "original_date": old_date,
                    "shifted_date": samples.at[index, "date"],
                }
            )
    shifted = pd.DataFrame(shifted_rows, columns=SHIFTED_COLUMNS)
    shifted.to_csv(output_path / "shifted_negatives.csv", index=False)
    logger.info("Negatives shifted to high-rainfall days: %d", len(shifted))
    return samples, shifted
