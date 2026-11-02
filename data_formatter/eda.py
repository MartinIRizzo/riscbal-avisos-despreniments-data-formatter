"""Exploratory checks of the final dataset (plots and logged summaries).

Run by the export step. The correlation heatmaps show why the slope and
elevation columns are dropped from ``final_gdf.csv`` (they overlap too much
with each other; vault D-005).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def dataframe_analysis(df: pd.DataFrame, output_path: Path) -> None:
    """Exploratory checks of the final table: plots and logged summaries.

    Saves three correlation heatmaps to ``output_path``
    (``precipitation_correlation.png``, ``slope_correlation.png``,
    ``elevation_correlation.png``) and logs the correlation of every numeric
    column with ``is_rockfall`` and the rockfall rate by hardness,
    susceptibility and hillside. The slope and elevation heatmaps are why
    those columns are dropped from the export (they overlap too much).

    Parameters
    ----------
    df : pandas.DataFrame
        The final samples, with ``is_rockfall`` as 1/0, the four rain
        windows, the ``slope_*`` and ``elev_*`` columns, ``hardness``,
        ``susceptibility`` and ``hillside``.
    output_path : pathlib.Path
        The run's outputs folder (``runs/<run_id>/outputs/``).
    """
    precip_cols = [
        "event_day_acc",
        "last_week_acc",
        "last_month_acc",
        "last_three_months_acc",
    ]
    slope_cols = [c for c in df.columns if c.startswith("slope_")]
    elev_cols = [c for c in df.columns if c.startswith("elev_")]

    plt.figure()
    sns.heatmap(df[precip_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Acc precipitation correlation")
    plt.tight_layout()
    plt.savefig(output_path / "precipitation_correlation.png")

    plt.figure()
    sns.heatmap(df[slope_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Slope correlation")
    plt.tight_layout()
    plt.savefig(output_path / "slope_correlation.png")

    plt.figure()
    sns.heatmap(df[elev_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Elevation correlation")
    plt.tight_layout()
    plt.savefig(output_path / "elevation_correlation.png")

    corr_cols = [
        c
        for c in df.select_dtypes(include=np.number).columns
        if c not in ["POINT_X", "POINT_Y"]
    ]
    corr_with_target = (
        df[corr_cols].corr()["is_rockfall"].sort_values(ascending=False)
    )
    logger.info(
        "Correlation with is_rockfall:\n%s", corr_with_target.to_string()
    )

    logger.info(
        "Rockfall rate by hardness:\n%s",
        df.groupby("hardness")["is_rockfall"]
        .agg(["mean", "count"])
        .to_string(),
    )
    logger.info(
        "Rockfall rate by susceptibility:\n%s",
        df.groupby("susceptibility")["is_rockfall"]
        .agg(["mean", "count"])
        .to_string(),
    )

    logger.info(
        "is_rockfall share by hillside:\n%s",
        pd.crosstab(
            df["hillside"], df["is_rockfall"], normalize="index"
        ).to_string(),
    )
