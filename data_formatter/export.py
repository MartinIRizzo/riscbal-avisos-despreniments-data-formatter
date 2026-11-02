"""Write the modelling dataset, ``final_gdf.csv``."""

from pathlib import Path

import pandas as pd

from data_formatter.eda import dataframe_analysis


def export_dataset(samples: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Write the modelling dataset, ``final_gdf.csv``.

    The target becomes 1/0, the EDA plots are drawn, and the slope and
    elevation columns are dropped (they overlap too much with each other;
    see eda.py and vault D-005).

    Parameters
    ----------
    samples : pandas.DataFrame
        All samples with rain and susceptibility.
    output_path : pathlib.Path
        The run's outputs folder (``runs/<run_id>/outputs/``).

    Returns
    -------
    pandas.DataFrame
        The exported dataset.
    """
    samples["is_rockfall"] = samples["is_rockfall"].map({"T": 1, "F": 0})
    dataframe_analysis(samples, output_path)
    slope = list(samples.columns[samples.columns.str.contains("slope")])
    elevation = list(samples.columns[samples.columns.str.contains("elev")])
    final = samples.drop(columns=slope + elevation)
    final.to_csv(output_path / "final_gdf.csv", index=False)
    return final
