"""Read the formatter's inputs: gauges, samples and the AEMET daily rain."""

from pathlib import Path

import pandas as pd

from data_formatter.aemet_formatter import format_aemet_data
from data_formatter.config import Config
from data_formatter.inventory import SAMPLES_DATE_FORMAT, build_samples

# The reshaped AEMET table, gzipped (pandas infers it from the extension).
AEMET_TABLE_FILE = "formatted_historical_aemet_data.csv.gz"

# Columns of the inventory renamed to the names used downstream.
SAMPLE_COLUMNS = {
    "Fecha": "date",
    "Tipus": "is_rockfall",
    "Municipio": "municipality",
    "Vertiente": "hillside",
    "DURESA": "hardness",
}


def load_stations(config: Config) -> pd.DataFrame:
    """The rain gauges of ``stations.csv``.

    Parameters
    ----------
    config : Config
        Run settings (paths).

    Returns
    -------
    pandas.DataFrame
        One row per gauge: ``id`` (the ``id_station`` of the samples),
        ``code``, ``utm_x``, ``utm_y``.
    """
    return pd.read_csv(config.data_path / "stations.csv")


def load_samples(
    config: Config, stations: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    """Rockfalls and random points, each with its gauge.

    Writes ``samples_with_stations.csv`` to the run's outputs folder.

    Parameters
    ----------
    config : Config
        Run settings (paths, gauge reassignments).
    stations : pandas.DataFrame
        From ``load_stations``.
    output_path : pathlib.Path
        The run's outputs folder (``runs/<run_id>/outputs/``).

    Returns
    -------
    pandas.DataFrame
        One row per sample, with ``code`` (its gauge) and a parsed ``date``.
    """
    # Rockfalls from the original inventory (correct day-first dates) and
    # recent_rockfalls.csv, negatives from negatives.csv. See inventory.py.
    samples = build_samples(config.data_path, output_path)

    # Each sample's gauge comes from the workmate's slope-weighted Thiessen
    # polygons (id_station, from event_stations.csv and negatives.csv), not
    # from the nearest gauge.
    samples = pd.merge(
        samples, stations, left_on="id_station", right_on="id", how="inner"
    )
    samples = samples.drop(columns=["utm_x", "utm_y", "id"])
    samples = samples.rename(columns=SAMPLE_COLUMNS)
    # Explicit format: a date in any other format fails loudly instead of
    # being silently re-interpreted (8/1/2009 would become August 1st).
    samples["date"] = pd.to_datetime(
        samples["date"], format=SAMPLES_DATE_FORMAT
    )
    for reassignment in config.gauge_reassignments:
        samples.loc[
            (samples["municipality"] == reassignment.municipality)
            & (samples["date"] > reassignment.after)
            & (samples["date"] <= reassignment.until),
            "code",
        ] = reassignment.gauge
    samples["hardness"] = samples["hardness"].replace(
        {"Moderately": "Moderately hard rocks"}
    )
    samples.to_csv(output_path / "samples_with_stations.csv", index=False)
    return samples


def load_aemet(config: Config, output_path: Path) -> pd.DataFrame:
    """The AEMET daily rain, one row per gauge and day.

    Writes ``formatted_historical_aemet_data.csv.gz`` (gzipped: ~200 MB
    uncompressed) to the run's outputs folder.

    Parameters
    ----------
    config : Config
        Run settings (paths).
    output_path : pathlib.Path
        The run's outputs folder (``runs/<run_id>/outputs/``).

    Returns
    -------
    pandas.DataFrame
        As returned by ``format_aemet_data``.
    """
    raw = pd.read_csv(config.data_path / "historical_aemet_data.csv", sep=";")
    aemet = format_aemet_data(raw)
    aemet.to_csv(
        output_path / AEMET_TABLE_FILE,
        index=False,
    )
    return aemet
