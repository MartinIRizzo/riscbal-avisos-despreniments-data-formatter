"""Susceptibility class of each sample, from the official map.

The Spanish Government's landslide susceptibility map
(``vectorial_gravitacionals_INES.shp``): each sample gets the class of the
polygon it falls in, or none outside the polygons.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

# Coordinates of the samples: UTM zone 31N.
SAMPLES_CRS = "EPSG:32631"
SUSCEPTIBILITY_FILE = (
    Path("susceptibility") / "vectorial_gravitacionals_INES.shp"
)


def add_susceptibility(
    samples: pd.DataFrame, data_path: Path, output_path: Path
) -> pd.DataFrame:
    """Add the susceptibility class of the polygon each sample falls in.

    Writes ``samples_with_susceptibility.csv`` to the run's outputs folder.

    Parameters
    ----------
    samples : pandas.DataFrame
        All samples, with ``POINT_X`` and ``POINT_Y`` (UTM 31N).
    data_path : pathlib.Path
        Input folder (holds the susceptibility shapefile).
    output_path : pathlib.Path
        The run's outputs folder (``runs/<run_id>/outputs/``).

    Returns
    -------
    pandas.DataFrame
        The samples with ``susceptibility`` (empty outside the polygons),
        and ``code`` renamed to ``rain_station``.
    """
    susceptibility = gpd.read_file(data_path / SUSCEPTIBILITY_FILE)
    points = gpd.GeoDataFrame(
        samples,
        geometry=gpd.points_from_xy(samples["POINT_X"], samples["POINT_Y"]),
        crs=SAMPLES_CRS,
    )
    susceptibility = susceptibility.to_crs(points.crs)
    joined = gpd.sjoin(
        left_df=points,
        right_df=susceptibility,
        how="left",
        predicate="within",
    )
    joined = joined.drop(columns=["index_right", "geometry", "fid", "Value"])
    # `code` is the gauge actually used for the rain: metadata, never a
    # feature.
    joined = joined.rename(
        columns={"POTENCIALI": "susceptibility", "code": "rain_station"}
    )
    joined.to_csv(output_path / "samples_with_susceptibility.csv", index=False)
    return joined
