"""Check our cost-distance allocation against the workmate's ArcGIS one.

Phase 1 of the vault note "Cost-distance Thiessen polygons" (decision
D-027): data_formatter/cost_distance.py should reproduce ArcGIS's Distance
Allocation (``DistAll_slope_v4``), whose polygons gave every sample its
``id_station``. Diagnostic only: it doesn't change the pipeline. Run from
the repo root::

    uv run python -m scripts.validate_cost_allocation [--neighbours 16]

Inputs: the terrain rasters of ``cost_distance`` in config.yaml, the
reference allocation data/terrain/DistAll_slope_v4.tif (255 = NoData),
data/stations.csv, and the points of data/negatives.csv and
data/event_stations.csv with their ``id_station``. It reads no formatter
run, so it writes outside runs/ (the result doesn't depend on any run).

Outputs, in ``analyses/validate_cost_allocation/neighbours_<n>/`` (repo
root, gitignored):
    summary.json       cell and point agreement, timings, memory, settings
    points.csv         every point: id_station, our gauge, and for the
                       disagreeing ones the cost to both gauges
    cell_swaps.csv     the most common (ArcGIS gauge, our gauge) pairs
                       among the disagreeing cells
    allocation.tif     our allocation (gauge id; 255 = NoData)
    allocation.gpkg    the same as polygons, one per gauge (layer
                       ``allocation``, field ``id_station``), to draw over
                       Thiessen_slope_v4.gpkg
    disagreement.tif   1 where ours differs from ArcGIS, 0 where equal,
                       255 where either has no gauge (for QGIS)

How to read it: the points matter most (they decide the rain each sample
reads). Disagreements should be few and near a boundary (``cost_gap``
small: the two gauges cost almost the same). On the cells, thin lines
along the boundaries are expected (ArcGIS's propagation isn't plain
8-neighbour Dijkstra); patches mean an error in the cost.
"""

import argparse
import json
import resource
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from scipy.sparse import csr_matrix

from data_formatter import cost_distance, inventory
from data_formatter.config import DEFAULT_CONFIG_FILE, Config

REFERENCE_ALLOCATION = Path("terrain") / "DistAll_slope_v4.tif"
# Not under runs/: the validation reads terrain and sample files only.
OUTPUT_PATH = Path("analyses") / "validate_cost_allocation"
# NoData of the byte rasters written here (and of the reference).
BYTE_NODATA = 255
DIFFERENT, EQUAL = 1, 0
POINTS_CRS = "EPSG:32631"
# Number of (ArcGIS gauge, our gauge) pairs listed in cell_swaps.csv.
TOP_SWAPS = 20
KILOBYTES_PER_MEGABYTE = 1024


def _peak_memory_mb() -> float:
    """Peak resident memory of this process so far (Linux: ru_maxrss in KB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / KILOBYTES_PER_MEGABYTE


def _timed(step: str, timings: dict[str, float], started: float) -> float:
    """Record and print how long a step took; returns the new start time."""
    now = time.perf_counter()
    timings[step] = round(now - started, 1)
    print(
        f"{step}: {timings[step]} s (peak memory {_peak_memory_mb():.0f} MB)"
    )
    return now


def load_points(data_path: Path) -> pd.DataFrame:
    """The sample points with the workmate's ``id_station``.

    Parameters
    ----------
    data_path : pathlib.Path
        The data folder.

    Returns
    -------
    pandas.DataFrame
        ``source`` (negatives / event_stations), ``POINT_X``, ``POINT_Y``
        (EPSG:32631) and ``id_station``.
    """
    files = {
        "negatives": inventory.NEGATIVES_FILE,
        "event_stations": inventory.EVENT_STATIONS_FILE,
    }
    return pd.concat(
        [
            pd.read_csv(
                data_path / name, sep=";", decimal=",", encoding="utf-8-sig"
            )[["POINT_X", "POINT_Y", "id_station"]].assign(source=source)
            for source, name in files.items()
        ],
        ignore_index=True,
    )


def write_byte_raster(
    path: Path, grid: np.ndarray, terrain: cost_distance.Terrain
) -> None:
    """Write a byte GeoTIFF on the terrain grid (255 = NoData).

    Parameters
    ----------
    path : pathlib.Path
        Output file.
    grid : numpy.ndarray
        Values 0-254, or ``BYTE_NODATA``.
    terrain : cost_distance.Terrain
        Gives the grid's georeference.
    """
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.shape[0],
        width=grid.shape[1],
        count=1,
        dtype="uint8",
        crs=terrain.crs,
        transform=terrain.transform,
        nodata=BYTE_NODATA,
        compress="deflate",
        tiled=True,
    ) as output:
        output.write(grid.astype(np.uint8), 1)


def allocation_polygons(
    grid: np.ndarray, terrain: cost_distance.Terrain
) -> gpd.GeoDataFrame:
    """The allocation as polygons, one (multi)polygon per gauge.

    Parameters
    ----------
    grid : numpy.ndarray
        Gauge id per cell, ``BYTE_NODATA`` where there is none.
    terrain : cost_distance.Terrain
        Gives the grid's georeference.

    Returns
    -------
    geopandas.GeoDataFrame
        ``id_station`` and geometry, in the grid's CRS (EPSG:25831).
    """
    pieces = shapes(
        grid.astype(np.uint8),
        mask=grid != BYTE_NODATA,
        transform=terrain.transform,
    )
    cells = gpd.GeoDataFrame.from_features(
        [
            {"geometry": geometry, "properties": {"id_station": int(value)}}
            for geometry, value in pieces
        ],
        crs=terrain.crs,
    )
    return cells.dissolve(by="id_station").reset_index()


def compare_cells(
    ours: np.ndarray, reference: np.ndarray
) -> tuple[dict[str, object], np.ndarray, pd.DataFrame]:
    """Cell-by-cell agreement of two allocation grids.

    Parameters
    ----------
    ours, reference : numpy.ndarray
        Gauge id per cell; ``BYTE_NODATA`` where there is none.

    Returns
    -------
    summary : dict
        Cells allocated by both, the share equal, and the cells only one
        of them allocates.
    disagreement : numpy.ndarray
        ``EQUAL`` / ``DIFFERENT`` per cell, ``BYTE_NODATA`` elsewhere.
    swaps : pandas.DataFrame
        The most common (reference, ours) pairs among different cells.
    """
    ours_set, reference_set = ours != BYTE_NODATA, reference != BYTE_NODATA
    both = ours_set & reference_set
    different = both & (ours != reference)
    disagreement = np.full(ours.shape, BYTE_NODATA, dtype=np.uint8)
    disagreement[both] = EQUAL
    disagreement[different] = DIFFERENT
    swaps = (
        pd.DataFrame(
            {
                "arcgis_gauge": reference[different],
                "our_gauge": ours[different],
            }
        )
        .value_counts()
        .head(TOP_SWAPS)
        .rename("cells")
        .reset_index()
    )
    summary = {
        "cells_both": int(both.sum()),
        "cells_equal_share": round(float(1 - different.sum() / both.sum()), 5),
        "cells_different": int(different.sum()),
        "cells_only_ours": int((ours_set & ~reference_set).sum()),
        "cells_only_arcgis": int((reference_set & ~ours_set).sum()),
    }
    return summary, disagreement, swaps


def compare_points(
    points: pd.DataFrame,
    terrain: cost_distance.Terrain,
    graph: csr_matrix,
    gauge_of_node: np.ndarray,
    gauge_nodes: pd.Series,
) -> pd.DataFrame:
    """Our gauge at each sample point vs its ``id_station``.

    For the points that disagree, also the cost to both gauges: a small
    ``cost_gap`` means the point is near the boundary between them.

    Parameters
    ----------
    points : pandas.DataFrame
        From ``load_points``.
    terrain : cost_distance.Terrain
        From ``load_terrain``.
    graph : scipy.sparse.csr_matrix
        From ``build_graph``.
    gauge_of_node : numpy.ndarray
        Our allocation, gauge id per node.
    gauge_nodes : pandas.Series
        Node of each gauge, indexed by gauge id.

    Returns
    -------
    pandas.DataFrame
        The points with ``node``, ``our_gauge``, ``agrees`` and, where they
        disagree, ``cost_arcgis_gauge``, ``cost_our_gauge`` and
        ``cost_gap`` (relative: (arcgis - ours) / ours).
    """
    points = points.copy()
    points["node"] = cost_distance.cells_of_points(
        terrain,
        points["POINT_X"].to_numpy(),
        points["POINT_Y"].to_numpy(),
        POINTS_CRS,
    )
    on_land = points["node"] != cost_distance.NO_NODE
    points["our_gauge"] = cost_distance.NO_GAUGE
    points.loc[on_land, "our_gauge"] = gauge_of_node[
        points.loc[on_land, "node"]
    ]
    points["agrees"] = points["our_gauge"] == points["id_station"]

    disagreeing = points[~points["agrees"] & on_land]
    if not disagreeing.empty:
        # Undirected graph: the cost from a point to a gauge is the cost
        # from the gauge to the point, so one Dijkstra per point suffices.
        costs = cost_distance.costs_between(
            graph,
            disagreeing["node"].to_numpy(),
            gauge_nodes.to_numpy(),
        )
        column_of = {gauge: i for i, gauge in enumerate(gauge_nodes.index)}
        rows = np.arange(len(disagreeing))
        arcgis = costs[rows, disagreeing["id_station"].map(column_of)]
        ours = costs[rows, disagreeing["our_gauge"].map(column_of)]
        points.loc[disagreeing.index, "cost_arcgis_gauge"] = arcgis
        points.loc[disagreeing.index, "cost_our_gauge"] = ours
        points.loc[disagreeing.index, "cost_gap"] = (arcgis - ours) / ours
    return points


def main() -> None:
    """Run the validation and write its outputs (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--neighbours",
        type=int,
        choices=sorted(cost_distance.HALF_NEIGHBOURHOODS),
        help="override cost_distance.neighbours of config.yaml",
    )
    args = parser.parse_args()
    config = Config.from_yaml(DEFAULT_CONFIG_FILE)
    settings = config.cost_distance
    neighbours = args.neighbours or settings.neighbours
    # One subfolder per neighbourhood, so the 8 and 16 results can be
    # compared side by side.
    output = OUTPUT_PATH / f"neighbours_{neighbours}"
    output.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    started = time.perf_counter()
    terrain = cost_distance.load_terrain(settings, config.data_path)
    started = _timed("load terrain", timings, started)
    graph = cost_distance.build_graph(terrain, neighbours)
    started = _timed("build graph", timings, started)

    stations = pd.read_csv(config.data_path / "stations.csv")
    gauge_nodes = pd.Series(
        cost_distance.cells_of_points(
            terrain,
            stations["utm_x"].to_numpy(),
            stations["utm_y"].to_numpy(),
            POINTS_CRS,
        ),
        index=stations["id"],
    )
    gauge_of_node, _ = cost_distance.allocate(
        graph, gauge_nodes.to_numpy(), gauge_nodes.index.to_numpy()
    )
    started = _timed("allocate", timings, started)

    ours = cost_distance.to_grid(
        terrain,
        np.where(
            gauge_of_node == cost_distance.NO_GAUGE, BYTE_NODATA, gauge_of_node
        ),
        BYTE_NODATA,
    )
    with rasterio.open(config.data_path / REFERENCE_ALLOCATION) as reference:
        arcgis = reference.read(1)
    cell_summary, disagreement, swaps = compare_cells(ours, arcgis)
    write_byte_raster(output / "allocation.tif", ours, terrain)
    allocation_polygons(ours, terrain).to_file(
        output / "allocation.gpkg", layer="allocation"
    )
    write_byte_raster(output / "disagreement.tif", disagreement, terrain)
    swaps.to_csv(output / "cell_swaps.csv", index=False)

    points = compare_points(
        load_points(config.data_path),
        terrain,
        graph,
        gauge_of_node,
        gauge_nodes,
    )
    points.to_csv(output / "points.csv", index=False)
    started = _timed("compare", timings, started)

    summary = {
        "neighbours": neighbours,
        "dem": str(settings.dem),
        "cost": str(settings.cost),
        "reference": str(REFERENCE_ALLOCATION),
        "land_cells": terrain.node_count,
        "graph_edges": int(graph.nnz),
        **cell_summary,
        "points": len(points),
        "points_equal": int(points["agrees"].sum()),
        "points_different": points.loc[
            ~points["agrees"],
            ["source", "POINT_X", "POINT_Y", "id_station", "our_gauge"],
        ].to_dict("records"),
        "timings_s": timings,
        "peak_memory_mb": round(_peak_memory_mb()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=int), encoding="utf-8"
    )

    print(
        f"cells: {cell_summary['cells_equal_share']:.2%} equal of "
        f"{cell_summary['cells_both']} allocated by both"
    )
    print(f"points: {summary['points_equal']} of {len(points)} equal")
    if summary["points_different"]:
        print(
            points.loc[
                ~points["agrees"],
                ["source", "id_station", "our_gauge", "cost_gap"],
            ].to_string()
        )
    print(f"written to {output}")


if __name__ == "__main__":
    main()
