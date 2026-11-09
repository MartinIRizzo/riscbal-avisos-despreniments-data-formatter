"""Slope-weighted cost distance between places and rain gauges.

Reproduces the workmate's slope-weighted Thiessen polygons (ArcGIS
Distance Allocation, ``DistAll_slope_v4`` in their project; vault D-027,
"Cost-distance Thiessen polygons"), so the formatter can measure "how far"
a sample is from each gauge the same way the polygons did.

Method, as in ArcGIS:
  - The terrain is a grid of 20 m cells. Each land cell (the DEM has a
    value) is a node of a graph; sea cells (DEM NoData) are left out, so
    no path can cross the sea. The slope raster covers the sea too, so
    land always comes from the DEM.
  - Each cell is joined to its neighbours (8: the grid moves; or 16: plus
    the knight moves). A step costs the distance over the terrain between
    the two cells (ArcGIS's "surface raster": the horizontal distance and
    the height difference, by Pythagoras) times the mean slope of the two
    cells in degrees (the "cost raster"). Climbing a steep slope is
    therefore much dearer than crossing flat land, and polygon boundaries
    follow the ridges.
  - The accumulated cost from a gauge to a cell is the cheapest sum of
    steps between them (Dijkstra's algorithm). The **allocation** gives
    every cell the gauge with the lowest accumulated cost: that is the
    slope-weighted Thiessen polygon it falls in.

The graph is terrain only, so it never depends on the date. Which gauges
had data on a date is a separate question: a gauge's cost to a place
doesn't depend on the other gauges (they aren't barriers), so the cheapest
gauge among those with data is the same as the allocation recomputed with
only those gauges (see gauge_fallback.py).

Grid and coordinates: the rasters keep their own CRS (EPSG:25831, ETRS89 /
UTM 31N) and are never resampled; points (EPSG:32631, WGS 84 / UTM 31N,
less than 1 m apart) are transformed to it instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

logger = logging.getLogger(__name__)

# Node number of a cell that isn't in the graph (sea, or outside the grid).
NO_NODE = -1
# Gauge id of a cell no gauge reaches (and of the sea), in allocations.
NO_GAUGE = -1
# Half of each neighbourhood, as (row, column) offsets: every pair of
# neighbours is listed once, since a step costs the same both ways.
HALF_NEIGHBOURHOODS = {
    8: ((0, 1), (1, -1), (1, 0), (1, 1)),
    16: ((0, 1), (1, -1), (1, 0), (1, 1), (1, -2), (1, 2), (2, -1), (2, 1)),
}
# scipy may treat a stored 0 in a sparse graph as "no edge", so a step
# between two 0-degree cells (94 land cells in Slope_DEM20) gets this
# negligible cost instead (metres x degrees).
MIN_EDGE_COST = 1e-9
# What scipy's Dijkstra returns as the source of a node no source reaches.
UNREACHED_SOURCE = -9999


@dataclass(frozen=True)
class CostDistanceSettings:
    """Cost-distance terrain (``cost_distance`` in config.yaml).

    Attributes
    ----------
    enabled : bool
        Feature flag, checked by the orchestrator (main.py), not here.
    dem : pathlib.Path
        Elevation raster (surface), relative to the data folder; its
        NoData cells (the sea) can't be crossed.
    cost : pathlib.Path
        Cost per metre raster (slope in degrees), relative to the data
        folder; same grid as ``dem``.
    neighbours : int
        Cells joined to each cell: 8 or 16.
    """

    enabled: bool
    dem: Path
    cost: Path
    neighbours: int


@dataclass(frozen=True)
class Terrain:
    """The terrain grid the graph is built on.

    Attributes
    ----------
    elevation : numpy.ndarray
        Elevation in metres per cell (NaN on the sea).
    slope : numpy.ndarray
        Slope in degrees per cell (the cost per metre).
    node : numpy.ndarray
        Node number of each land cell, 0..N-1 in row order; ``NO_NODE`` on
        the sea.
    transform : affine.Affine
        Grid georeference (cell -> coordinates).
    crs : str
        The grid's CRS (``EPSG:25831``).
    """

    elevation: np.ndarray
    slope: np.ndarray
    node: np.ndarray
    transform: Affine
    crs: str

    @property
    def land(self) -> np.ndarray:
        """Boolean grid: True on the cells that are nodes of the graph."""
        return self.node != NO_NODE

    @property
    def node_count(self) -> int:
        """Number of land cells (nodes of the graph)."""
        return int(self.land.sum())


def load_terrain(settings: CostDistanceSettings, data_path: Path) -> Terrain:
    """Read the DEM and the slope raster and number the land cells.

    Parameters
    ----------
    settings : CostDistanceSettings
        Paths of the two rasters.
    data_path : pathlib.Path
        The data folder the paths are relative to.

    Returns
    -------
    Terrain
        Elevation, slope, node numbers and georeference.

    Raises
    ------
    ValueError
        If the two rasters aren't on the same grid.
    """
    with rasterio.open(data_path / settings.dem) as dem_file:
        dem = dem_file.read(1, masked=True)
        transform, crs = dem_file.transform, dem_file.crs.to_string()
    with rasterio.open(data_path / settings.cost) as cost_file:
        slope = cost_file.read(1).astype(np.float64)
        same_grid = (
            cost_file.transform == transform and cost_file.shape == dem.shape
        )
    if not same_grid:
        message = f"{settings.dem} and {settings.cost} are on different grids"
        raise ValueError(message)

    land = ~np.ma.getmaskarray(dem)
    node = np.full(dem.shape, NO_NODE, dtype=np.int64)
    node[land] = np.arange(land.sum())
    elevation = dem.astype(np.float64).filled(np.nan)
    logger.info(
        "Terrain: %d x %d cells of %g m, %d land cells",
        dem.shape[0],
        dem.shape[1],
        abs(transform.a),
        land.sum(),
    )
    return Terrain(elevation, slope, node, transform, crs)


def _shifted_pair(
    grid: np.ndarray, row_offset: int, column_offset: int
) -> tuple[np.ndarray, np.ndarray]:
    """Two views of a grid such that cell [i, j] of the second is the
    neighbour at (row_offset, column_offset) of cell [i, j] of the first.

    Shifting the whole grid lets numpy handle every pair of neighbours in
    one direction at once, instead of a Python loop over millions of cells.
    ``row_offset`` is >= 0 (see ``HALF_NEIGHBOURHOODS``).
    """
    rows, columns = grid.shape
    left, right = max(0, -column_offset), max(0, column_offset)
    first = grid[: rows - row_offset, left : columns - right]
    second = grid[row_offset:, right : columns - left]
    return first, second


def build_graph(terrain: Terrain, neighbours: int) -> csr_matrix:
    """The weighted graph of the land cells.

    Parameters
    ----------
    terrain : Terrain
        From ``load_terrain``.
    neighbours : int
        8 or 16 (``HALF_NEIGHBOURHOODS``).

    Returns
    -------
    scipy.sparse.csr_matrix
        N x N, N = land cells. Entry [a, b] is the cost of a step between
        neighbours a and b: the surface distance (horizontal and vertical)
        times the mean slope of the two cells. Each pair is stored once
        (a step costs the same both ways); use it as undirected.
    """
    cell_height = abs(terrain.transform.e)
    cell_width = abs(terrain.transform.a)
    starts, ends, costs = [], [], []
    for row_offset, column_offset in HALF_NEIGHBOURHOODS[neighbours]:
        first, second = _shifted_pair(terrain.node, row_offset, column_offset)
        both_land = (first != NO_NODE) & (second != NO_NODE)
        first_elevation, second_elevation = _shifted_pair(
            terrain.elevation, row_offset, column_offset
        )
        first_slope, second_slope = _shifted_pair(
            terrain.slope, row_offset, column_offset
        )
        horizontal = np.hypot(
            row_offset * cell_height, column_offset * cell_width
        )
        rise = first_elevation[both_land] - second_elevation[both_land]
        surface_length = np.hypot(horizontal, rise)
        mean_slope = (first_slope[both_land] + second_slope[both_land]) / 2
        starts.append(first[both_land])
        ends.append(second[both_land])
        costs.append(np.maximum(surface_length * mean_slope, MIN_EDGE_COST))

    size = terrain.node_count
    graph = coo_matrix(
        (
            np.concatenate(costs),
            (np.concatenate(starts), np.concatenate(ends)),
        ),
        shape=(size, size),
    ).tocsr()
    logger.info(
        "Graph: %d nodes, %d edges (%d neighbours)",
        size,
        graph.nnz,
        neighbours,
    )
    return graph


def cells_of_points(
    terrain: Terrain, x: np.ndarray, y: np.ndarray, crs: str
) -> np.ndarray:
    """The node of the cell each point falls in.

    Parameters
    ----------
    terrain : Terrain
        From ``load_terrain``.
    x, y : numpy.ndarray
        Point coordinates in ``crs``.
    crs : str
        The points' CRS, e.g. ``EPSG:32631``.

    Returns
    -------
    numpy.ndarray
        One node number per point; ``NO_NODE`` if the point falls on the
        sea or outside the grid.
    """
    to_grid = Transformer.from_crs(crs, terrain.crs, always_xy=True)
    grid_x, grid_y = to_grid.transform(x, y)
    # ~transform maps coordinates to fractional (column, row).
    columns, rows = ~terrain.transform * (
        np.asarray(grid_x),
        np.asarray(grid_y),
    )
    rows, columns = np.floor(rows).astype(int), np.floor(columns).astype(int)
    height, width = terrain.node.shape
    inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    node = np.full(len(rows), NO_NODE, dtype=np.int64)
    node[inside] = terrain.node[rows[inside], columns[inside]]
    return node


def to_grid(terrain: Terrain, values: np.ndarray, fill: float) -> np.ndarray:
    """Put one value per node back on the grid.

    Parameters
    ----------
    terrain : Terrain
        From ``load_terrain``.
    values : numpy.ndarray
        One value per node (length N).
    fill : float
        Value of the sea cells.

    Returns
    -------
    numpy.ndarray
        Grid of the terrain's shape.
    """
    grid = np.full(terrain.node.shape, fill, dtype=values.dtype)
    grid[terrain.land] = values
    return grid


def allocate(
    graph: csr_matrix, gauge_nodes: np.ndarray, gauge_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The cheapest gauge of every land cell, and its accumulated cost.

    One multi-source Dijkstra: costs spread from all the gauges at once,
    and each cell keeps the gauge that reaches it most cheaply.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        From ``build_graph``.
    gauge_nodes : numpy.ndarray
        Node of each gauge's cell (``cells_of_points``); gauges on
        ``NO_NODE`` are left out.
    gauge_ids : numpy.ndarray
        The gauges' ids (``id`` of stations.csv), same order.

    Returns
    -------
    gauge : numpy.ndarray
        Gauge id per node; ``NO_GAUGE`` where no gauge is reached.
    cost : numpy.ndarray
        Accumulated cost to that gauge per node; inf where unreached.
    """
    on_land = gauge_nodes != NO_NODE
    for gauge_id in gauge_ids[~on_land]:
        logger.warning("Gauge %s is not on a land cell: left out", gauge_id)
    # Two gauges on the same cell (B013 / B013X): ArcGIS gives the cell to
    # the lower id, and Dijkstra wouldn't say which one won, so only the
    # lower id of each cell is a source.
    order = np.argsort(gauge_ids[on_land], kind="stable")
    nodes, ids = gauge_nodes[on_land][order], gauge_ids[on_land][order]
    nodes, first = np.unique(nodes, return_index=True)
    ids = ids[first]

    cost, _, source = dijkstra(
        graph,
        directed=False,
        indices=nodes,
        min_only=True,
        return_predecessors=True,
    )
    # `source` is the source *node* of each node; translate it to a gauge
    # id (`nodes` is sorted, from np.unique, so searchsorted finds it).
    gauge = np.full(len(source), NO_GAUGE, dtype=np.int64)
    reached = source != UNREACHED_SOURCE
    gauge[reached] = ids[np.searchsorted(nodes, source[reached])]
    logger.info(
        "Allocation: %d gauges, %d land cells reached by none",
        len(nodes),
        int((gauge == NO_GAUGE).sum()),
    )
    return gauge, cost


def costs_between(
    graph: csr_matrix, sources: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Accumulated cost from each source node to each target node.

    One Dijkstra per source, keeping only the targets' values: all the
    sources at once would hold a full row of N costs for each of them.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        From ``build_graph``.
    sources, targets : numpy.ndarray
        Node numbers (none ``NO_NODE``).

    Returns
    -------
    numpy.ndarray
        Shape (sources, targets); inf where a target can't be reached.
    """
    table = np.empty((len(sources), len(targets)))
    for row, source in enumerate(sources):
        table[row] = dijkstra(graph, directed=False, indices=source)[targets]
    return table
