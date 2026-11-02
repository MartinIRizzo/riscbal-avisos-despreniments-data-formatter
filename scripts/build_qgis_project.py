"""Build a styled QGIS project from a run's riscbal_gauges.gpkg.

Run with QGIS's own Python (not the uv environment), from the repo root, after
`uv run python -m scripts.export_gis` for the same run:

    /usr/bin/python3.14 -m scripts.build_qgis_project [--run <run_id>]

Reads and writes runs/<run_id>/analyses/gis/ (default: the latest run; see
run_selection.py). Output: riscbal_gauges.qgz there. Open it in QGIS; the year is picked
with the Temporal Controller (View → Panels → Temporal Controller), which steps
one year at a time between 1990 and 2022. Styling follows the web map:
    - gauge colour: blue = automatic (served by the AEMET API), green = manual
      (only in the file AEMET delivered); filled = data on ≥ 90% of the days of
      the selected year, pale = some days, hollow grey = no data that year;
    - black ring: the gauge is in stations.csv (used to build the polygons);
    - dashed ring: part of that year is accumulated (-4) readings;
    - orange triangles: rockfalls with fewer than 81 of their 90 antecedent days
      measured; lines go to the assigned gauge (solid) and to the nearest gauge
      with data (dashed).
This script only uses the qgis package and the standard library (through
run_selection.py), so it doesn't import the formatter.
"""

import argparse
import sys
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsDateTimeRange,
    QgsLineSymbol,
    QgsMarkerSymbol,
    QgsPalLayerSettings,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsReferencedRectangle,
    QgsRuleBasedRenderer,
    QgsTextBufferSettings,
    QgsTextFormat,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QDate, QDateTime, QTime
from qgis.PyQt.QtGui import QColor, QFont

from scripts.run_selection import add_run_argument, run_folder

GPKG_FILE = "riscbal_gauges.gpkg"
PROJECT_FILE = "riscbal_gauges.qgz"
CRS = QgsCoordinateReferenceSystem("EPSG:32631")

# Same colours as the web map (validated for colour-blind separation).
AUTOMATIC = "#2a78d6"
MANUAL = "#1baf7a"
PROBLEM = "#eb6834"
INK = "#1d2422"
OFF = "#9aa5a2"

# Mallorca, in EPSG:32631 metres.
MALLORCA = QgsRectangle(444000, 4346000, 536000, 4425000)
YEARS = (1990, 2022)


def layer(gpkg: Path, name: str, title: str) -> QgsVectorLayer:
    """Open one layer of the GeoPackage, or exit if it can't be read.

    Parameters
    ----------
    gpkg : pathlib.Path
        The run's ``riscbal_gauges.gpkg`` (absolute path).
    name : str
        Layer name in ``riscbal_gauges.gpkg``.
    title : str
        Name shown in the QGIS layer panel.

    Returns
    -------
    qgis.core.QgsVectorLayer
        The layer.
    """
    vector = QgsVectorLayer(f"{gpkg}|layername={name}", title, "ogr")
    if not vector.isValid():
        sys.exit(f"Could not open layer {name} in {gpkg}")
    return vector


def marker(
    shape: str,
    size: float,
    fill: str,
    stroke: str,
    stroke_width: float = 0.3,
    dashed: bool = False,
    fill_alpha: int = 255,
) -> QgsMarkerSymbol:
    """A simple marker symbol.

    Parameters
    ----------
    shape : str
        QGIS marker shape (e.g. ``circle``, ``triangle``).
    size : float
        Size in mm.
    fill : str
        Fill colour, or ``none`` for a transparent fill.
    stroke : str
        Outline colour.
    stroke_width : float, optional
        Outline width in mm.
    dashed : bool, optional
        Dashed outline.
    fill_alpha : int, optional
        Fill opacity, 0..255.

    Returns
    -------
    qgis.core.QgsMarkerSymbol
        The symbol.
    """
    color = QColor(fill) if fill != "none" else QColor(0, 0, 0, 0)
    if fill != "none":
        color.setAlpha(fill_alpha)
    symbol = QgsMarkerSymbol.createSimple(
        {
            "name": shape,
            "size": str(size),
            "outline_color": stroke,
            "outline_width": str(stroke_width),
            "outline_style": "dash" if dashed else "solid",
        }
    )
    symbol.setColor(color)
    symbol.symbolLayer(0).setStrokeColor(QColor(stroke))
    return symbol


def rule(
    symbol: QgsMarkerSymbol | QgsLineSymbol,
    expression: str | None,
    label: str,
    active: bool = True,
) -> QgsRuleBasedRenderer.Rule:
    """One rule of a rule-based renderer.

    Parameters
    ----------
    symbol : qgis.core.QgsMarkerSymbol or qgis.core.QgsLineSymbol
        How the matching features are drawn.
    expression : str or None
        QGIS filter expression; None matches every feature.
    label : str
        Legend text.
    active : bool, optional
        Whether the rule is switched on when the project opens.

    Returns
    -------
    qgis.core.QgsRuleBasedRenderer.Rule
        The rule.
    """
    item = QgsRuleBasedRenderer.Rule(
        symbol, filterExp=expression or "", label=label
    )
    item.setActive(active)
    return item


def rule_renderer(
    rules: list[QgsRuleBasedRenderer.Rule],
) -> QgsRuleBasedRenderer:
    """A rule-based renderer with these rules, in order.

    Parameters
    ----------
    rules : list of qgis.core.QgsRuleBasedRenderer.Rule
        From ``rule``.

    Returns
    -------
    qgis.core.QgsRuleBasedRenderer
        The renderer.
    """
    root = QgsRuleBasedRenderer.Rule(None)
    for item in rules:
        root.appendChild(item)
    return QgsRuleBasedRenderer(root)


def labels(vector: QgsVectorLayer, expression: str, size: float = 8) -> None:
    """Label a layer's features with an expression (white halo).

    Parameters
    ----------
    vector : qgis.core.QgsVectorLayer
        The layer to label.
    expression : str
        QGIS expression giving the label text.
    size : float, optional
        Font size in points.
    """
    settings = QgsPalLayerSettings()
    settings.fieldName = expression
    settings.isExpression = True
    text = QgsTextFormat()
    text.setFont(QFont("Sans Serif"))
    text.setSize(size)
    text.setColor(QColor(INK))
    buffer = QgsTextBufferSettings()
    buffer.setEnabled(True)
    buffer.setSize(0.8)
    buffer.setColor(QColor("white"))
    text.setBuffer(buffer)
    settings.setFormat(text)
    vector.setLabeling(QgsVectorLayerSimpleLabeling(settings))
    vector.setLabelsEnabled(True)


def main() -> None:
    """Build ``riscbal_gauges.qgz`` in the run's ``analyses/gis/``."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_argument(parser)
    gis_path = (
        run_folder(parser.parse_args().run) / "analyses" / "gis"
    ).resolve()
    gpkg = gis_path / GPKG_FILE
    project_file = gis_path / PROJECT_FILE
    if not gpkg.exists():
        sys.exit(
            f"{gpkg} not found: run `uv run python -m scripts.export_gis` first"
        )

    QgsApplication.setPrefixPath("/usr", True)
    app = QgsApplication([], False)
    app.initQgis()

    project = QgsProject.instance()
    project.setCrs(CRS)
    project.setTitle("RiscBal: rain gauges and rockfalls")

    # Basemap
    osm = QgsRasterLayer(
        "type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png&zmax=19&zmin=0",
        "OpenStreetMap",
        "wms",
    )

    # All gauges: hollow grey (= "no data in the selected year" once the coverage layer is drawn on top),
    # with a black ring for the stations.csv ones.
    gauges = layer(
        gpkg, "gauges", "Gauges (hollow = no data in the selected year)"
    )
    in_csv = marker("circle", 3.2, "none", OFF, 0.4)
    # Keep the ring's symbol alive while cloning its layer: a temporary would be
    # freed by Python first, and QGIS would read freed memory (segfault).
    ring_symbol = marker("circle", 4.6, "none", INK, 0.3)
    in_csv.appendSymbolLayer(ring_symbol.symbolLayer(0).clone())
    gauges.setRenderer(
        rule_renderer(
            [
                rule(
                    in_csv, '"in_stations_csv"', "In stations.csv (black ring)"
                ),
                rule(
                    marker("circle", 3.2, "none", OFF, 0.4),
                    'NOT "in_stations_csv"',
                    "Other gauges",
                ),
            ]
        )
    )
    labels(gauges, 'CASE WHEN "in_stations_csv" THEN "code" END', 7)
    gauges.setLabelsEnabled(False)

    # Coverage per year (temporal): fill by type and coverage, dashed ring for accumulated (-4) days.
    years = layer(gpkg, "gauge_years", "Gauge coverage in the selected year")
    years.setRenderer(
        rule_renderer(
            [
                rule(
                    marker("circle", 3.2, AUTOMATIC, "white", 0.2),
                    '"gauge_type" = \'automatic\' AND "coverage" >= 0.9',
                    "Automatic (in API), ≥ 90% of days",
                ),
                rule(
                    marker(
                        "circle", 3.2, AUTOMATIC, "white", 0.2, fill_alpha=110
                    ),
                    '"gauge_type" = \'automatic\' AND "coverage" < 0.9',
                    "Automatic (in API), some days",
                ),
                rule(
                    marker("circle", 3.2, MANUAL, "white", 0.2),
                    '"gauge_type" = \'manual\' AND "coverage" >= 0.9',
                    "Manual (file only), ≥ 90% of days",
                ),
                rule(
                    marker(
                        "circle", 3.2, MANUAL, "white", 0.2, fill_alpha=110
                    ),
                    '"gauge_type" = \'manual\' AND "coverage" < 0.9',
                    "Manual (file only), some days",
                ),
                rule(
                    marker("circle", 6.0, "none", INK, 0.35, dashed=True),
                    '"accumulated_share" > 0',
                    "Part of the year accumulated (−4)",
                ),
            ]
        )
    )
    temporal = years.temporalProperties()
    temporal.setIsActive(True)
    temporal.setMode(
        Qgis.VectorTemporalMode.FeatureDateTimeStartAndEndFromFields
    )
    temporal.setStartField("year_start")
    temporal.setEndField("year_end")
    temporal.setLimitMode(Qgis.VectorTemporalLimitMode.IncludeBeginIncludeEnd)

    api_only = layer(gpkg, "api_only_gauges", "In the API but not in our file")
    api_only.setRenderer(
        rule_renderer(
            [
                rule(
                    marker("circle", 3.2, "white", AUTOMATIC, 0.5),
                    None,
                    "API-only station",
                )
            ]
        )
    )

    # Links rockfall → gauge
    links = layer(gpkg, "rockfall_links", "Rockfall → gauge")
    solid = QgsLineSymbol.createSimple(
        {"line_color": INK, "line_width": "0.45"}
    )
    dashed = QgsLineSymbol.createSimple(
        {"line_color": PROBLEM, "line_width": "0.55", "line_style": "dash"}
    )
    faint = QgsLineSymbol.createSimple(
        {"line_color": OFF, "line_width": "0.2"}
    )
    links.setRenderer(
        rule_renderer(
            [
                rule(
                    solid,
                    '"link" = \'assigned\' AND "under_measured"',
                    "Under-measured rockfall → assigned gauge",
                ),
                rule(
                    dashed,
                    "\"link\" = 'nearest with data'",
                    "Under-measured rockfall → nearest gauge with data",
                ),
                rule(
                    faint,
                    '"link" = \'assigned\' AND NOT "under_measured"',
                    "Other rockfalls → assigned gauge",
                    active=False,
                ),
            ]
        )
    )

    # Rockfalls (always shown) and a temporal highlight of the selected year's rockfalls.
    rockfalls = layer(gpkg, "rockfalls", "Rockfalls")
    rockfalls.setRenderer(
        rule_renderer(
            [
                rule(
                    marker("triangle", 4.2, PROBLEM, "white", 0.3),
                    '"under_measured"',
                    "Fewer than 81 of 90 days measured",
                ),
                rule(
                    marker("triangle", 3.0, INK, "white", 0.25),
                    'NOT "under_measured" OR "under_measured" IS NULL',
                    "Rockfall",
                ),
            ]
        )
    )
    labels(rockfalls, 'CASE WHEN "under_measured" THEN "sample_id" END', 7.5)

    this_year = layer(gpkg, "rockfalls", "Rockfalls in the selected year")
    this_year.setRenderer(
        rule_renderer(
            [
                rule(
                    marker("triangle", 6.5, "none", INK, 0.5),
                    None,
                    "Rockfall in the selected year",
                )
            ]
        )
    )
    temporal = this_year.temporalProperties()
    temporal.setIsActive(True)
    temporal.setMode(
        Qgis.VectorTemporalMode.FeatureDateTimeStartAndEndFromExpressions
    )
    temporal.setStartExpression('to_datetime("date")')
    temporal.setEndExpression("to_datetime(\"date\") + to_interval('1 day')")

    # Layer order, top first.
    tree = project.layerTreeRoot()
    for item in (rockfalls, this_year, links, years, api_only, gauges, osm):
        project.addMapLayer(item, False)
        tree.addLayer(item)

    # Time: one-year steps over 1990–2022.
    time_settings = project.timeSettings()
    time_settings.setTemporalRange(
        QgsDateTimeRange(
            QDateTime(QDate(YEARS[0], 1, 1), QTime(0, 0)),
            QDateTime(QDate(YEARS[1], 12, 31), QTime(23, 59, 59)),
        )
    )
    time_settings.setTimeStep(1)
    time_settings.setTimeStepUnit(Qgis.TemporalUnit.Years)

    project.viewSettings().setDefaultViewExtent(
        QgsReferencedRectangle(MALLORCA, CRS)
    )
    if not project.write(str(project_file)):
        sys.exit(f"Could not write {project_file}")
    print(f"Written to {project_file}")
    app.exitQgis()


if __name__ == "__main__":
    main()
