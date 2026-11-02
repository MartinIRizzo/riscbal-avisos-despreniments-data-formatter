"""Auxiliary scripts: diagnostics, one-off conversions and analyses.

Not part of the formatter pipeline (data_formatter/), and never imported by it.
Run each from the repo root as a module, e.g.:

    uv run python -m scripts.aemet_api_check
    uv run python -m scripts.export_gis
    uv run python -m scripts.gauge_study

build_qgis_project.py is the exception: it runs with QGIS's own Python.
"""
