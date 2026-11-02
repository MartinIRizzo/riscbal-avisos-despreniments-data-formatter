"""Which formatter run a script reads, and where it writes its analysis.

Every formatter run keeps all its outputs in ``runs/<run_id>/outputs/``. A
script reads the outputs of one run (by default the latest finished one,
the last row of ``runs/index.csv``; another with ``--run <run_id>``) and
writes its own results to ``runs/<run_id>/analyses/<script>/``, next to the
data it analysed. Downloads that don't depend on a run (the AEMET API
responses) go to ``cache/``.

Standard library only: ``build_qgis_project.py`` runs with QGIS's own
Python, which may not have pandas.
"""

import argparse
import csv
from pathlib import Path

RUNS_PATH = Path("runs")
INDEX_FILE = RUNS_PATH / "index.csv"
CACHE_PATH = Path("cache")


def add_run_argument(parser: argparse.ArgumentParser) -> None:
    """Add the ``--run`` option to a script's argument parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The script's parser.
    """
    parser.add_argument(
        "--run",
        help="formatter run id to read (default: the latest in runs/index.csv)",
    )


def run_folder(run_id: str | None) -> Path:
    """The folder of a formatter run.

    Parameters
    ----------
    run_id : str or None
        A run id (``formatter_<UTC timestamp>``), or None for the latest
        finished run (the last row of ``runs/index.csv``).

    Returns
    -------
    pathlib.Path
        ``runs/<run_id>/``.

    Raises
    ------
    FileNotFoundError
        If the run has no ``outputs/`` folder. Runs from before
        2026-11-02 only kept copies of their small outputs there (no AEMET
        table), so scripts that need it can't read those runs.
    """
    if run_id is None:
        with INDEX_FILE.open(newline="", encoding="utf-8") as index:
            run_id = list(csv.DictReader(index))[-1]["run_id"]
    folder = RUNS_PATH / run_id
    if not (folder / "outputs").is_dir():
        message = f"{folder / 'outputs'} not found"
        raise FileNotFoundError(message)
    return folder


def analysis_folder(run: Path, name: str) -> Path:
    """Where a script writes its results for one run (created if needed).

    Parameters
    ----------
    run : pathlib.Path
        From ``run_folder``.
    name : str
        The analysis, e.g. ``gauge_study``.

    Returns
    -------
    pathlib.Path
        ``runs/<run_id>/analyses/<name>/``.
    """
    folder = run / "analyses" / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder
