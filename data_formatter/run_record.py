"""History of formatter runs: what data was produced, from what, and how.

The formatter's counterpart of the modelling repo's experiments/. Every run of
`python -m data_formatter.main` gets its own folder:

    runs/<run_id>/                 run_id = formatter_<UTC timestamp>
        run_config.json            settings, inputs, outputs, code and environment
        run.log                    every log message of the run, with time and level
        outputs/                   everything the run writes: final_gdf.csv, the
                                   intermediate CSVs, the EDA plots,
                                   formatter_config.json and the gzipped AEMET table
        config.yaml                the settings file the run used
        analyses/                  only if a script in scripts/ analysed this run
        inputs/                    copies of the small input files (< 5 MB)
        code/                      the data_formatter/*.py that ran
    runs/index.csv                 one row per finished run

Why each part:
  - run_config.json: the settings (negative shift, temporal controls, study
    period, seeds...) and the SHA-256 of every input and output. Same hash =
    same file. `outputs.final_gdf.csv.sha256` equals `data_sha256` in the
    modelling repo's run_config.json, so an experiment leads back here.
  - inputs/: the small inputs change by hand (negatives.csv from a workmate,
    the inventory), so a copy shows exactly what was used. Big inputs
    (AEMET history, susceptibility shapefile) are only hashed.
  - The RiscBal database reads are not kept (your choice, 2026-10-17): if
    the database changes, a rerun may differ for the 3 Esporles events and
    their controls, and the hashes will show it.
  - code/: the code is often not committed when it runs (git "dirty"). The
    commit alone wouldn't say which code made the data; a copy does.
  - outputs/: the run writes its outputs straight here and nowhere else, so
    one run can never overwrite another's files (an offline test run
    included). The modelling repo copies outputs/final_gdf.csv.
  - run.log: counts and warnings (e.g. rockfalls with fewer controls) that
    aren't stored anywhere else. The modules log through `logging` (logger
    per module, under "data_formatter"); FormatterRun sends those messages
    both to the terminal and to run.log, and also logs a crash's traceback.

A run that crashes keeps its folder with run.log, but gets no run_config.json
and no row in index.csv, so only finished runs count as history.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import TracebackType

import pandas as pd

from data_formatter import inventory
from data_formatter.config import Config
from data_formatter.gauge_fallback import REASSIGNED

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
CODE_DIR = Path(__file__).resolve().parent

TRACKED_PACKAGES = [
    "pandas",
    "numpy",
    "geopandas",
    "shapely",
    "pyogrio",
    "psycopg",
    "seaborn",
    "pyyaml",
]
logger = logging.getLogger(__name__)

# Every module of the package logs under this name (logging.getLogger(__name__)),
# so configuring it captures them all, but not third-party libraries.
PACKAGE_LOGGER = "data_formatter"

SECONDS_PER_HOUR = 3600
MAX_INPUT_COPY_BYTES = 5 * 1024 * 1024


def _sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MB chunks (the AEMET table is ~200 MB).

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    str
        Hex digest. Same hash = same bytes.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_info(path: Path) -> dict[str, object]:
    """Hash and size of a file, as recorded in ``run_config.json``.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    dict
        ``sha256`` and ``bytes``.
    """
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


def _git_state() -> dict[str, object]:
    """Current commit and whether there are uncommitted changes.

    Only read-only git calls (``rev-parse``, ``status``).

    Returns
    -------
    dict
        ``commit`` (hash) and ``dirty`` (bool); both None if git or the
        repository isn't available.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def _package_versions() -> dict[str, str | None]:
    """Installed versions of the packages that shape the output.

    Returns
    -------
    dict of str to str or None
        Version per name in ``TRACKED_PACKAGES``; None if not installed.
    """
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


class _ConsoleFormatter(logging.Formatter):
    """Plain messages; warnings and errors prefixed with their level."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a record for the terminal.

        Parameters
        ----------
        record : logging.LogRecord
            The record to print.

        Returns
        -------
        str
            The message, prefixed with the level for warnings and errors.
        """
        message = super().format(record)
        return (
            f"{record.levelname}: {message}"
            if record.levelno >= logging.WARNING
            else message
        )


class FormatterRun:
    """One formatter run: create at the start of main.py, ``finish()`` at the end.

    Creating it starts the run's history folder and its logging: every
    message of the ``data_formatter`` package goes to the terminal and to
    ``runs/<run_id>/run.log``, and an uncaught exception is logged too.

    Parameters
    ----------
    runs_dir : pathlib.Path, optional
        Folder of the run histories (default: ``runs/`` at the repo root;
        ``paths.runs`` in config.yaml).

    Attributes
    ----------
    started : datetime.datetime
        Start time (UTC).
    run_id : str
        ``formatter_<UTC timestamp>``.
    dir : pathlib.Path
        This run's folder, ``runs_dir / run_id``.
    outputs : pathlib.Path
        Where the run writes all its outputs, ``dir / "outputs"``.
    """

    def __init__(self, runs_dir: Path = RUNS_DIR) -> None:
        """Create the run folder and start logging (see the class docstring)."""
        self.started = datetime.now(timezone.utc)
        self.run_id = f"formatter_{self.started.strftime('%Y%m%dT%H%M%SZ')}"
        self.dir = runs_dir / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.outputs = self.dir / "outputs"
        self.outputs.mkdir()
        self._configure_logging()
        logger.info("Formatter run: %s (history in %s)", self.run_id, self.dir)

    def _configure_logging(self) -> None:
        """Send the package's log to the terminal and to ``run.log``.

        Terminal: the message (level only for warnings). ``run.log``: time,
        level, module. Also installs an exception hook that logs a crash to
        ``run.log`` before Python prints it.
        """
        package = logging.getLogger(PACKAGE_LOGGER)
        package.setLevel(logging.INFO)
        self._previous_propagate = package.propagate
        package.propagate = False  # don't also print through the root logger

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(_ConsoleFormatter("%(message)s"))
        # A crash's traceback is printed by Python itself; only the file gets it here.
        console.addFilter(lambda record: not getattr(record, "crash", False))
        file = logging.FileHandler(self.dir / "run.log", encoding="utf-8")
        file.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
            )
        )
        self._handlers = [console, file]
        for handler in self._handlers:
            package.addHandler(handler)

        # An uncaught exception is written to run.log as well, so a failed
        # run's folder says why it failed.
        previous_hook = sys.excepthook

        def log_crash(
            exc_type: type[BaseException],
            exc: BaseException,
            traceback: TracebackType | None,
        ) -> None:
            package.critical(
                "Run failed",
                exc_info=(exc_type, exc, traceback),
                extra={"crash": True},
            )
            previous_hook(exc_type, exc, traceback)

        sys.excepthook = log_crash
        self._previous_excepthook = previous_hook

    def finish(
        self,
        settings: dict[str, object],
        config_file: Path,
        data_path: Path,
        summary: dict[str, object],
        index_fields: dict[str, object],
    ) -> dict[str, object]:
        """Record the run and stop its logging.

        Hashes the outputs (already in ``outputs/``), copies the inputs
        (under 5 MB), the code and the config file into the run folder,
        writes ``run_config.json`` and appends a row to ``runs/index.csv``.

        Parameters
        ----------
        settings : dict
            The run's settings (``settings_record``).
        config_file : pathlib.Path
            The YAML the run was configured with.
        data_path : pathlib.Path
            Input folder (``backup/`` is left out).
        summary : dict
            Counts of the export (``run_summary``).
        index_fields : dict
            Extra columns for the run's row in ``runs/index.csv``.

        Returns
        -------
        dict
            The content of ``run_config.json``.
        """
        inputs = {}
        for path in sorted(
            p
            for p in data_path.rglob("*")
            if p.is_file() and "backup" not in p.parts
        ):
            relative = path.relative_to(data_path)
            info = _file_info(path)
            info["copied"] = info["bytes"] <= MAX_INPUT_COPY_BYTES
            if info["copied"]:
                target = self.dir / "inputs" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            inputs[str(relative)] = info

        # Every output is in the run's own folder: hashed, nothing copied.
        outputs = {
            path.name: _file_info(path)
            for path in sorted(self.outputs.iterdir())
            if path.is_file()
        }

        code_dir = self.dir / "code"
        code_dir.mkdir()
        code = {}
        for path in sorted(CODE_DIR.glob("*.py")):
            shutil.copy2(path, code_dir / path.name)
            code[path.name] = _sha256(path)

        # The YAML the run was configured with, as it was (settings has the
        # parsed values; the file also keeps its comments).
        shutil.copy2(config_file, self.dir / config_file.name)

        finished = datetime.now(timezone.utc)
        config = {
            "run_id": self.run_id,
            "started_utc": self.started.isoformat(timespec="seconds"),
            "finished_utc": finished.isoformat(timespec="seconds"),
            "duration_s": round((finished - self.started).total_seconds()),
            # dirty = uncommitted changes: then code/ (not the commit) is the
            # exact code of this run.
            "git": _git_state(),
            "python": platform.python_version(),
            "packages": _package_versions(),
            "settings": settings,
            "config_file": {
                "path": str(config_file),
                "sha256": _sha256(config_file),
            },
            "summary": summary,
            "inputs": inputs,
            "outputs": outputs,
            "code_sha256": code,
        }
        (self.dir / "run_config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n"
        )

        row = {
            "run_id": self.run_id,
            "finished_utc": config["finished_utc"],
            "final_gdf_sha256": outputs.get("final_gdf.csv", {}).get("sha256"),
            "git_commit": config["git"]["commit"],
            "git_dirty": config["git"]["dirty"],
            **index_fields,
        }
        index = self.dir.parent / "index.csv"
        pd.DataFrame([row]).to_csv(
            index, mode="a", header=not index.exists(), index=False
        )

        logger.info("Run recorded: %s", self.dir)
        self._close_logging()
        return config

    def _close_logging(self) -> None:
        """Remove this run's log handlers and restore the previous state."""
        package = logging.getLogger(PACKAGE_LOGGER)
        for handler in self._handlers:
            package.removeHandler(handler)
            handler.close()
        package.propagate = self._previous_propagate
        sys.excepthook = self._previous_excepthook


# --------------------------------------------------------------------------
# What this run did: settings and counts, for run_config.json and
# results/formatter_config.json
# --------------------------------------------------------------------------
def settings_record(config: Config) -> dict[str, object]:
    """The run's settings, as stored in ``runs/<run_id>/run_config.json``.

    Parameters
    ----------
    config : Config
        Run settings.

    Returns
    -------
    dict
        JSON-ready settings, grouped by topic.
    """
    return {
        "samples": {
            "inventory_event_type": inventory.EVENT_TYPE,
            "inventory_date_format": inventory.INVENTORY_DATE_FORMAT,
            "period": [
                str(inventory.PERIOD_START.date()),
                str(inventory.PERIOD_END.date()),
            ],
            "coordinate_match_decimals": inventory.COORDINATE_DECIMALS,
            "required_columns": inventory.REQUIRED_COLUMNS,
        },
        "gauges": {
            "assignment": "slope-weighted Thiessen polygons (ArcGIS, workmate); event_stations.csv and negatives.csv",
            "reassignments": [
                asdict(reassignment)
                for reassignment in config.gauge_reassignments
            ],
            "riscbal_gauges": list(config.riscbal_stations),
            "temporal_controls_at_riscbal_gauges": False,
            "fallback": {
                "rule": "own gauge if it covers the window, else nearest stations.csv gauge that does",
                "distance": "straight-line (stand-in for slope-weighted cost distance)",
                **vars(config.fallback),
            },
        },
        "rain": {
            "windows": "event day; 7, 30 and 90 days before it (excluding it)",
            "pluviometric_day_offset_h": (
                config.pluviometric_day_offset.total_seconds()
                / SECONDS_PER_HOUR
            ),
            "missing_days": "counted as 0 mm (see rain_days_90d); under-measured samples use a fallback gauge (gauges.fallback)",
        },
        "negative_shift": {
            **asdict(config.negative_shift),
            "period": list(config.negative_shift.period),
        },
        "temporal_controls": {
            **{
                key: list(value) if isinstance(value, tuple) else value
                for key, value in vars(config.controls).items()
            },
        },
    }


def run_summary(
    final: pd.DataFrame,
    shifted: pd.DataFrame,
    control_summary: pd.DataFrame,
    fallback_report: pd.DataFrame,
    config: Config,
) -> dict[str, object]:
    """Counts of the export, as stored in ``run_config.json``.

    Parameters
    ----------
    final : pandas.DataFrame
        The exported dataset.
    shifted : pandas.DataFrame
        The negatives moved to high-rainfall days.
    control_summary : pandas.DataFrame
        Candidates and controls per rockfall.
    fallback_report : pandas.DataFrame
        Samples not covered by their own gauge.
    config : Config
        Run settings (controls per rockfall).

    Returns
    -------
    dict
        Sample counts, shifted negatives, rockfalls with fewer controls and
        the fallback outcomes.
    """
    rockfalls_with_fewer_controls = (
        control_summary.loc[
            control_summary["n_controls"] < config.controls.per_rockfall,
            "sample_id",
        ].tolist()
        if not control_summary.empty
        else []
    )
    # Samples not covered by their own gauge, by sample type and outcome.
    gauge_fallback = (
        fallback_report.groupby(["sample_type", "status"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
        if not fallback_report.empty
        else {}
    )
    n_reassigned = (
        int((fallback_report["status"] == REASSIGNED).sum())
        if not fallback_report.empty
        else 0
    )
    return {
        "n_samples": len(final),
        "n_positives": int(final["is_rockfall"].sum()),
        "n_by_sample_type": final["sample_type"].value_counts().to_dict(),
        "n_shifted": len(shifted),
        "rockfalls_with_fewer_controls": rockfalls_with_fewer_controls,
        "gauge_fallback": gauge_fallback,
        "n_fallback_reassigned": n_reassigned,
    }


def write_formatter_config(
    run_id: str,
    settings: dict[str, object],
    summary: dict[str, object],
    config: Config,
    output_path: Path,
) -> None:
    """Write ``formatter_config.json`` to the run's outputs folder.

    Kept for the existing docs and checks: the same settings and counts as
    before, plus the run id. The full record is
    ``runs/<run_id>/run_config.json``. ``final_gdf_sha256`` is the hash the
    modelling repo stores as ``data_sha256`` in each experiment, so an
    experiment can be traced back to its formatter run.

    Parameters
    ----------
    run_id : str
        The formatter run.
    settings : dict
        From ``settings_record``.
    summary : dict
        From ``run_summary``.
    config : Config
        Run settings (input folder).
    output_path : pathlib.Path
        The run's outputs folder (``runs/<run_id>/outputs/``).
    """
    n_by_sample_type = summary["n_by_sample_type"]
    formatter_config = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "negative_shift": {
            **settings["negative_shift"],
            "n_shifted": summary["n_shifted"],
        },
        "temporal_controls": {
            **settings["temporal_controls"],
            "n_controls": int(n_by_sample_type.get("temporal_control", 0)),
            "rockfalls_with_fewer_controls": summary[
                "rockfalls_with_fewer_controls"
            ],
        },
        "n_by_sample_type": n_by_sample_type,
        "n_samples": summary["n_samples"],
        "n_positives": summary["n_positives"],
        "inputs_sha256": {
            path.name: _sha256(path)
            for path in sorted(config.data_path.glob("*.csv"))
        },
        "final_gdf_sha256": _sha256(output_path / "final_gdf.csv"),
    }
    (output_path / "formatter_config.json").write_text(
        json.dumps(formatter_config, indent=2) + "\n"
    )
