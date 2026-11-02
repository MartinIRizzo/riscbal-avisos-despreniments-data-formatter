"""Build the modelling dataset (runs/<run_id>/outputs/final_gdf.csv).

Run from the repo root::

    python -m data_formatter.main                      # config.yaml
    python -m data_formatter.main --config other.yaml

Needs the database credentials in ``.env`` for the RiscBal gauges. The
settings, including the feature flags, are in ``config.yaml``.

This module only orchestrates: each stage lives in the module of its topic,
and the feature flags (the ``enabled`` field of each optional section of
config.yaml) are checked here, before a stage is called. Stages, in
order (each writes its intermediate file to the run's outputs folder,
``runs/<run_id>/outputs/``; nothing is written anywhere else):

1. samples and their gauges           data_loading.py      samples_with_stations.csv
2. reshaped AEMET daily rain          data_loading.py      formatted_historical_aemet_data.csv
3. negative shift (flag)              negative_shift.py    shifted_negatives.csv
4. temporal controls (flag)           controls.py          temporal_controls_summary.csv
5. fallback gauges (flag)             gauge_fallback.py    gauge_fallback.csv
6. rain windows                       prec_accumulation.py samples_with_accumulations.csv
7. susceptibility                     susceptibility.py    samples_with_susceptibility.csv
8. export                             export.py            final_gdf.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from data_formatter.config import DEFAULT_CONFIG_FILE, Config
from data_formatter.controls import (
    add_temporal_controls,
    measured_days,
    with_own_source_ids,
)
from data_formatter.data_loading import load_aemet, load_samples, load_stations
from data_formatter.export import export_dataset
from data_formatter.gauge_fallback import use_fallback_gauges
from data_formatter.negative_shift import SHIFTED_COLUMNS, shift_negatives
from data_formatter.prec_accumulation import add_rain
from data_formatter.run_record import (
    FormatterRun,
    run_summary,
    settings_record,
    write_formatter_config,
)
from data_formatter.susceptibility import add_susceptibility

logger = logging.getLogger("data_formatter.main")


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns
    -------
    argparse.Namespace
        ``config``: the YAML settings file (default ``config.yaml``).
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help=f"YAML settings file (default: {DEFAULT_CONFIG_FILE})",
    )
    return parser.parse_args()


def _on_off(enabled: bool) -> str:
    """Write a feature flag as text, for the log.

    Parameters
    ----------
    enabled : bool
        The flag.

    Returns
    -------
    str
        ``on`` or ``off``.
    """
    return "on" if enabled else "off"


def main() -> None:
    """Build ``final_gdf.csv`` and record the run."""
    args = parse_args()
    load_dotenv()
    # History of this run (runs/<run_id>/): settings, input/output hashes and
    # copies, code, config file and the full log. See run_record.py. It also
    # sets up logging: every message goes to the terminal and to run.log.
    config = Config.from_yaml(args.config)
    run = FormatterRun(config.runs_path)
    outputs = run.outputs
    logger.info(
        "Settings: %s (negative shift %s, temporal controls %s, "
        "fallback gauges %s)",
        args.config,
        _on_off(config.negative_shift.enabled),
        _on_off(config.controls.enabled),
        _on_off(config.fallback.enabled),
    )

    stations = load_stations(config)
    samples = load_samples(config, stations, outputs)
    aemet = load_aemet(config, outputs)

    shifted = pd.DataFrame(columns=SHIFTED_COLUMNS)
    if config.negative_shift.enabled:
        samples, shifted = shift_negatives(
            samples, aemet, config.negative_shift, outputs
        )
    else:
        logger.info("Negative shift: off")

    # Days each AEMET gauge measured (candidate control days, rain_days_90d).
    days_by_gauge = measured_days(dict(tuple(aemet.groupby("code"))))

    samples = with_own_source_ids(samples)
    control_summary = pd.DataFrame()
    if config.controls.enabled:
        samples, control_summary = add_temporal_controls(
            samples,
            days_by_gauge,
            list(config.riscbal_stations),
            config.controls,
            outputs,
        )
    else:
        logger.info("Temporal controls: off")
    logger.info(
        "Samples by type: %s",
        samples["sample_type"].value_counts().to_dict(),
    )

    fallback_report = pd.DataFrame()
    if config.fallback.enabled:
        samples, fallback_report = use_fallback_gauges(
            samples,
            stations,
            days_by_gauge,
            config.fallback,
            outputs,
        )
    else:
        logger.info("Fallback gauges: off")

    samples = add_rain(
        samples,
        aemet,
        days_by_gauge,
        config.riscbal_stations,
        config.pluviometric_day_offset,
        outputs,
    )
    samples = add_susceptibility(samples, config.data_path, outputs)
    final = export_dataset(samples, outputs)

    settings = settings_record(config)
    summary = run_summary(
        final, shifted, control_summary, fallback_report, config
    )
    write_formatter_config(run.run_id, settings, summary, config, outputs)
    n_by_sample_type = summary["n_by_sample_type"]
    run.finish(
        settings=settings,
        config_file=args.config,
        data_path=config.data_path,
        summary=summary,
        index_fields={
            "n_samples": summary["n_samples"],
            "n_rockfall": int(n_by_sample_type.get("rockfall", 0)),
            "n_temporal_control": int(
                n_by_sample_type.get("temporal_control", 0)
            ),
            "n_random_point": int(n_by_sample_type.get("random_point", 0)),
            "negative_shift": config.negative_shift.enabled,
            "temporal_controls": config.controls.enabled,
            "controls_per_rockfall": config.controls.per_rockfall,
        },
    )
    print(f"Dataset: {outputs / 'final_gdf.csv'}")


if __name__ == "__main__":
    main()
