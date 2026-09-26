"""Run complete data preparation and model training from configured real CSVs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.data_pipeline import run_data_pipeline
from src.utils.logger import get_logger, setup_logging


def main() -> None:
    """CLI entrypoint; --data-only permits inspection before training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--data-only", action="store_true")
    arguments = parser.parse_args()
    setup_logging()
    try:
        run_data_pipeline(arguments.data_config)
        if not arguments.data_only:
            from pipelines.training_pipeline import run_training_pipeline

            run_training_pipeline(arguments.data_config, arguments.model_config)
    except (ValueError, FileNotFoundError) as error:
        get_logger(__name__).error("Pipeline stopped: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()