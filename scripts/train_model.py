"""Train models from an existing processed weekly table."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logger import get_logger, setup_logging


def main() -> None:
    """Parse config paths and execute training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    arguments = parser.parse_args()
    setup_logging()
    from pipelines.training_pipeline import run_training_pipeline

    try:
        run_training_pipeline(arguments.data_config, arguments.model_config)
    except (ValueError, FileNotFoundError) as error:
        get_logger(__name__).error("Training stopped: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()