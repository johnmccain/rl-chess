import argparse
import logging
import pandas as pd

from rl_chess import base_path
from rl_chess.config.config import AppConfig
from rl_chess.evaluation.stockfish_evaluator import StockfishEvaluator
from rl_chess.inference.inference import ChessAgent

app_config = AppConfig()

logging.basicConfig(level=app_config.LOG_LEVEL)

logger = logging.getLogger(__name__)


PERCENTAGE_METRICS = ["BlunderRate", "OptimalityRate"]


def main(args: argparse.Namespace):
    stockfish_evaluator = StockfishEvaluator()

    app_config.APP_MODEL_NAME = args.model_name
    app_config.STOCKFISH_DEPTH = args.depth

    agent = ChessAgent(app_config)
    if args.dataset_path:
        df = pd.read_csv(args.dataset_path)
    else:
        df = pd.read_csv(base_path/"data"/app_config.APP_MOVE_EVAL_DATASET)

    stockfish_evaluator.set_depth(app_config.STOCKFISH_DEPTH)
    average_move_quality, blunder_rate, optimal_rate, type_breakout = stockfish_evaluator.evaluate_move_ratings(agent, df, progress=True)

    logger.info("Completed move evaluation.")
    logger.info(f"Average move quality: {average_move_quality}")
    logger.info(f"Blunder rate: {blunder_rate * 100:.2f}%")
    logger.info(f"Optimal rate: {optimal_rate * 100:.2f}%")
    logger.info("KPI by state type:")
    for state_type, kpi_dict in type_breakout.items():
        for kpi_name, v in kpi_dict.items():
            if kpi_name in PERCENTAGE_METRICS:
                v *= 100
                logger.info(f"\t{state_type} {kpi_name}: {v:.2f}%")
            else:
                logger.info(f"\t{state_type} {kpi_name}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move-quality evaluation script.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the move evaluation dataset. Defaults to value in app config.",
        default=None,
    )
    parser.add_argument(
        "--depth",
        type=int,
        help="Depth of the Stockfish evaluation.",
        default=app_config.STOCKFISH_DEPTH,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Filename of the model to evaluate. Defaults to value in app config.",
        default=app_config.APP_MODEL_NAME,
    )
    args = parser.parse_args()

    main(args)
