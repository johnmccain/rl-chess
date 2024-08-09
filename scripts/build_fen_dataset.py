from typing import TypeVar, Iterable
import argparse
import logging
import random

import chess
import chess.pgn
import pandas as pd
from tqdm import tqdm

from rl_chess import base_path
from rl_chess.config.config import AppConfig
from rl_chess.modeling.utils import (
    calculate_material_score,
)

app_config = AppConfig()


logging.basicConfig(level=app_config.LOG_LEVEL)

logger = logging.getLogger(__name__)

EARLY_GAME_THRESHOLD = 15
ENDGAME_PIECE_COUNT = 12

T = TypeVar("T")

def reservoir_sampling(iterable: Iterable[T], k: int) -> list[T]:
    """
    Reservoir sampling algorithm to sample k items from an iterable.

    :param iterable: The iterable to sample from.
    :param k: The number of items to sample.
    """
    sample = []
    for i, item in enumerate(iterable):
        if i < k:
            sample.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                sample[j] = item
    return sample


def reservoir_sample_update(sample: list[T], item: T, k: int, i: int) -> None:
    """
    Update the reservoir sample with a new item in O(1) time.

    :param sample: The current reservoir sample (list of items).
    :param item: The new item to potentially add to the sample.
    :param k: The size of the reservoir.
    :param i: The index of the current item in the stream.
    """
    if i < k:
        sample.append(item)
    else:
        j = random.randint(0, i)
        if j < k:
            sample[j] = item


def categorize_board_state(board: chess.Board, move_count: int) -> str:
    """
    Categorize the board state into early, midgame, or endgame.
    """
    total_pieces = len(board.piece_map())
    if total_pieces <= ENDGAME_PIECE_COUNT:
        return "endgame"
    elif move_count <= EARLY_GAME_THRESHOLD:
        return "early_game"
    else:
        return "midgame"


def main(args: argparse.Namespace):

    logging.info(f"Reading PGN file from {args.pgn_path}")
    logging.info(f"Sampling {6 * args.sample_size_per_category} states from {args.n_games} games")

    early_game_states: list[str] = []
    midgame_states: list[str] = []
    endgame_states: list[str] = []
    advantageous_material_states: list[str] = []
    disadvantageous_material_states: list[str] = []
    checkmate_states: list[str] = []

    # Tracking observed states for reservoir sampling
    early_game_count = midgame_count = endgame_count = 0
    adv_material_count = disadv_material_count = checkmate_count = 0

    progress = tqdm(total=args.n_games)

    with open(args.pgn_path, "r") as f:
        for _ in range(args.n_games):
            try:
                # Skip games at the beginning of the PGN file (useful if generating a train + test split)
                for _ in range(args.skip_n_games):
                    game = chess.pgn.read_game(f)

                game = chess.pgn.read_game(f)
                if game.errors:
                    # Skip games with errors
                    continue
                board = game.board()
                last_board = None
                move_count = 0

                for move in game.mainline_moves():
                    last_board = board.copy()
                    board.push(move)
                    move_count += 1

                    phase = categorize_board_state(board, move_count)
                    fen = board.fen()

                    if phase == "early_game":
                        reservoir_sample_update(early_game_states, fen, args.sample_size_per_category, early_game_count)
                        early_game_count += 1
                    elif phase == "midgame":
                        reservoir_sample_update(midgame_states, fen, args.sample_size_per_category, midgame_count)
                        midgame_count += 1
                    elif phase == "endgame":
                        reservoir_sample_update(endgame_states, fen, args.sample_size_per_category, endgame_count)
                        endgame_count += 1

                    threshold = args.advantage_threshold
                    material_score = calculate_material_score(board)
                    if material_score > threshold:
                        reservoir_sample_update(advantageous_material_states, fen, args.sample_size_per_category, adv_material_count)
                        adv_material_count += 1
                    elif material_score < -threshold:
                        reservoir_sample_update(disadvantageous_material_states, fen, args.sample_size_per_category, disadv_material_count)
                        disadv_material_count += 1

                if board.is_checkmate() and last_board:
                    checkmate_states = reservoir_sampling(checkmate_states + [last_board.fen()], args.sample_size_per_category)

            except Exception as e:
                logger.warning(f"Error processing game: {e}")
                continue
            progress.update(1)

    df = pd.DataFrame(
        [
            {"fen": fen, "type": "early_game"}
            for fen in early_game_states
        ]
        + [
            {"fen": fen, "type": "midgame"}
            for fen in midgame_states
        ]
        + [
            {"fen": fen, "type": "endgame"}
            for fen in endgame_states
        ]
        + [
            {"fen": fen, "type": "advantageous_material"}
            for fen in advantageous_material_states
        ]
        + [
            {"fen": fen, "type": "disadvantageous_material"}
            for fen in disadvantageous_material_states
        ]
        + [
            {"fen": fen, "type": "checkmate"}
            for fen in checkmate_states
        ]
    )
    # Filter out illegal moves
    df = df[[bool(chess.Board(fen).legal_moves) for fen in df["fen"]]]
    logger.info(f"Writing dataset to {args.output_path}")
    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--pgn_path",
        type=str,
        help="Path to the PGN file containing the chess games.",
    )
    argparser.add_argument(
        "--output_path",
        type=str,
        default=base_path / "data" / "fen_dataset.csv",
    )
    argparser.add_argument(
        "--n_games",
        type=int,
        default=1000,
        help="Number of games to sample from the database.",
    )
    argparser.add_argument(
        "--skip_n_games",
        type=int,
        default=0,
        help="Number of games to skip at the beginning of the PGN file.",
    )
    argparser.add_argument(
        "--advantage_threshold",
        type=float,
        default=0.02,
        help="Threshold for determining advantageous material balance.",
    )
    argparser.add_argument(
        "--sample_size_per_category",
        type=int,
        default=500,
        help="Number of samples to take from each of 6 categories (early, mid, endgame, advantageous, disadvantageous, checkmate).",
    )
    args = argparser.parse_args()
    main(args)
