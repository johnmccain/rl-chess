import argparse
import logging
import random
from typing import Iterable, TypeVar

import chess
import chess.pgn
import pandas as pd
from tqdm import tqdm

from rl_chess import base_path
from rl_chess.config.config import AppConfig
from rl_chess.modeling.utils import calculate_material_score

app_config = AppConfig()


logging.basicConfig(level="CRITICAL")

logger = logging.getLogger(__name__)

EARLY_GAME_THRESHOLD = 15
ENDGAME_PIECE_COUNT = 12

T = TypeVar("T")

pgn_path = "data/DATABASE4U.pgn"
output_path = "data/chess_states.csv"
n_games = 100000
skip_n_games = 20000
advantage_threshold = 0.02
sample_size_per_category = None


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


logging.info(f"Reading PGN file from {pgn_path}")

states = []

# Tracking observed states for reservoir sampling
early_game_count = midgame_count = endgame_count = 0
adv_material_count = disadv_material_count = checkmate_count = 0

progress = tqdm(total=n_games)

with open(pgn_path, "r") as f:
    for _ in tqdm(range(skip_n_games), desc="Skipping games"):
        try:
            game = chess.pgn.read_game(f)
        except:
            pass
    for game_num in range(n_games):
        try:
            # Skip games at the beginning of the PGN file (useful if generating a train + test split)

            if game_num % 1000 == 0:
                print(f"Processed {game_num} games; {len(states)} states collected")
            game = chess.pgn.read_game(f)
            if game.errors:
                # Skip games with errors
                continue
            board = game.board()
            last_board = None
            move_count = 0

            for move in game.mainline_moves():
                last_board = board.copy()
                move_count += 1

                phase = categorize_board_state(board, move_count)
                fen = board.fen()
                tags = []

                if phase == "early_game":
                    tags.append("bg")
                elif phase == "midgame":
                    tags.append("mg")
                elif phase == "endgame":
                    tags.append("eg")

                threshold = advantage_threshold
                material_score = calculate_material_score(board)
                if material_score > threshold:
                    tags.append("adv")
                elif material_score < -threshold:
                    tags.append("dis")

                board.push(move)
                if board.is_checkmate() and last_board:
                    tags.append("#m")
                states.append((last_board.fen(), move.uci(), ";".join(tags)))

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.warning(f"Error processing game: {e}")
            continue
        progress.update(1)

random.shuffle(states)
with open(output_path, "w") as f:
    f.write("fen,move,tags\n")
    for state in states:
        f.write(",".join(state) + "\n")
