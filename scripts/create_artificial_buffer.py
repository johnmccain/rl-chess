from typing import Generator
import uuid
import collections
import logging
import os
import pathlib
import pickle
import random
import sys
import gzip
import chess
import chess.pgn
import numpy as np
import pandas as pd
import torch
from chess import Board, Move
from tqdm import tqdm
import argparse

from rl_chess import base_path
from rl_chess.config.config import AppConfig
from rl_chess.evaluation.stockfish_evaluator import StockfishEvaluator
from rl_chess.inference.inference import ChessAgent
from rl_chess.modeling.chess_cnn import ChessCNN
from rl_chess.modeling.chess_transformer import ChessTransformer
from rl_chess.modeling.experience_buffer import (
    ExperienceBuffer,
    ExperienceRecord,
    FullEvaluationRecord,
)
from rl_chess.modeling.utils import (
    board_to_tensor,
    calculate_reward,
    get_legal_moves_mask,
    index_to_move,
    move_to_index,
    tensor_to_board,
)

app_config = AppConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

def write_buffer_to_file(buffer: list[ExperienceRecord] | list[FullEvaluationRecord], buffer_path: pathlib.Path):
    file_name = f"{uuid.uuid4()}.pkl.gzip"
    file_path = str(buffer_path / file_name)
    with gzip.open(file_path, "wb") as f:
        buffer = [record.make_serializeable() for record in buffer]
        random.shuffle(buffer)
        pickle.dump(buffer, f)


def main(args: argparse.Namespace):
    pgn_path = base_path / "data" / "DATABASE4U.pgn"
    NUM_BUFFERS = 1024 * 8
    BUFFER_SIZE = 2000
    buffers_created = 0
    mode: str = args.mode
    assert mode in ("experiences", "full_evaluations")
    output_path_str: str | None = args.output_path
    if output_path_str is None:
        if mode == "experiences":
            output_path = base_path / "data" / "curriculum_buffers"
        else:
            output_path = base_path / "data" / "full_evaluations"
    else:
        output_path = base_path / "data" / output_path_str

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    skip: int = args.skip or 0

    progress = tqdm(total=NUM_BUFFERS, desc="Creating buffers")

    states_generator = generate_states(pgn_path, skip=skip)
    states_buffer = []
    experiences_buffer = []
    for state in states_generator:
        states_buffer.append(state)
        if len(states_buffer) >= 400:
            if mode == "full_evaluations":
                experiences = create_full_evaluation(states_buffer)
            else:
                experiences = create_experiences(states_buffer)
            experiences_buffer.extend(experiences)
            states_buffer = []

        if len(experiences_buffer) >= BUFFER_SIZE:
            # experiences = create_experiences(states_buffer)
            experiences, experiences_buffer = experiences_buffer[:BUFFER_SIZE], experiences_buffer[BUFFER_SIZE:]
            write_buffer_to_file(experiences, output_path)
            buffers_created += 1
            progress.update(1)
        if buffers_created >= NUM_BUFFERS:
            break


def generate_states(pgn_path: str, skip: int = 0) -> Generator[tuple[tuple[str, str], tuple[str, str], tuple[str, str], int], None, None]:
    done = False
    with open(pgn_path, "r", errors="ignore") as f:
        if skip:
            skip_progress = tqdm(total=skip, desc="Skipping games", leave=False)
            while skip > 0:
                line = next(f)
                if line.startswith("[Event"):
                    skip -= 1
                    skip_progress.update(1)
        while not done:
            try:
                buf = collections.deque(maxlen=3)
                game = chess.pgn.read_game(f)
                if not game:
                    done = True
                    break
                if game.errors:
                    # Skip games with errors
                    continue
                board = game.board()
                turn = 0
                for move in game.mainline_moves():
                    buf.append((board.fen(), move.uci()))
                    board.push(move)
                    if len(buf) == 3:
                        # Turn - 2 because we want the turn for buf[0]
                        yield (buf[0], buf[1], buf[2], turn - 2)
                    turn += 1

            except Exception as e:
                logger.warning(f"Error processing game: {e}")
                continue

def create_experiences(states: list[tuple[tuple[str, str], tuple[str, str], tuple[str, str], int]]) -> list[ExperienceRecord]:
    records = []
    with torch.no_grad():
        for (fen1, uci1), (fen2, uci2), (fen3, uci3), move_count in tqdm(states, desc="Creating experiences", leave=False):
            # Three moves--player, opponent, player
            board1 = chess.Board(fen1)
            board2 = chess.Board(fen2)
            board3 = chess.Board(fen3)
            move1 = chess.Move.from_uci(uci1)

            state1 = board_to_tensor(board1, board1.turn).to(device)
            legal_moves_mask1 = get_legal_moves_mask(board1).to(device)
            done1 = board1.is_game_over()

            done2 = board2.is_game_over()

            state3 = board_to_tensor(board3, board3.turn).to(device)
            legal_moves_mask3 = get_legal_moves_mask(board3).to(device)

            move_index1 = move_to_index(move1)

            reward1 = calculate_reward(board1, move1, move_count=move_count)

            next_state = state3
            next_legal_moves_mask = legal_moves_mask3

            done = done1
            opp_done = done2

            record = ExperienceRecord(
                q_diff=0.0,
                state=state1,
                legal_moves_mask=legal_moves_mask1,
                action=torch.tensor(move_index1, device=device).unsqueeze(0),
                reward=reward1,
                next_state=next_state,
                next_legal_moves_mask=next_legal_moves_mask,
                done=done,
                opp_done=opp_done,
                pred_q_values=None,
                max_next_q=None,
                color=board1.turn,
                move_count=move_count,
            )
            records.append(record)
    return records


def create_full_evaluation(
    states: list[tuple[tuple[str, str], tuple[str, str], tuple[str, str], int]]
) -> list[FullEvaluationRecord]:
    records = []
    with torch.no_grad():
        for (fen1, uci1), (fen2, uci2), (fen3, uci3), move_count in states:
            fen = fen3
            # Three moves--player,opponent,player
            board = chess.Board(fen)

            state = board_to_tensor(board, board.turn).to(device)
            legal_moves_mask = get_legal_moves_mask(board).to(device)
            done = board.is_game_over()

            rewards = torch.full_like(legal_moves_mask, -1e6)

            for move in board.legal_moves:
                rewards[move_to_index(move)] = calculate_reward(board, move, move_count=move_count)

            record = FullEvaluationRecord(
                fen=fen,
                state=state,
                legal_moves_mask=legal_moves_mask,
                rewards=rewards,
                done=done,
                color=board.turn,
                move_count=move_count,
            )
            records.append(record)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="experiences or full_evaluations")
    parser.add_argument("--output_path", type=str, help="Output directory for the buffer files (always in data folder)", default=None)
    parser.add_argument("--skip", type=int, help="Skip the first N games in the PGN", default=0)
    args = parser.parse_args()

    main(args)
