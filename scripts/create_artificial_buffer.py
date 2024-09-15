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

BUFFER_PATH = base_path / "data" / "full_move_evals"


def write_buffer_to_file(buffer: list[ExperienceRecord]):
    file_name = f"{uuid.uuid4()}.pkl.gzip"
    file_path = str(BUFFER_PATH / file_name)
    with gzip.open(file_path, "wb") as f:
        buffer = [record.make_serializeable() for record in buffer]
        random.shuffle(buffer)
        pickle.dump(buffer, f)


def main():
    pgn_path = base_path / "data" / "DATABASE4U.pgn"
    NUM_BUFFERS = 1024 * 8
    BUFFER_SIZE = 2000
    buffers_created = 0

    progress = tqdm(total=NUM_BUFFERS, desc="Creating buffers")

    states_generator = generate_states(pgn_path)
    states_buffer = []
    buf_progress = tqdm(desc="Creating buffer", total=BUFFER_SIZE, leave=False)
    for state in states_generator:
        states_buffer.append(state)
        buf_progress.update(1)
        if len(states_buffer) >= BUFFER_SIZE:
            # experiences = create_experiences(states_buffer)
            experiences = create_full_evaluation(states_buffer)
            states_buffer = []
            write_buffer_to_file(experiences)
            buffers_created += 1
            progress.update(1)
            buf_progress.reset()
        if buffers_created >= NUM_BUFFERS:
            break


def generate_states(pgn_path: str) -> Generator[tuple[tuple[str, str], tuple[str, str], tuple[str, str]], None, None]:
    done = False
    with open(pgn_path, "r") as f:
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
                for move in game.mainline_moves():
                    buf.append((board.fen(), move.uci()))
                    board.push(move)
                    if len(buf) == 3:
                        yield (buf[0], buf[1], buf[2])

            except Exception as e:
                logger.warning(f"Error processing game: {e}")
                continue

def create_experiences(states: list[tuple[tuple[str, str], tuple[str, str], tuple[str, str]]]) -> list[ExperienceRecord]:
    records = []
    with torch.no_grad():
        for (fen1, uci1), (fen2, uci2), (fen3, uci3) in tqdm(states, desc="Creating experiences", leave=False):
            # Three moves--player,opponent,player
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

            reward1 = calculate_reward(board1, move1)

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
            )
            records.append(record)
    return records


def create_full_evaluation(
    states: list[tuple[tuple[str, str], tuple[str, str], tuple[str, str]]]
) -> list[FullEvaluationRecord]:
    records = []
    with torch.no_grad():
        for (fen1, uci1), (fen2, uci2), (fen3, uci3) in tqdm(
            states, desc="Creating experiences", leave=False
        ):
            # Three moves--player,opponent,player
            board1 = chess.Board(fen1)

            state1 = board_to_tensor(board1, board1.turn).to(device)
            legal_moves_mask1 = get_legal_moves_mask(board1).to(device)
            done = board1.is_game_over()

            rewards = torch.full_like(legal_moves_mask1, -1e6)

            for move in board1.legal_moves:
                rewards[move_to_index(move)] = calculate_reward(board1, move)

            record = FullEvaluationRecord(
                fen=fen1,
                state=state1,
                legal_moves_mask=legal_moves_mask1,
                rewards=rewards,
                done=done,
                color=board1.turn,
            )
            records.append(record)
    return records


if __name__ == "__main__":
    main()
