import collections
import logging
import os
import pathlib
import pickle
import random
import sys

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
from rl_chess.modeling.experience_buffer import ExperienceBuffer, ExperienceRecord
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

pgn_path = base_path / "data" / "SovietChamp1974.pgn"
n_games = 200
states = []

with open(pgn_path, "r") as f:
    for _ in range(n_games):
        try:
            buf = collections.deque(maxlen=3)
            game = chess.pgn.read_game(f)
            if not game:
                break
            if game.errors:
                # Skip games with errors
                continue
            board = game.board()
            for move in game.mainline_moves():
                last_board = board.copy()
                buf.append((board.fen(), move.uci()))
                board.push(move)
                if len(buf) == 3:
                    states.append((buf[0], buf[1], buf[2]))

        except Exception as e:
            logger.warning(f"Error processing game: {e}")
            continue

model = ChessCNN(
    num_filters=app_config.MODEL_NUM_FILTERS,
    num_residual_blocks=app_config.MODEL_RESIDUAL_BLOCKS,
).to(device)

pgn_path = base_path / "data" / "SovietChamp1974.pgn"
n_games = 200
states = []

with open(pgn_path, "r") as f:
    for _ in range(n_games):
        try:
            buf = collections.deque(maxlen=3)
            game = chess.pgn.read_game(f)
            if not game:
                break
            if game.errors:
                # Skip games with errors
                continue
            board = game.board()
            for move in game.mainline_moves():
                last_board = board.copy()
                buf.append((board.fen(), move.uci()))
                board.push(move)
                if len(buf) == 3:
                    states.append((buf[0], buf[1], buf[2]))

        except Exception as e:
            logger.warning(f"Error processing game: {e}")
            continue

records = []

model.eval()
with torch.no_grad():
    for (fen1, uci1), (fen2, uci2), (fen3, uci3) in tqdm(states):
        # Three moves--player,opponent,player
        board1 = chess.Board(fen1)
        board2 = chess.Board(fen2)
        board3 = chess.Board(fen3)
        move1 = chess.Move.from_uci(uci1)
        move2 = chess.Move.from_uci(uci2)
        move3 = chess.Move.from_uci(uci3)

        state1 = board_to_tensor(board1, board1.turn).to(device)
        legal_moves_mask1 = get_legal_moves_mask(board1).to(device)
        done1 = board1.is_game_over()

        state2 = board_to_tensor(board2, board2.turn).to(device)
        legal_moves_mask2 = get_legal_moves_mask(board2).to(device)
        done2 = board2.is_game_over()

        state3 = board_to_tensor(board3, board3.turn).to(device)
        legal_moves_mask3 = get_legal_moves_mask(board3).to(device)
        done3 = board3.is_game_over()

        masked_q_values1 = torch.zeros((1, 4096), dtype=torch.float32, device=device)

        move_index1 = move_to_index(move1)
        move_index2 = move_to_index(move2)
        move_index3 = move_to_index(move3)

        reward1 = calculate_reward(board1, move1)
        reward2 = calculate_reward(board2, move2)
        reward3 = calculate_reward(board3, move3)

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
            pred_q_values=masked_q_values1,
            max_next_q=None,
            color=board1.turn,
        )
        records.append(record)

records = random.sample(records, 64 * 64)

with open("buf.pkl", "wb") as f:
    pickle.dump([record.make_serializeable() for record in records], f)
