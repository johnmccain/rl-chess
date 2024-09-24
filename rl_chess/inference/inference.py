import logging
import random
import pickle

import chess
import torch

from rl_chess import base_path
from rl_chess.config.config import AppConfig
from rl_chess.modeling.chess_cnn import ChessCNN
from rl_chess.modeling.chess_transformer import ChessTransformer
from rl_chess.modeling.chess_ensemble import EnsembleCNNTransformer
from rl_chess.modeling.utils import (
    board_to_tensor,
    calculate_reward,
    get_legal_moves_mask,
    index_to_move,
)

logger = logging.getLogger(__name__)


class ChessAgent:
    def __init__(
        self, app_config: AppConfig = AppConfig(), device: str = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.model: ChessTransformer | ChessCNN | EnsembleCNNTransformer = self.load_model(app_config)
        self.model.to(self.device)

    def load_model(self, app_config: AppConfig = AppConfig()) -> ChessTransformer | ChessCNN | EnsembleCNNTransformer:
        """
        Load a trained model from disk.

        :param app_config: The application configuration.

        :returns: The trained model.
        """
        if app_config.APP_MODEL_NAME.endswith(".pt"):
            # Load state dict
            # Use config to determine model type
            if app_config.MODEL_CLASS == "ChessTransformer":
                model = ChessTransformer(
                    d_model=256,
                    nhead=8,
                    num_layers=4,
                    dim_feedforward=512,
                    dropout=0.1,
                )
            elif app_config.MODEL_CLASS == "ChessCNN":
                model = ChessCNN(
                    num_filters=app_config.MODEL_CNN_NUM_FILTERS,
                    num_residual_blocks=app_config.MODEL_CNN_RESIDUAL_BLOCKS,
                    negative_slope=app_config.MODEL_CNN_NEGATIVE_SLOPE,
                    dropout=app_config.MODEL_CNN_DROPOUT,
                )
            elif app_config.MODEL_CLASS == "EnsembleCNNTransformer":
                model = EnsembleCNNTransformer(
                    cnn=ChessCNN(
                        num_filters=app_config.MODEL_CNN_NUM_FILTERS,
                        num_residual_blocks=app_config.MODEL_CNN_RESIDUAL_BLOCKS,
                        negative_slope=app_config.MODEL_CNN_NEGATIVE_SLOPE,
                        dropout=app_config.MODEL_CNN_DROPOUT,
                    ),
                    transformer=ChessTransformer(
                        d_model=app_config.MODEL_TRANSFORMER_D_MODEL,
                        nhead=app_config.MODEL_TRANSFORMER_NUM_HEADS,
                        num_layers=app_config.MODEL_TRANSFORMER_NUM_LAYERS,
                        dim_feedforward=app_config.MODEL_TRANSFORMER_DIM_FEEDFORWARD,
                        dropout=app_config.MODEL_TRANSFORMER_DROPOUT,
                        freeze_pos=app_config.MODEL_TRANSFORMER_FREEZE_POS,
                        # add_global=app_config.MODEL_TRANSFORMER_ADD_GLOBAL,
                    ),
                )
            else:
                raise ValueError(f"Invalid model class {app_config.MODEL_CLASS}.")
            with open(base_path / app_config.APP_OUTPUT_DIR / app_config.APP_MODEL_NAME, "rb") as f:
                state_dict = torch.load(f, map_location=self.device)
                model.load_state_dict(state_dict)
        elif app_config.APP_MODEL_NAME.endswith(".pkl"):
            # Load model
            with open(base_path / app_config.APP_OUTPUT_DIR / app_config.APP_MODEL_NAME, "rb") as f:
                model = pickle.load(f)
                model.to(self.device)
                assert isinstance(model, (ChessTransformer, ChessCNN, EnsembleCNNTransformer))
        else:
            raise ValueError("Invalid model file extension.")
        model.eval()
        return model

    def select_top_rated_move(self, board: chess.Board) -> chess.Move:
        """
        Perform inference using the ChessTransformer model and return the best legal move.

        :param model: The trained ChessTransformer model.
        :param board: The current chess board state.

        :returns: The top-rated legal move as determined by the model.
        """

        # Convert the current board state to a tensor
        current_state = board_to_tensor(board, board.turn).to(self.device)
        current_state = current_state.unsqueeze(0)  # Add a batch dimension

        with torch.no_grad():  # Disable gradient computation for inference
            # Get the model's predictions for the current state
            q_values, logits = self.model(
                current_state,
                move_count=torch.tensor(
                    [board.fullmove_number * 2 - int(board.turn)], device=self.device
                ),
            )
            q_values = q_values.view(-1)  # Flatten the q values

        # Generate a mask for the legal moves
        legal_moves_mask = get_legal_moves_mask(board).to(self.device)
        masked_q_values = q_values.masked_fill(legal_moves_mask == 0, -1e10)

        # Find the index of the highest scoring legal move
        best_move_index = torch.argmax(masked_q_values).item()

        # Convert this index back to a chess move
        best_move = index_to_move(best_move_index, board)
        logger.debug(f"Best move: {best_move}")

        best_move_score = masked_q_values[best_move_index].item()
        logger.debug(f"Best move score: {best_move_score}")

        return best_move

    def select_topk_sampling_move(self, board: chess.Board, k=5) -> chess.Move:
        """
        Perform inference using the ChessTransformer model and return a move from a weighted random sample of the top 5 rated moves.

        :param model: The trained ChessTransformer model.
        :param board: The current chess board state.

        :returns: The selected move
        """

        current_state = board_to_tensor(board, board.turn).to(self.device)
        current_state = current_state.unsqueeze(0)

        with torch.no_grad():
            q_values, logits = self.model(
                current_state,
                move_count=torch.tensor(
                    [board.fullmove_number * 2 - int(board.turn)], device=self.device
                ),
            )
            q_values = q_values.view(-1)

        legal_moves_mask = get_legal_moves_mask(board).to(self.device)
        masked_q_values = q_values.masked_fill(legal_moves_mask == 0, -1e10)
        topk = torch.topk(masked_q_values, k)
        topk_indices, topk_values = (
            topk.indices.cpu().numpy(),
            topk.values.cpu().numpy(),
        )
        topk_indices = [
            index for index, value in zip(topk_indices, topk_values) if value > -1e6
        ]
        topk_values = torch.tensor([value for value in topk_values if value > -1e6])
        topk_moves = [index_to_move(index, board) for index in topk_indices]

        # Calculate the probabilities for each move
        topk_probs = torch.softmax(topk_values, dim=0)
        selected_move = random.choices(topk_moves, weights=topk_probs)[0]

        return selected_move

    def rate_moves_from_position(
        self, board: chess.Board, square: chess.Square
    ) -> dict[chess.Move, float]:
        """
        Perform inference using the ChessTransformer model and return a dict mapping from legal moves to their scores given a board state and a square.
        """

        # Convert the current board state to a tensor
        current_state = board_to_tensor(board, board.turn)
        current_state = current_state.unsqueeze(0)  # Add a batch dimension

        with torch.no_grad():  # Disable gradient computation for inference
            # Get the model's predictions for the current state
            q_values, logits = self.model(
                current_state,
                move_count=torch.tensor(
                    [board.fullmove_number * 2 - int(board.turn)], device=self.device
                ),
            )
            q_values = q_values.view(-1)  # Flatten the q_values

        # Generate a mask for the legal moves
        legal_moves_mask = get_legal_moves_mask(board)
        masked_q_values = q_values.masked_fill(legal_moves_mask == 0, -1e10)

        move_scores = {}
        for index, score in enumerate(masked_q_values):
            if index // 64 != square:
                continue
            if score < -1e6:
                continue
            move = index_to_move(index, board)
            move_scores[move] = score.item()

        return move_scores


class MinimaxAgent:
    def __init__(self, depth: int = 3):
        self.depth = depth
        self.max_memo = {}
        self.min_memo = {}

    def alpha_beta_search(
        self,
        board: chess.Board,
        depth: int,
        alpha: float = float("-inf"),
        beta: float = float("inf"),
        maximizing_player: bool = True,
    ) -> tuple[chess.Move, float]:
        if maximizing_player and board.fen() in self.max_memo:
            return self.max_memo[board.fen()]
        if not maximizing_player and board.fen() in self.min_memo:
            return self.min_memo[board.fen()]
        if depth == 0 or board.is_game_over():
            return None, calculate_reward(board, chess.Move.null())

        best_move = None
        best_moves = []
        max_eval = float("-inf")
        if maximizing_player:
            for move in board.legal_moves:
                board.push(move)
                _, eval = self.alpha_beta_search(board, depth - 1, alpha, beta, False)
                board.pop()
                # Negate the evaluation since it's from the opponent's perspective
                eval = -eval
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                    best_moves = [move]
                elif eval == max_eval:
                    best_moves.append(move)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            if len(best_moves) > 1:
                best_move = random.choice(best_moves)
            self.max_memo[board.fen()] = (best_move, max_eval)
            return best_move, max_eval
        else:
            min_eval = float("inf")
            for move in board.legal_moves:
                board.push(move)
                _, eval = self.alpha_beta_search(board, depth - 1, alpha, beta, True)
                board.pop()
                # Negate the evaluation since it's from the opponent's perspective
                eval = -eval
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                    best_moves = [move]
                elif eval == max_eval:
                    best_moves.append(move)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            if len(best_moves) > 1:
                best_move = random.choice(best_moves)
            self.min_memo[board.fen()] = (best_move, min_eval)
            return best_move, min_eval

    def select_top_rated_move(self, board: chess.Board) -> chess.Move:
        best_move, _ = self.alpha_beta_search(board, self.depth)
        return best_move
