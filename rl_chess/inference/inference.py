import logging

import chess
import torch

from rl_chess import base_path
from rl_chess.config.config import AppConfig
from rl_chess.modeling.chess_transformer import ChessTransformer
from rl_chess.modeling.utils import board_to_tensor, get_legal_moves_mask, index_to_move

logger = logging.getLogger(__name__)


class ChessAgent:
    def __init__(self, app_config: AppConfig = AppConfig()):
        self.model = self.load_model(app_config)

    def load_model(self, app_config: AppConfig = AppConfig()) -> ChessTransformer:
        """
        Load a trained ChessTransformer model from disk.

        :param app_config: The application configuration.

        :returns: The trained ChessTransformer model.
        """
        model = ChessTransformer(
            d_model=256,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1,
        )
        logger.info(
            f"Loading model from {base_path / app_config.APP_OUTPUT_DIR / app_config.APP_MODEL_NAME}"
        )
        model.load_state_dict(
            torch.load(
                base_path / app_config.APP_OUTPUT_DIR / app_config.APP_MODEL_NAME
            )
        )
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
        current_state = board_to_tensor(board, board.turn)
        current_state = current_state.unsqueeze(0)  # Add a batch dimension

        with torch.no_grad():  # Disable gradient computation for inference
            # Get the model's predictions for the current state
            logits: torch.Tensor = self.model(current_state)
            logits = logits.view(-1)  # Flatten the logits

        # Generate a mask for the legal moves
        legal_moves_mask = get_legal_moves_mask(board)
        masked_logits = logits.masked_fill(legal_moves_mask == 0, -1e10)

        # Find the index of the highest scoring legal move
        best_move_index = torch.argmax(masked_logits).item()

        # Convert this index back to a chess move
        best_move = index_to_move(best_move_index, board)
        logger.info(f"Best move: {best_move}")

        best_move_score = masked_logits[best_move_index].item()
        logger.info(f"Best move score: {best_move_score}")

        return best_move
