import chess
import torch

from rl_chess.modeling.chess_transformer import ChessTransformer
from rl_chess.modeling.utils import board_to_tensor, get_legal_moves_mask, index_to_move


def select_top_rated_move(model: ChessTransformer, board: chess.Board) -> chess.Move:
    """
    Perform inference using the ChessTransformer model and return the best legal move.

    :param model: The trained ChessTransformer model.
    :param board: The current chess board state.

    :returns: The top-rated legal move as determined by the model.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Convert the current board state to a tensor
    current_state = board_to_tensor(board, board.turn)
    current_state = current_state.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():  # Disable gradient computation for inference
        # Get the model's predictions for the current state
        logits: torch.Tensor = model(current_state)
        logits = logits.view(-1)  # Flatten the logits

    # Generate a mask for the legal moves
    legal_moves_mask = get_legal_moves_mask(board)
    masked_logits = logits.masked_fill(legal_moves_mask == 0, -1e10)

    # Find the index of the highest scoring legal move
    best_move_index = torch.argmax(masked_logits).item()

    # Convert this index back to a chess move
    best_move = index_to_move(best_move_index, board)

    return best_move
