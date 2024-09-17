import logging

import chess
import numpy as np
import torch

logger = logging.getLogger(__name__)


def calculate_material_score(board: chess.Board) -> float:
    """
    Material score normalized by the total material on the board (from the perspective of the current player).
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        # If the king has no material value, then we run into a situation where (king + queen + pawn) vs (king) is equivalent to (king + pawn) vs (king)
        # This leads to disregarding material advantage in the endgame, which is not desirable
        # Assigning the king a value ensures that the denominator of the total material is always greater than a single player's material
        chess.KING: 10,
    }

    white_material = sum(
        len(board.pieces(piece_type, chess.WHITE)) * value
        for piece_type, value in piece_values.items()
    )
    black_material = sum(
        len(board.pieces(piece_type, chess.BLACK)) * value
        for piece_type, value in piece_values.items()
    )

    total_material = white_material + black_material
    if total_material == 0:
        return 0

    white_material_score = (white_material - black_material) / total_material
    return white_material_score if board.turn == chess.WHITE else -white_material_score


def calculate_move_quality(board: chess.Board, move: chess.Move) -> float:
    """
    Calculate heuristic score for a given move.
    """
    score = 0

    # Reward for controlling the center
    central_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    if move.to_square in central_squares:
        score += 0.005

    # Reward for developing pieces in the opening
    if board.fullmove_number <= 10:
        if board.piece_type_at(move.from_square) in [chess.KNIGHT, chess.BISHOP]:
            score += 0.005

    # Penalize moving the same piece multiple times in the opening
    if board.fullmove_number <= 10:
        if board.move_stack and move.from_square == board.move_stack[-1].to_square:
            score -= 0.005

    return score


CHECKMATE_REWARD = 5.0

def calculate_reward(
    board: chess.Board, move: chess.Move, flip_perspective: bool = False
) -> float:
    player = board.turn
    move_quality = calculate_move_quality(board, move)
    board.push(move)

    if board.is_checkmate():
        reward = CHECKMATE_REWARD if board.turn != player else -CHECKMATE_REWARD
    elif board.is_stalemate() or board.is_insufficient_material():
        reward = 0
    else:
        # Material score is calculated based on the perspective of the moving player, and we just moved
        material_score = -calculate_material_score(board)
        reward = material_score + move_quality
    board.pop()

    if flip_perspective:
        reward = -reward

    return reward


def tensor_to_board(tensor: torch.IntTensor, player: chess.Color) -> chess.Board:
    board = chess.Board(None)  # Create an empty board

    # Mapping from piece IDs to chess.Piece objects
    piece_map = {
        1: chess.Piece(chess.PAWN, player),
        2: chess.Piece(chess.KNIGHT, player),
        3: chess.Piece(chess.BISHOP, player),
        4: chess.Piece(chess.ROOK, player),
        5: chess.Piece(chess.QUEEN, player),
        6: chess.Piece(chess.KING, player),
        7: chess.Piece(chess.PAWN, not player),
        8: chess.Piece(chess.KNIGHT, not player),
        9: chess.Piece(chess.BISHOP, not player),
        10: chess.Piece(chess.ROOK, not player),
        11: chess.Piece(chess.QUEEN, not player),
        12: chess.Piece(chess.KING, not player),
    }

    # Iterate through the tensor and place pieces on the board
    for square, piece_id in enumerate(tensor):
        if piece_id != 0:  # 0 represents an empty square
            board.set_piece_at(square, piece_map[piece_id.item()])

    # Set the turn
    board.turn = player

    # Castling rights and en passant square are not stored in the tensor
    board.clean_castling_rights()
    board.ep_square = None

    return board


def get_piece_id(piece_type: chess.PieceType, is_player_piece: bool) -> int:
    base_id = piece_type  # base IDs for player pieces (1-6)
    if not is_player_piece:
        base_id += 6  # adjust for opponent pieces (7-12)
    return base_id


def board_to_tensor(board: chess.Board, player: chess.Color) -> torch.IntTensor:
    tensor = torch.zeros(64, dtype=torch.int)

    # Process each piece type separately
    for piece_type in range(1, 7):  # chess.PieceType values are from 1 to 6
        # Player pieces
        player_squareset = board.pieces(piece_type, player)
        tensor[list(player_squareset)] = get_piece_id(piece_type, True)

        # Opponent pieces
        opponent_squareset = board.pieces(piece_type, not player)
        tensor[list(opponent_squareset)] = get_piece_id(piece_type, False)

    return tensor


def get_legal_moves_mask(board: chess.Board) -> torch.Tensor:
    """
    Generate a mask for the legal moves in the given board state for use in softmax.

    :param board: The current state of the chess game.
    :returns: A flattened tensor of shape (64*64,) where legal moves are 1 and illegal are 0.
    """
    mask = torch.full(
        (64 * 64,), 0.0
    )  # Initialize with a large negative value for illegal moves
    for move in board.legal_moves:
        from_square = move.from_square
        to_square = move.to_square
        index = from_square * 64 + to_square  # Calculate the flat index
        mask[index] = 1.0  # Set to zero for legal moves
    return mask


def index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    """
    Convert a flat index into a chess move.

    :param index: The index in the flattened action space array.
    :param board: The current chess board state.
    """
    from_index = index // 64  # Dividing by 64 gives the 'from' square
    to_index = index % 64  # Modulo 64 gives the 'to' square

    # Convert numerical index to chess square notation
    from_square = chess.SQUARES[from_index]
    to_square = chess.SQUARES[to_index]

    promotion_piece = None
    if (
        board.piece_at(from_square)
        and board.piece_at(from_square).piece_type == chess.PAWN
    ):
        if (board.turn == chess.WHITE and chess.square_rank(to_square) == 7) or (
            board.turn == chess.BLACK and chess.square_rank(to_square) == 0
        ):
            logger.debug("Promoting pawn")
            promotion_piece = chess.QUEEN

    # Create the move
    move = chess.Move(from_square, to_square, promotion=promotion_piece)

    # Check if the move is legal
    if move in board.legal_moves:
        return move
    else:
        logger.warning(f"Move {move} is not legal!")
        return None  # Returning None if the move is not legal


def move_to_index(move: chess.Move) -> int:
    """
    Convert a chess move into a flat index.

    :param move: The chess move to convert.
    """
    return move.from_square * 64 + move.to_square
