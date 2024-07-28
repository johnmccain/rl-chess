import chess
import numpy as np
import torch


def material_balance(board: chess.Board):
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    return (
        chess.popcount(white & board.pawns)
        - chess.popcount(black & board.pawns)
        + 3
        * (
            chess.popcount(white & board.knights)
            - chess.popcount(black & board.knights)
        )
        + 3
        * (
            chess.popcount(white & board.bishops)
            - chess.popcount(black & board.bishops)
        )
        + 5
        * (chess.popcount(white & board.rooks) - chess.popcount(black & board.rooks))
        + 9
        * (chess.popcount(white & board.queens) - chess.popcount(black & board.queens))
    )


def evaluate_fitness(board: chess.Board, player: chess.Color) -> float:
    """
    Evaluate the fitness of a given board state for the given player.
    :param board: The board state to evaluate.
    :param player: The player for whom to evaluate the board state.
    """
    # Material balance
    material = material_balance(board)

    # Checkmate
    if board.is_checkmate():
        return 1000 if board.turn == player else -1000

    # Stalemate
    if board.is_stalemate():
        return 0

    # Insufficient material
    if board.is_insufficient_material():
        return 0

    # Draw by 50-move rule
    if board.is_seventyfive_moves():
        return 0

    return material


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
            promotion_piece = chess.QUEEN

    # Create the move
    move = chess.Move(from_square, to_square, promotion=promotion_piece)

    # Check if the move is legal
    if move in board.legal_moves:
        return move
    else:
        return None  # Returning None if the move is not legal
