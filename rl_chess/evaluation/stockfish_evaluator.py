import logging
import collections
from typing import Iterable

from tqdm import tqdm
import chess
import numpy as np
import pandas as pd
from stockfish import Stockfish

from rl_chess.config.config import AppConfig
from rl_chess.inference.inference import ChessAgent

logger = logging.getLogger(__name__)


class StockfishEvaluator:
    def __init__(
        self,
        app_config: AppConfig = AppConfig(),
    ):
        logger.info(f"Loading Stockfish from {app_config.STOCKFISH_PATH}")
        self.stockfish = Stockfish(app_config.STOCKFISH_PATH)
        self.set_depth(app_config.STOCKFISH_DEPTH)

    def set_elo_rating(self, elo: int):
        self.stockfish.set_elo_rating(elo)

    def set_depth(self, depth: int):
        self.stockfish.set_depth(depth)

    def move(self, board: chess.Board) -> chess.Move:
        self.stockfish.set_fen_position(board.fen())
        move_ucis = [m["Move"] for m in self.stockfish.get_top_moves(5)]
        if not move_ucis:
            return None  # Handle cases where no move is available (game over scenarios)
        while move_ucis:
            move_uci = move_ucis.pop(0)
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                return move
        return None

    def rate_action(self, board: chess.Board, move: chess.Move) -> float:
        """
        Evaluate the quality of a move according to the Stockfish engine.
        """
        # Evaluate the current board position
        self.stockfish.set_fen_position(board.fen())
        before_eval = self.stockfish.get_evaluation().get("value", 0.0)

        # Apply the move to the board
        board.push(move)

        # Evaluate the new board position
        self.stockfish.set_fen_position(board.fen())
        after_eval = self.stockfish.get_evaluation().get("value", 0.0)

        # Undo the move to return to the original position
        board.pop()

        # Calculate the difference in evaluation
        return after_eval - before_eval

    def evaluate_move_ratings(
        self,
        chess_agent: ChessAgent,
        fen_dataset: pd.DataFrame,
        blunder_threshold: int = -100,
        progress: bool = True,
    ) -> tuple[float, float, float, dict]:
        """
        Evaluate the chess model using the given FEN dataset based on stockfish evaluations of individual moves.
        Evaluates as per the following KPIs:
        1. Average Move Quality (AMQ) - Average evaluation of the model's moves
        2. Blunder Rate (BR) - Rate of moves rated below the blunder threshold
        3. Optimality Rate (OR) - Rate of optimal moves

        :param chess_agent:
        :param fen_dataset: dataframe with two columns: 'fen' and 'type'. 'type' is used for logging more granular results.
        :param blunder_threshold: Threshold for considering a move as a blunder (in centipawns).
        :param progress: Whether to display a progress bar.
        :returns: Tuple containing the Average Move Quality (AMQ), Blunder Rate (BR), Optimality Rate (OR), and a dictionary of KPIs per type.
        """
        total_move_quality = 0.0
        blunders = 0
        optimal_moves = 0
        total_moves = len(fen_dataset)
        quality_per_type = collections.defaultdict(float)
        blunders_per_type = collections.defaultdict(int)
        optimal_moves_per_type = collections.defaultdict(int)
        count_per_type = collections.defaultdict(int)

        iterable = zip(fen_dataset["fen"], fen_dataset["type"])
        if progress:
            iterable = tqdm(iterable, total=total_moves, desc="Evaluating moves")
        for fen, type in iterable:
            board = chess.Board(fen)
            count_per_type[type] += 1

            model_move = chess_agent.select_top_rated_move(board)

            move_quality = self.rate_action(board, model_move)
            total_move_quality += move_quality
            quality_per_type[type] += move_quality

            if move_quality < blunder_threshold:
                blunders += 1
                blunders_per_type[type] += 1

            # Check for optimal move (if the move matches Stockfish's top move)
            stockfish_best_move = self.move(board)
            if model_move == stockfish_best_move:
                optimal_moves += 1
                optimal_moves_per_type[type] += 1

        # Calculate the KPIs
        average_move_quality = total_move_quality / total_moves
        blunder_rate = blunders / total_moves  # Rate of moves that were blunders
        optimality_ratio = optimal_moves / total_moves  # Rate of optimal moves

        kpis_per_type = {}
        for type in count_per_type:
            kpis_per_type[type] = {
                "AverageMoveQuality": quality_per_type[type] / count_per_type[type],
                "BlunderRate": blunders_per_type[type] / count_per_type[type],
                "OptimalityRate": optimal_moves_per_type[type] / count_per_type[type],
            }

        return average_move_quality, blunder_rate, optimality_ratio, kpis_per_type

    def take_action(self, board: chess.Board) -> int:
        """
        Select the best move according to the Stockfish engine. Return as an integer such that
        1. Dividing by 64 gives the 'from' square
        2. Modulo 64 gives the 'to' square
        """
        move = self.move(board)
        if move is None:
            raise ValueError("No legal moves available.")
        from_index = move.from_square
        to_index = move.to_square
        return from_index * 64 + to_index

    def simulate_games(
        self,
        chess_agent: ChessAgent,
        games_per_elo=10,
        elo_range: Iterable[int] | None = None,
        depth_range: Iterable[int] | None = None,
    ) -> tuple[int, int, int, int]:
        results = []
        if elo_range is None and depth_range is None:
            raise ValueError("elo_range or depth_range must be provided.")
        if elo_range is not None and depth_range is not None:
            raise ValueError(
                "elo_range and depth_range cannot be provided simultaneously."
            )
        elif elo_range is not None:
            use_elo = True
        else:
            use_elo = False

        difficulty_range = elo_range if use_elo else depth_range

        for difficulty in difficulty_range:
            if use_elo:
                self.set_elo_rating(difficulty)
            else:
                self.set_depth(difficulty)
            wins, losses, draws = 0, 0, 0

            for _ in range(games_per_elo):
                board = chess.Board()
                while not board.is_game_over():
                    if board.turn == chess.WHITE:
                        move = chess_agent.select_top_rated_move(
                            board
                        )  # model moves as white
                        board.push(move)
                    else:
                        move = self.move(board)  # Stockfish moves as black
                        board.push(move)
                    if move is None:  # no move possible, game over
                        break

                result = board.result()
                if result == "1-0":
                    wins += 1
                elif result == "0-1":
                    losses += 1
                else:
                    draws += 1

            results.append((difficulty, wins, losses, draws))
            print(
                f"Difficulty {difficulty}: {wins} wins, {losses} losses, {draws} draws"
            )

        return results

    def estimate_elo(self, results: list[tuple[int, int, int, int]]) -> int | None:
        """
        Estimate the ELO rating of the model based on the results of simulated games against Stockfish.

        :param results: A list of tuples containing the ELO rating, wins, losses, and draws against Stockfish.
        :returns: The estimated ELO rating of the model.
        """
        elos = []
        win_rates = []

        for elo, wins, losses, draws in results:
            total_games = wins + losses + draws
            win_rate = (wins + 0.5 * draws) / total_games  # draw = 0.5
            elos.append(elo)
            win_rates.append(win_rate)

        elos = np.array(elos)
        win_rates = np.array(win_rates)

        # find the index of the win rate just below 50% and just above 50%
        if not np.any(win_rates >= 0.5):
            logger.warning(
                "Model does not reach 50% win rate against any tested ELO ratings."
            )
            return None
        if not np.any(win_rates <= 0.5):
            logger.warning(
                "Model always wins more than 50% against all tested ELO ratings."
            )
            return None

        # find the first point where win rate exceeds 50%, then interpolate
        above_50_index = np.where(win_rates >= 0.5)[0][0]
        below_50_index = above_50_index - 1

        if above_50_index > 0:
            elo_estimate = elos[below_50_index] + (0.5 - win_rates[below_50_index]) * (
                elos[above_50_index] - elos[below_50_index]
            ) / (win_rates[above_50_index] - win_rates[below_50_index])
        else:
            elo_estimate = elos[
                0
            ]  # if the model is too strong even at the lowest tested ELO

        return elo_estimate
