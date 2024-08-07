import logging

import chess
import numpy as np

from rl_chess.config.config import AppConfig
from rl_chess.inference.inference import ChessAgent
from stockfish import Stockfish

logger = logging.getLogger(__name__)


class StockfishEvaluator:
    def __init__(
        self,
        app_config: AppConfig = AppConfig(),
    ):
        logger.info(f"Loading Stockfish from {app_config.STOCKFISH_PATH}")
        self.stockfish = Stockfish(app_config.STOCKFISH_PATH)

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
        self, chess_agent: ChessAgent, games_per_elo=10, elo_range=range(400, 1200, 200)
    ) -> tuple[int, int, int, int]:
        results = []
        for elo in elo_range:
            self.set_elo_rating(elo)
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

            results.append((elo, wins, losses, draws))
            print(f"ELO {elo}: {wins} wins, {losses} losses, {draws} draws")

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
