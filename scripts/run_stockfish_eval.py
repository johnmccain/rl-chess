import chess
from chess import Board, Move

from rl_chess.config.config import AppConfig
from rl_chess.evaluation.stockfish_evaluator import StockfishEvaluator
from rl_chess.inference.inference import ChessAgent

stockfish_evaluator = StockfishEvaluator()
chess_agent = ChessAgent(
    app_config=AppConfig(),
    device="cuda:0",
)
app_config = AppConfig()


results = stockfish_evaluator.simulate_games(
    chess_agent=chess_agent, games_per_elo=10, elo_range=range(100, 300, 100)
)


print("elo: ", stockfish_evaluator.estimate_elo(results))
