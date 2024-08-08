import chess
from chess import Board, Move
from rl_chess.inference.inference import ChessAgent, MinimaxAgent
from rl_chess.config.config import AppConfig


chess_agent = ChessAgent(
    app_config=AppConfig(),
    device="cuda:0",
)
minimax_agent = MinimaxAgent(1)
app_config = AppConfig()

def simulate_game(chess_agent: ChessAgent, minimax_agent: MinimaxAgent, depth: int) -> tuple[int, int]:
    board = Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = chess_agent.select_topk_sampling_move(board, k=3)
        else:
            move = minimax_agent.select_top_rated_move(board)
        board.push(move)
    if board.result() == "1-0":
        return 1, 0
    elif board.result() == "0-1":
        return 0, 1
    else:
        return 0, 0

N_GAMES = 100

for depth in range(1, 6):
    print(f"Depth: {depth}")
    results = [simulate_game(chess_agent, minimax_agent, depth) for _ in range(N_GAMES)]
    agent_wins = sum(result[0] for result in results)
    minimax_wins = sum(result[1] for result in results)
    print(f"Results: agent {agent_wins}, minimax {minimax_wins}")
    print()
