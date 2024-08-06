import datetime
import logging
import random

import chess
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rl_chess import base_path
from rl_chess.config.config import AppConfig
from rl_chess.modeling.chess_cnn import ChessCNN  # Import the new ChessCNN model
from rl_chess.modeling.utils import (
    board_to_tensor,
    calculate_reward,
    get_legal_moves_mask,
    index_to_move,
)
from rl_chess.evaluation.stockfish_evaluator import StockfishEvaluator

app_config = AppConfig()

logging.basicConfig(
    level=app_config.LOG_LEVEL,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def write_log(
    model_timestamp: str,
    board: chess.Board,
    move: chess.Move,
    score: float,
    episode: int,
    total_loss: float,
    loss: float,
):
    """
    Detailed move-by-move logging for debugging.
    """
    with open("log.csv", "a") as f:
        f.write(
            f"{model_timestamp},{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{episode},{board.fen()},{move},{score},{total_loss},{loss}\n"
        )


def sample_opening_state(openings_df: pd.DataFrame) -> chess.Board:
    """
    Sample a random opening move from the dataset.
    """
    opening = openings_df.sample()
    board = chess.Board(opening["fen"].values[0])
    return board


def train_deep_q_network(
    model: ChessCNN,
    episodes: int,
    app_config: AppConfig = AppConfig(),
):
    stockfish_evaluator = StockfishEvaluator()
    stockfish_evaluator.set_depth(10)
    openings_df = pd.read_csv(base_path / "data/openings_fen7.csv")
    model_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if torch.cuda.is_available():
        logger.info("CUDA available, using GPU")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        logger.info("MPS available, using MPS")
        device = torch.device("mps")
    else:
        logger.info("CUDA unavailable, using CPU")
        device = torch.device("cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=app_config.MODEL_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=episodes, eta_min=1e-6
    )

    loss_fn = F.mse_loss
    epsilon = 1.0
    writer = SummaryWriter(
        log_dir=base_path
        / app_config.APP_TENSORBOARD_DIR
        / f"qchess_{model_timestamp}",
    )

    # Save hparams to Tensorboard
    hparams = {
        "num_filters": model.num_filters,
        "num_residual_blocks": model.num_residual_blocks,
        "gamma": app_config.MODEL_GAMMA,
        "initial_gamma": app_config.MODEL_INITIAL_GAMMA,
        "gamma_ramp_steps": app_config.MODEL_GAMMA_RAMP_STEPS,
        "lr": app_config.MODEL_LR,
        "decay": app_config.MODEL_DECAY,
        "clip_grad": app_config.MODEL_CLIP_GRAD,
        "optimizer": str(type(optimizer)),
    }
    writer.add_hparams(
        hparam_dict=hparams,
        metric_dict={},
    )

    gamma_ramp = (
        app_config.MODEL_GAMMA - app_config.MODEL_INITIAL_GAMMA
    ) / app_config.MODEL_GAMMA_RAMP_STEPS

    for episode in range(episodes):
        gamma = min(
            app_config.MODEL_INITIAL_GAMMA + gamma_ramp * episode,
            app_config.MODEL_GAMMA,
        )
        # 25% of the time, start with a random opening state
        if random.random() < 0.25:
            board = sample_opening_state(openings_df)
        else:
            board = chess.Board()
        total_loss = 0.0
        moves = 0

        if episode % 100 == 0:
            write_log(
                model_timestamp=model_timestamp,
                board=board,
                move=None,
                score=None,
                episode=episode,
                total_loss=None,
                loss=None,
            )

        while not board.is_game_over() and moves < app_config.MODEL_MAX_MOVES:
            current_state = board_to_tensor(board, board.turn).to(device)
            current_state = current_state.unsqueeze(0)  # Batch size of 1

            # Predict Q-values
            predicted_q_values: torch.Tensor = model(current_state)

            # Mask illegal moves
            legal_moves_mask = get_legal_moves_mask(board).to(device)
            masked_q_values = predicted_q_values.masked_fill(
                legal_moves_mask == 0, -1e10
            )

            # Epsilon-greedy action selection
            if random.random() > epsilon:
                action = masked_q_values.max(1)[1].view(1, 1)
            elif random.random() < app_config.STOCKFISH_PROB:
                action = torch.tensor(
                    [stockfish_evaluator.take_action(board)], device=device
                ).unsqueeze(0)
            else:
                # Select action randomly with softmax
                action = torch.multinomial(F.softmax(masked_q_values, dim=-1), 1)

            # Take action and observe reward and next state
            move = index_to_move(action, board)
            if move is None:
                logger.warning("Invalid move selected!")
                break
            reward = calculate_reward(board, move)
            # Calculate default reward for opponent move (if opponent can't make a move such as in checkmate or stalemate)
            opp_reward = calculate_reward(board, move, flip_perspective=True)

            # Take the action
            board.push(move)
            opp_next_state = board_to_tensor(board, board.turn).to(device)
            opp_next_state = opp_next_state.unsqueeze(0)

            done = torch.tensor([int(board.is_game_over())], device=device)

            if not done:
                # Select opponent next move
                with torch.no_grad():
                    opp_next_q_values = model(opp_next_state)
                opp_next_legal_moves_mask = get_legal_moves_mask(board).to(device)
                opp_masked_next_q_values = opp_next_q_values.masked_fill(
                    opp_next_legal_moves_mask == 0, -1e10
                )
                if random.random() > epsilon:
                    opp_action = opp_masked_next_q_values.max(1)[1].view(1, 1)
                elif random.random() < app_config.STOCKFISH_PROB:
                    opp_action = torch.tensor(
                        [stockfish_evaluator.take_action(board)], device=device
                    ).unsqueeze(0)
                else:
                    opp_action = torch.multinomial(
                        F.softmax(opp_masked_next_q_values, dim=-1), 1
                    )
                # Take opponent action
                opp_move = index_to_move(opp_action, board)
                if opp_move is None:
                    logger.warning("Invalid opponent move selected!")
                    break
                # Calculate reward for opponent move
                opp_reward = calculate_reward(board, opp_move)
                board.push(opp_move)

                # Compute the next-state max Q-value for active player
                next_state = board_to_tensor(board, board.turn).to(device)
                next_state = next_state.unsqueeze(0)
                next_q_values = model(next_state)
                # Roll back the board state
                board.pop()
                next_legal_moves_mask = get_legal_moves_mask(board).to(device)
                masked_next_q_values = next_q_values.masked_fill(
                    next_legal_moves_mask == 0, -1e10
                )
                max_next_q_values = masked_next_q_values.max(1)[0].detach()
            else:
                max_next_q_values = torch.tensor([0.0], device=device)
            # Compute the target Q-value
            target_q_values = (
                reward - opp_reward + (gamma * max_next_q_values * (1 - done))
            )
            predicted_q = predicted_q_values.gather(1, action)

            # Compute loss
            loss = (
                loss_fn(predicted_q, target_q_values.unsqueeze(1))
                / app_config.MODEL_GRAD_STEPS
            )
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), app_config.MODEL_CLIP_GRAD
            )
            moves += 1

            if episode % 100 == 0:
                write_log(
                    model_timestamp=model_timestamp,
                    board=board,
                    move=move,
                    score=predicted_q.item(),
                    episode=episode,
                    total_loss=total_loss,
                    loss=loss.item(),
                )
            if moves % app_config.MODEL_GRAD_STEPS == 0 or done:
                # Gradient accumulation
                optimizer.step()
                optimizer.zero_grad()

        # Log metrics to Tensorboard
        writer.add_scalar("Loss/Episode", total_loss, episode)
        writer.add_scalar("Loss/Move", total_loss / moves, episode)
        writer.add_scalar("Epsilon/Episode", epsilon, episode)
        writer.add_scalar("Moves/Episode", moves, episode)
        writer.add_scalar("Gamma/Episode", gamma, episode)
        writer.add_scalar("LR/Episode", scheduler.get_last_lr()[0], episode)

        if episode % 10 == 0:
            logger.info(
                f"Episode {episode}, Loss: {total_loss}, Moves: {moves}, Loss/move: {total_loss / moves}"
            )

        if episode % app_config.APP_SAVE_STEPS == 0:
            torch.save(
                model.state_dict(),
                base_path
                / app_config.APP_OUTPUT_DIR
                / f"model_{model_timestamp}_e{episode}.pt",
            )
            # Save optimizer state
            torch.save(
                optimizer.state_dict(),
                base_path
                / app_config.APP_OUTPUT_DIR
                / f"optimizer_{model_timestamp}_e{episode}.pt",
            )

        # Epsilon decay
        epsilon = max(epsilon * app_config.MODEL_DECAY, app_config.MODEL_MIN_EPSILON)

        # Learning rate scheduler
        scheduler.step()

    writer.close()
    logger.info("Training complete")
    model_output_path = (
        base_path / app_config.APP_OUTPUT_DIR / f"model_cnn_{model_timestamp}_final.pt"
    )
    torch.save(model.state_dict(), model_output_path)
    logger.info(f"Model saved to {model_output_path}")


if __name__ == "__main__":
    app_config = AppConfig()
    model = ChessCNN(num_filters=256, num_residual_blocks=12)
    train_deep_q_network(model, episodes=50000, app_config=app_config)
