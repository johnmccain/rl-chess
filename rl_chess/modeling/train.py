import datetime
import logging
import random

import chess
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_optimizer  # noqa: F401
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from rl_chess import base_path
from rl_chess.config.config import AppConfig
from rl_chess.modeling.chess_transformer import ChessTransformer
from rl_chess.modeling.utils import (
    board_to_tensor,
    calculate_reward,
    get_legal_moves_mask,
    index_to_move,
)

app_config = AppConfig()

logging.basicConfig(
    level=app_config.LOG_LEVEL,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def write_log(
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
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, {episode}, Board('{board.fen()}'), Move('{move}'), {score}, {total_loss}, {loss}\n"
        )


def train_deep_q_network(
    model: ChessTransformer,
    episodes: int,
    app_config: AppConfig = AppConfig(),
):
    model_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # device = torch.device("mps")
    device = torch.device("cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=app_config.MODEL_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=episodes, eta_min=1e-6
    )
    # optimizer = torch_optimizer.Ranger(model.parameters(), lr=app_config.MODEL_LR)

    loss_fn = F.mse_loss
    epsilon = 1.0
    writer = SummaryWriter(
        log_dir=base_path
        / app_config.APP_TENSORBOARD_DIR
        / f"qchess_{model_timestamp}",
    )

    # Save hparams to Tensorboard
    hparams = model.get_hparams()
    hparams.update(
        {
            "gamma": app_config.MODEL_GAMMA,
            "initial_gamma": app_config.MODEL_INITIAL_GAMMA,
            "gamma_ramp_steps": app_config.MODEL_GAMMA_RAMP_STEPS,
            "lr": app_config.MODEL_LR,
            "decay": app_config.MODEL_DECAY,
            "clip_grad": app_config.MODEL_CLIP_GRAD,
            "optimizer": str(type(optimizer)),
        }
    )
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
        board = chess.Board()
        total_loss = 0.0
        moves = 0

        while not board.is_game_over():
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
            else:
                # Select action randomly with softmax
                action = torch.multinomial(F.softmax(masked_q_values, dim=-1), 1)

            # Take action and observe reward and next state
            move = index_to_move(action, board)
            if move is None:
                logger.warning("Invalid move selected!")
                break
            reward = calculate_reward(board, move)

            board.push(move)
            next_state = board_to_tensor(board, board.turn).to(device)
            next_state = next_state.unsqueeze(0)

            done = torch.tensor([int(board.is_game_over())], device=device)

            # Predict next Q-values
            next_q_values = model(next_state)

            # Mask illegal moves
            next_legal_moves_mask = get_legal_moves_mask(board).to(device)
            masked_next_q_values = next_q_values.masked_fill(
                next_legal_moves_mask == 0, -1e10
            )
            max_next_q_values = masked_next_q_values.max(1)[0].detach()

            # Compute the target Q-value
            target_q_values = reward + (gamma * max_next_q_values * (1 - done))
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

            # write_log(
            #     board=board,
            #     move=move,
            #     score=predicted_q.item(),
            #     episode=episode,
            #     total_loss=total_loss,
            #     loss=loss.item(),
            # )
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

        # Epsilon decay
        epsilon = max(epsilon * app_config.MODEL_DECAY, app_config.MODEL_MIN_EPSILON)

        # Learning rate scheduler
        scheduler.step()

    writer.close()
    logger.info("Training complete")
    model_output_path = (
        base_path / app_config.APP_OUTPUT_DIR / f"model_{model_timestamp}_final.pt"
    )
    torch.save(model.state_dict(), model_output_path)
    logger.info(f"Model saved to {model_output_path}")


if __name__ == "__main__":
    app_config = AppConfig()
    model = ChessTransformer(
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        freeze_pos=False,
    )
    train_deep_q_network(model, episodes=50000, app_config=app_config)
