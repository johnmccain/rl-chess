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


def sample_opening_states(
    openings_df: pd.DataFrame, batch_size: int
) -> list[chess.Board]:
    """
    Sample random opening moves from the dataset.
    """
    openings = openings_df.sample(n=batch_size)
    boards = [chess.Board(fen) for fen in openings["fen"].values]
    return boards


def train_deep_q_network(
    model: ChessTransformer,
    episodes: int,
    batch_size: int,
    app_config: AppConfig = AppConfig(),
):
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
            "batch_size": batch_size,
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
        # 25% of the time, start with random opening states
        if random.random() < 0.25:
            boards = sample_opening_states(openings_df, batch_size)
        else:
            boards = [chess.Board() for _ in range(batch_size)]

        total_loss = 0.0
        moves = 0

        active_games = batch_size
        while active_games > 0 and moves < app_config.MODEL_MAX_MOVES:
            current_states = torch.stack(
                [
                    board_to_tensor(board, board.turn)
                    for board in boards
                    if not board.is_game_over()
                ]
            ).to(device)

            # Predict Q-values
            predicted_q_values: torch.Tensor = model(current_states)

            # Mask illegal moves
            legal_moves_masks = torch.stack(
                [
                    get_legal_moves_mask(board)
                    for board in boards
                    if not board.is_game_over()
                ]
            ).to(device)
            masked_q_values = predicted_q_values.masked_fill(
                legal_moves_masks == 0, -1e10
            )

            # Epsilon-greedy action selection
            if random.random() > epsilon:
                actions = masked_q_values.max(1)[1].view(-1, 1)
            else:
                # Select actions randomly with softmax
                actions = torch.multinomial(F.softmax(masked_q_values, dim=-1), 1)

            # Take actions and observe rewards and next states
            rewards = []
            opp_rewards = []
            next_states = []
            dones = []

            active_boards = [board for board in boards if not board.is_game_over()]
            valid_actions = []
            for board, action in zip(active_boards, actions):
                move = index_to_move(action.item(), board)
                if move is None:
                    # If the move is not legal, choose a random legal move
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        move = random.choice(legal_moves)
                    else:
                        # If no legal moves, consider the game as done
                        dones.append(1)
                        rewards.append(0)
                        opp_rewards.append(0)
                        next_states.append(board_to_tensor(board, board.turn))
                        continue

                valid_actions.append(action)
                rewards.append(calculate_reward(board, move))
                opp_rewards.append(calculate_reward(board, move, flip_perspective=True))

                board.push(move)
                next_states.append(board_to_tensor(board, board.turn))
                dones.append(int(board.is_game_over()))

                if not board.is_game_over():
                    # Select opponent next move
                    opp_state = (
                        board_to_tensor(board, board.turn).unsqueeze(0).to(device)
                    )
                    with torch.no_grad():
                        opp_q_values = model(opp_state)
                    opp_legal_moves_mask = get_legal_moves_mask(board).to(device)
                    opp_masked_q_values = opp_q_values.masked_fill(
                        opp_legal_moves_mask == 0, -1e10
                    )
                    if random.random() > epsilon:
                        opp_action = opp_masked_q_values.max(1)[1].view(1, 1)
                    else:
                        opp_action = torch.multinomial(
                            F.softmax(opp_masked_q_values, dim=-1), 1
                        )
                    opp_move = index_to_move(opp_action.item(), board)
                    if opp_move is None:
                        # If opponent move is not legal, choose a random legal move
                        legal_moves = list(board.legal_moves)
                        if legal_moves:
                            opp_move = random.choice(legal_moves)
                        else:
                            # If no legal moves, consider the game as done
                            dones[-1] = 1
                            continue
                    board.push(opp_move)

            rewards = torch.tensor(rewards, device=device)
            opp_rewards = torch.tensor(opp_rewards, device=device)
            next_states = torch.stack(next_states).to(device)
            dones = torch.tensor(dones, device=device)
            valid_actions = torch.stack(valid_actions)

            # Compute next state Q-values
            with torch.no_grad():
                next_q_values = model(next_states)
                next_legal_moves_masks = torch.stack(
                    [get_legal_moves_mask(board) for board in active_boards]
                ).to(device)
                masked_next_q_values = next_q_values.masked_fill(
                    next_legal_moves_masks == 0, -1e10
                )
                max_next_q_values = masked_next_q_values.max(1)[0]

            # Compute target Q-values
            target_q_values = (
                rewards - opp_rewards + (gamma * max_next_q_values * (1 - dones))
            )
            predicted_q = predicted_q_values.gather(1, valid_actions)

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

            if moves % app_config.MODEL_GRAD_STEPS == 0:
                # Gradient accumulation
                optimizer.step()
                optimizer.zero_grad()

            active_games = sum(1 for board in boards if not board.is_game_over())

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
        base_path / app_config.APP_OUTPUT_DIR / f"model_{model_timestamp}_final.pt"
    )
    torch.save(model.state_dict(), model_output_path)
    logger.info(f"Model saved to {model_output_path}")


if __name__ == "__main__":
    app_config = AppConfig()
    model = ChessTransformer(
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        freeze_pos=True,
    )
    train_deep_q_network(model, episodes=50000, batch_size=64, app_config=app_config)
