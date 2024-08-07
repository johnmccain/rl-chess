import datetime
import logging
import random
from dataclasses import dataclass, field
import collections
import pickle

import chess
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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


@dataclass(order=True)
class ExperienceRecord:
    q_diff: float
    state: torch.Tensor= field(compare=False)
    action: torch.Tensor= field(compare=False)
    reward: float= field(compare=False)
    next_state: torch.Tensor = field(compare=False)
    done: bool= field(compare=False)


class ExperienceBuffer:

    def __init__(self, window_size: int) -> None:
        self.buffer: collections.deque[ExperienceRecord] = collections.deque(maxlen=window_size)
        self.window_size = window_size

    def add(self, experience: ExperienceRecord) -> None:
        self.buffer.append(experience)

    def sample(self) -> ExperienceRecord:
        """
        Sample a random experience from the buffer and return it.
        """
        if not self.buffer:
            raise IndexError("sample from an empty buffer")
        return random.choice(self.buffer)

    def sample_n(self, n: int) -> list[ExperienceRecord]:
        """
        Sample n random experiences from the buffer without replacement and return them.
        """
        if len(self.buffer) < n:
            raise IndexError("sample from an empty buffer")
        return random.sample(self.buffer, n)

    def extend(self, iterable: list[ExperienceRecord]) -> None:
        self.buffer.extend(iterable)

    def __len__(self) -> int:
        return len(self.buffer)


class CNNTrainer:

    def __init__(
        self,
        stockfish_evaluator: StockfishEvaluator | None = None,
        device: str | None = None,
        app_config: AppConfig = AppConfig(),
    ) -> None:

        self.stockfish_evaluator = stockfish_evaluator or StockfishEvaluator()
        self.stockfish_evaluator.set_depth(app_config.STOCKFISH_DEPTH)

        if device:
            self.device = torch.device(device)
        else:
            self.device = self.select_device()

        # Load curriculum data
        self.openings_df = pd.read_csv(base_path / "data/openings_fen7.csv")

        self.model_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.writer = SummaryWriter(
            log_dir=base_path
            / app_config.APP_TENSORBOARD_DIR
            / f"qchess_{self.model_timestamp}",
        )

        self.experience_buffer = ExperienceBuffer(app_config.MODEL_BUFFER_SIZE)


    @staticmethod
    def find_latest_model_episode(model_timestamp: str) -> int | None:
        """
        Find the latest episode number for a given model timestamp.
        """
        model_files = list(
            base_path.glob(f"{app_config.APP_OUTPUT_DIR}/model_{model_timestamp}_e*.pt")
        )
        if not model_files:
            return None
        return max([int(f.stem.split("_e")[-1]) for f in model_files])

    def save_checkpoint(self, model: ChessCNN, optimizer: optim.Optimizer, episode: int) -> None:
        torch.save(
            model.state_dict(),
            base_path
            / app_config.APP_OUTPUT_DIR
            / f"model_{self.model_timestamp}_e{episode}.pt",
        )
        # Save optimizer state
        torch.save(
            optimizer.state_dict(),
            base_path
            / app_config.APP_OUTPUT_DIR
            / f"optimizer_{self.model_timestamp}_e{episode}.pt",
        )
        # Save experience buffer
        with open(
            base_path
            / app_config.APP_OUTPUT_DIR
            / f"experience_buffer_{self.model_timestamp}_e{episode}.pkl",
            "wb",
        ) as f:
            pickle.dump(self.experience_buffer, f)

    def select_device(self) -> torch.device:
        if torch.cuda.is_available():
            logger.info("CUDA available, using GPU")
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            logger.info("MPS available, using MPS")
            device = torch.device("mps")
        else:
            logger.info("CUDA unavailable, using CPU")
            device = torch.device("cpu")
        return device

    def sample_opening_state(self) -> chess.Board:
        """
        Sample a random opening move from the curriculum dataset.
        """
        opening = self.openings_df.sample()
        board = chess.Board(opening["fen"].values[0])
        return board

    def select_action(self, masked_q_values: torch.Tensor, epsilon: float, board: chess.Board) -> torch.Tensor:
        """
        Select an action based on epsilon-greedy policy or stockfish evaluation.

        :param masked_q_values: Q-values for the current state with illegal moves masked
        :param epsilon: Exploration rate
        :param board: Current board state
        """
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            action = masked_q_values.max(1)[1].view(1, 1)
        elif random.random() < app_config.STOCKFISH_PROB:
            action = torch.tensor(
                [self.stockfish_evaluator.take_action(board)], device=self.device
            ).unsqueeze(0)
        else:
            # Select action randomly with softmax
            action = torch.multinomial(F.softmax(masked_q_values, dim=-1), 1)
        return action

    def explore(self, model: ChessCNN, episodes: int, gamma: float, epsilon: float) -> list[ExperienceRecord]:
        experience_buffer = []
        for episode in tqdm(range(episodes), total=episodes, desc="Exploring"):
            # 25% of the time, start with a random opening state
            if random.random() < 0.25:
                board = self.sample_opening_state()
            else:
                board = chess.Board()
            moves = 0

            while not board.is_game_over() and moves < app_config.MODEL_MAX_MOVES:
                current_state = board_to_tensor(board, board.turn).to(self.device)
                current_state = current_state.unsqueeze(0)  # Batch size of 1

                # Predict Q-values
                with torch.no_grad():
                    predicted_q_values: torch.Tensor = model(current_state)

                # Mask illegal moves
                legal_moves_mask = get_legal_moves_mask(board).to(self.device)
                masked_q_values = predicted_q_values.masked_fill(
                    legal_moves_mask == 0, -1e10
                )

                action = self.select_action(masked_q_values, epsilon, board)

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
                opp_next_state = board_to_tensor(board, board.turn).to(self.device)
                opp_next_state = opp_next_state.unsqueeze(0)

                done = torch.tensor([int(board.is_game_over())], device=self.device)

                if not done:
                    # Select opponent next move
                    with torch.no_grad():
                        opp_next_q_values = model(opp_next_state)
                    opp_next_legal_moves_mask = get_legal_moves_mask(board).to(self.device)
                    opp_masked_next_q_values = opp_next_q_values.masked_fill(
                        opp_next_legal_moves_mask == 0, -1e10
                    )
                    opp_action = self.select_action(opp_masked_next_q_values, epsilon, board)

                    # Take opponent action
                    opp_move = index_to_move(opp_action, board)
                    if opp_move is None:
                        logger.warning("Invalid opponent move selected!")
                        break
                    # Calculate reward for opponent move
                    opp_reward = calculate_reward(board, opp_move)
                    board.push(opp_move)

                    # Compute the next-state max Q-value for active player given the opponent's possible move
                    next_state = board_to_tensor(board, board.turn).to(self.device)
                    next_state = next_state.unsqueeze(0)
                    with torch.no_grad():
                        next_q_values = model(next_state)
                    next_legal_moves_mask = get_legal_moves_mask(board).to(self.device)
                    masked_next_q_values = next_q_values.masked_fill(
                        next_legal_moves_mask == 0, -1e10
                    )
                    max_next_q_values = masked_next_q_values.max(1)[0].detach()
                    # Roll back the board state to before the opponent move
                    board.pop()
                else:
                    # NOTE: is 0.0 the correct default value for max_next_q_values?
                    max_next_q_values = torch.tensor([0.0], device=self.device)
                    next_state = current_state  # This is a terminal state; ends up being ignored but we need some value of the correct shape
                # Compute the target Q-value
                target_q_value = (
                    reward - opp_reward + (gamma * max_next_q_values * (1 - done))
                )
                predicted_q = predicted_q_values.gather(1, action)

                experience_buffer.append(
                    ExperienceRecord(
                        q_diff=(predicted_q - target_q_value).item(),
                        state=current_state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=bool(done),
                    )
                )
                moves += 1
        return experience_buffer

    def learn(
        self,
        model: ChessCNN,
        optimizer: optim.Optimizer,
        loss_fn: torch.nn.Module,
        gamma: float,
        steps: int,
        batch_size: int,
        step_offset: int = 0,
    ) -> float:
        total_loss = 0.0
        for step in tqdm(range(steps), total=steps, desc="Learning"):
            batch = self.experience_buffer.sample_n(batch_size)
            state_batch = torch.cat([exp.state for exp in batch])
            action_batch = torch.cat([exp.action for exp in batch])
            reward_batch = torch.tensor([exp.reward for exp in batch], device=self.device)
            next_state_batch = torch.cat([exp.next_state for exp in batch if exp.next_state is not None])
            done_batch = torch.tensor([int(exp.done) for exp in batch], device=self.device)

            predicted_q_values = model(state_batch)
            predicted_q = predicted_q_values.gather(1, action_batch)

            max_next_q_values = model(next_state_batch).max(1)[0].detach()

            target_q_values = reward_batch + (gamma * max_next_q_values * (1 - done_batch))

            # Compute loss
            loss = loss_fn(predicted_q, target_q_values.unsqueeze(1))
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), app_config.MODEL_CLIP_GRAD
            )
            optimizer.step()
            optimizer.zero_grad()

            # Log metrics to Tensorboard
            self.writer.add_scalar("Loss/Step", loss.item(), step + step_offset)
        return total_loss


    def train_deep_q_network_off_policy(
        self,
        model: ChessCNN,
        episodes: int,
        app_config: AppConfig = AppConfig(),
    ) -> ChessCNN:
        model.to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=app_config.MODEL_LR)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=episodes, eta_min=1e-6
        )

        loss_fn = F.mse_loss
        epsilon = 1.0

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
        self.writer.add_hparams(
            hparam_dict=hparams,
            metric_dict={},
        )

        gamma_ramp = (
            app_config.MODEL_GAMMA - app_config.MODEL_INITIAL_GAMMA
        ) / app_config.MODEL_GAMMA_RAMP_STEPS

        episode = 0
        step = 0

        while episode < episodes:
            gamma = min(
                app_config.MODEL_INITIAL_GAMMA + gamma_ramp * episode,
                app_config.MODEL_GAMMA,
            )

            new_experiences = self.explore(model, app_config.MODEL_EXPLORE_EPISODES, gamma, epsilon)
            self.experience_buffer.extend(new_experiences)
            episodes += app_config.MODEL_EXPLORE_EPISODES

            total_loss = self.learn(
                model, optimizer, loss_fn, gamma, app_config.MODEL_LEARN_STEPS, app_config.MODEL_GRAD_STEPS
            )

            # Log metrics to Tensorboard
            self.writer.add_scalar("Loss/Step", total_loss, step)
            self.writer.add_scalar("Epsilon/Step", epsilon, step)
            self.writer.add_scalar("Gamma/Step", gamma, step)
            self.writer.add_scalar("LR/Step", scheduler.get_last_lr()[0], episode)

            logger.info(
                f"Episode {episode}, Loss: {total_loss}"
            )

            if episode % app_config.APP_SAVE_STEPS == 0:
                self.save_checkpoint(model, optimizer, episode)

            # Epsilon decay
            epsilon = max((1.0 * app_config.MODEL_DECAY)**episode, app_config.MODEL_MIN_EPSILON)

            # Learning rate scheduler
            scheduler.step()

        self.writer.close()
        logger.info("Training complete")
        model_output_path = (
            base_path / app_config.APP_OUTPUT_DIR / f"model_cnn_{self.model_timestamp}_final.pt"
        )
        torch.save(model.state_dict(), model_output_path)
        logger.info(f"Model saved to {model_output_path}")
        return model


if __name__ == "__main__":
    app_config = AppConfig()
    model = ChessCNN(num_filters=256, num_residual_blocks=12)
    trainer = CNNTrainer(app_config=app_config)
    trainer.train_deep_q_network_off_policy(model, episodes=50000, app_config=app_config)
