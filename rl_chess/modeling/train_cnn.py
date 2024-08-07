import os
import pathlib
import collections
import datetime
import logging
import pickle
import random
from dataclasses import dataclass, field

import chess
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score

from rl_chess import base_path
from rl_chess.config.config import AppConfig
from rl_chess.evaluation.stockfish_evaluator import StockfishEvaluator
from rl_chess.modeling.chess_cnn import ChessCNN  # Import the new ChessCNN model
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
    handlers=[logging.StreamHandler(), logging.FileHandler("train_cnn.log")],
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
    state: torch.Tensor = field(compare=False)
    legal_moves_mask: torch.Tensor = field(compare=False)
    action: torch.Tensor = field(compare=False)
    reward: float = field(compare=False)
    next_state: torch.Tensor = field(compare=False)
    next_legal_moves_mask: torch.Tensor = field(compare=False)
    done: bool = field(compare=False)
    pred_q_values: torch.Tensor | None = field(default=None, compare=False)
    max_next_q: float | None = field(default=None, compare=False)

    def make_serializeable(self) -> dict:
        """
        Make efficiently serializable by converting tensors to numpy arrays and converting to dictionary.
        """
        return {
            "q_diff": self.q_diff,
            "state": self.state.cpu().numpy(),
            "legal_moves_mask": self.legal_moves_mask.cpu().numpy(),
            "action": self.action.cpu().numpy(),
            "reward": self.reward,
            "next_state": self.next_state.cpu().numpy(),
            "next_legal_moves_mask": self.next_legal_moves_mask.cpu().numpy(),
            "done": self.done,
            "pred_q_values": self.pred_q_values.cpu().numpy() if self.pred_q_values is not None else None,
            "max_next_q": self.max_next_q,
        }

    @classmethod
    def from_serialized(cls, serialized: dict) -> "ExperienceRecord":
        """
        Load a serialized ExperienceRecord from a dictionary by converting numpy arrays back to tensors.
        """
        return cls(
            q_diff=serialized["q_diff"],
            state=torch.tensor(serialized["state"]),
            legal_moves_mask=torch.tensor(serialized["legal_moves_mask"]),
            action=torch.tensor(serialized["action"]),
            reward=serialized["reward"],
            next_state=torch.tensor(serialized["next_state"]),
            next_legal_moves_mask=torch.tensor(serialized["next_legal_moves_mask"]),
            done=serialized["done"],
            pred_q_values=torch.tensor(serialized["pred_q_values"]) if serialized["pred_q_values"] is not None else None,
            max_next_q=serialized["max_next_q"],
        )


class ExperienceBuffer:
    def __init__(self, window_size: int) -> None:
        self.buffer: collections.deque[ExperienceRecord] = collections.deque(
            maxlen=window_size
        )
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

    def save_experience_buffer(self, filename: str | pathlib.Path) -> None:
        saveable_experience_buffer = [
            exp.make_serializeable()
            for exp in self.experience_buffer.buffer
        ]
        with open(filename, "wb") as f:
            pickle.dump(saveable_experience_buffer, f)

    def save_checkpoint(
        self, model: ChessCNN, optimizer: optim.Optimizer, episode: int
    ) -> None:
        logger.info(f"Saving checkpoint for episode {episode}")
        
        checkpoint_dir = base_path / app_config.APP_OUTPUT_DIR
        
        # Save current checkpoint
        torch.save(
            model.state_dict(),
            checkpoint_dir / f"model_{self.model_timestamp}_e{episode}.pt",
        )
        torch.save(
            optimizer.state_dict(),
            checkpoint_dir / f"optimizer_{self.model_timestamp}_e{episode}.pt",
        )
        self.save_experience_buffer(
            checkpoint_dir / f"experience_buffer_{self.model_timestamp}_e{episode}.pkl"
        )

        # Get all checkpoint files for this model
        model_files = list(checkpoint_dir.glob(f"model_{self.model_timestamp}_e*.pt"))
        optimizer_files = list(checkpoint_dir.glob(f"optimizer_{self.model_timestamp}_e*.pt"))
        buffer_files = list(checkpoint_dir.glob(f"experience_buffer_{self.model_timestamp}_e*.pkl"))
        
        # Sort files by episode number
        def get_episode(filename):
            return int(filename.stem.split('_e')[-1])
        
        model_files.sort(key=get_episode, reverse=True)
        optimizer_files.sort(key=get_episode, reverse=True)
        buffer_files.sort(key=get_episode, reverse=True)
        
        # Keep only the last 5 checkpoints
        for file_list in [model_files, optimizer_files, buffer_files]:
            for file in file_list[5:]:
                os.remove(file)
                logger.info(f"Deleted old checkpoint file: {file}")

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

    def select_action(
        self, masked_q_values: torch.Tensor, epsilon: float, board: chess.Board
    ) -> torch.Tensor:
        """
        Select an action based on epsilon-greedy policy or stockfish evaluation.

        :param masked_q_values: Q-values for the current state with illegal moves masked
        :param epsilon: Exploration rate
        :param board: Current board state
        """
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            action = masked_q_values.max(-1)[1].view(1)
        else:
            # Select action randomly with softmax
            action = torch.multinomial(F.softmax(masked_q_values, dim=-1), 1)
        return action

    def explore(
        self, model: ChessCNN, episodes: int, gamma: float, epsilon: float, batch_size: int
    ) -> list[ExperienceRecord]:
        experience_buffer = []
        for episode in tqdm(range(episodes), total=episodes, desc="Exploring"):

            boards = [
                chess.Board() if random.random() > 0.25 else self.sample_opening_state()
                for _ in range(batch_size)
            ]
            n_moves = 0

            while boards and not all(board.is_game_over() for board in boards) and n_moves < app_config.MODEL_MAX_MOVES:
                current_states = torch.stack([board_to_tensor(board, board.turn) for board in boards], dim=0).to(self.device)

                ### Calculate Player move ###

                logger.debug("START Predicting Q values")
                # Predict Q-values
                with torch.no_grad():
                    predicted_q_values, _ = model(current_states)
                logger.debug("END Predicting Q values")

                logger.debug("START Masking illegal moves")
                # Mask illegal moves
                legal_moves_mask = torch.stack([get_legal_moves_mask(board) for board in boards], dim=0).to(self.device)
                masked_q_values = predicted_q_values.masked_fill(
                    legal_moves_mask == 0, -1e10
                )
                logger.debug("END Masking illegal moves")

                logger.debug("START Selecting action")
                actions = torch.stack([
                    self.select_action(masked_q_values[idx], epsilon, board)
                    for idx, board in enumerate(boards)
                ], dim=0)
                logger.debug("END Selecting action")

                logger.debug("START Taking action & calculating reward")
                # Take action and observe reward and next state
                moves = [index_to_move(actions[idx].item(), board) for idx, board in enumerate(boards)]
                if any(move is None for move in moves):
                    logger.warning("Invalid move selected!")
                    # No handling for invalid moves; just break out of the loop
                    break
                rewards = torch.tensor([calculate_reward(board, move) for board, move in zip(boards, moves)]).to(self.device)
                # Calculate default reward for opponent move (if opponent can't make a move such as in checkmate or stalemate)
                default_opp_rewards = [calculate_reward(board, move, flip_perspective=True) for board, move in zip(boards, moves)]

                # Take the action
                for move, board in zip(moves, boards):
                    board.push(move)
                logger.debug("END Taking action & calculating reward")

                ### Calculate Opponent Move ###
                logger.debug("START Preparing Opponent move")
                opp_next_states = torch.stack([
                    board_to_tensor(board, board.turn)
                    for board in boards
                ], dim=0).to(self.device)

                # To process as a batch, we continue processing games that are over with dummy values, but need to ensure that we don't push moves to the board
                dones = torch.tensor([int(board.is_game_over()) for board in boards], device=self.device)
                logger.debug("END Preparing Opponent move")

                logger.debug("START Predicting Opponent Q values")
                # Select opponent next move
                with torch.no_grad():
                    opp_next_q_values, _ = model(opp_next_states)
                logger.debug("END Predicting Opponent Q values")

                logger.debug("START Masking illegal moves for Opponent")
                opp_next_legal_moves_mask = torch.stack([
                    get_legal_moves_mask(board)
                    for board in boards
                ], dim=0).to(
                    self.device
                )
                opp_masked_next_q_values = opp_next_q_values.masked_fill(
                    opp_next_legal_moves_mask == 0, -1e10
                )
                logger.debug("END Masking illegal moves for Opponent")

                logger.debug("START Selecting Opponent action")
                opp_actions = [
                    self.select_action(
                        opp_masked_next_q_values[idx], epsilon, board
                    ) if not done else -1  # -1 is a placeholder when there is no valid move
                    for idx, (board, done) in enumerate(zip(boards, dones))
                ]
                logger.debug("END Selecting Opponent action")

                logger.debug("START Taking Opponent action & calculating reward")
                # Take opponent action
                opp_moves = [
                    index_to_move(opp_action, board)
                    if opp_action != -1
                    else None
                    for opp_action, board in zip(opp_actions, boards)
                ]

                # Calculate reward for opponent move
                opp_rewards = torch.tensor([
                    calculate_reward(board, opp_move)
                    if opp_move
                    else default
                    for opp_move, default, board in zip(opp_moves, default_opp_rewards, boards)
                ]).to(self.device)
                for board, opp_move in zip(boards, opp_moves):
                    if opp_move:
                        board.push(opp_move)
                logger.debug("END Taking Opponent action & calculating reward")

                ### Calculate next state and reward ###
                # Compute the next-state max Q-value for active player given the opponent's possible move
                # next_states for done boards are ignored, but we need to provide a tensor of the correct shape so we use the current state (since no move was pushed)
                logger.debug("START Preparing next state")
                next_states = torch.stack([
                    board_to_tensor(board, board.turn)
                    for board in boards
                ], dim=0).to(self.device)
                logger.debug("END Preparing next state")

                logger.debug("START Predicting next Q values")
                with torch.no_grad():
                    next_q_values, _ = model(next_states)
                logger.debug("END Predicting next Q values")

                logger.debug("START Masking illegal moves for next state")
                next_legal_moves_mask = torch.stack([
                    get_legal_moves_mask(board)
                    for board in boards
                ], dim=0).to(self.device)

                masked_next_q_values = next_q_values.masked_fill(
                    next_legal_moves_mask == 0, -1e10
                )
                max_next_q_values = masked_next_q_values.max(1)[0]
                max_next_q_values[max_next_q_values == -1e10] = 0.0  # Set Q-value to 0.0 for situations where no legal moves are available
                logger.debug("END Masking illegal moves for next state")

                logger.debug("START Calculating target Q value")
                # Roll back the board state to before the opponent move (if any opponent move was made)
                for board, opp_move in zip(boards, opp_moves):
                    if opp_move:
                        board.pop()

                # Handle boards where opponent had no valid moves (game was over)
                # NOTE: is 0.0 the correct default value for max_next_q_values?
                max_next_q_values.masked_fill(
                    dones == 1, 0.0
                )
                # Compute the target Q-value
                target_q_value = (
                    rewards - opp_rewards + (gamma * max_next_q_values * (1 - dones))
                )
                predicted_q = predicted_q_values.gather(1, actions)
                logger.debug("END Calculating target Q value")

                logger.debug("START Creating ExperienceRecord")
                # Create separate ExperienceRecord for each board
                for current_state, legal_mask, action, reward, next_state, next_legal_mask, done, predicted_q, target_q_value, pred_q, max_next_q in zip(
                    current_states, legal_moves_mask, actions, rewards, next_states, next_legal_moves_mask, dones, predicted_q, target_q_value, predicted_q_values, max_next_q_values
                ):
                    experience_buffer.append(
                        ExperienceRecord(
                            q_diff=(predicted_q - target_q_value).item(),
                            state=current_state,
                            legal_moves_mask=legal_mask,
                            action=action,
                            reward=reward,
                            next_state=next_state,
                            next_legal_moves_mask=next_legal_mask,
                            done=bool(done),
                            pred_q_values=pred_q,
                            max_next_q=max_next_q.item()
                        )
                    )
                logger.debug("END Creating ExperienceRecord")
                # Remove any games that are done
                boards = [board for board in boards if not board.is_game_over()]
                n_moves += 1
        return experience_buffer

    def learn(
        self,
        model: ChessCNN,
        optimizer: optim.Optimizer,
        q_loss_fn: torch.nn.Module,
        aux_loss_fn: torch.nn.Module,
        gamma: float,
        steps: int,
        batch_size: int,
        step_offset: int = 0,
    ) -> tuple[float, float]:
        total_q_loss = 0.0
        total_aux_loss = 0.0
        for step in tqdm(range(steps), total=steps, desc="Learning"):
            batch = self.experience_buffer.sample_n(batch_size)
            state_batch = torch.stack(
                [exp.state for exp in batch],
                dim=0
            )
            legal_moves_mask_batch = torch.stack(
                [exp.legal_moves_mask for exp in batch],
                dim=0
            )
            action_batch = torch.stack(
                [exp.action for exp in batch],
                dim=0
            )
            reward_batch = torch.tensor(
                [exp.reward for exp in batch], device=self.device
            )
            next_state_batch = torch.stack(
                [exp.next_state for exp in batch],
                dim=0
            )
            next_legal_moves_mask_batch = torch.stack(
                [exp.next_legal_moves_mask for exp in batch],
                dim=0
            )
            done_batch = torch.tensor(
                [int(exp.done) for exp in batch], device=self.device
            )

            predicted_q_values, aux_logits = model(state_batch)
            predicted_q = predicted_q_values.gather(1, action_batch)

            next_q_values, _ = model(next_state_batch)
            masked_next_q_values = next_q_values.masked_fill(
                next_legal_moves_mask_batch == 0, -1e10
            )
            max_next_q_values = masked_next_q_values.max(1)[0].detach()

            target_q_values = reward_batch + (
                gamma * max_next_q_values * (1 - done_batch)
            )

            # Compute loss
            q_loss = q_loss_fn(predicted_q, target_q_values.unsqueeze(1))
            total_q_loss += q_loss.item()
            aux_loss = aux_loss_fn(
                aux_logits,
                legal_moves_mask_batch
            )
            total_aux_loss += aux_loss.item()
            loss = q_loss + aux_loss

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), app_config.MODEL_CLIP_GRAD
            )
            optimizer.step()
            optimizer.zero_grad()

            # Log metrics to Tensorboard
            self.writer.add_scalar("Loss/Step", loss.item(), step + step_offset)
            self.writer.add_scalar("QLoss/Step", q_loss.item(), step + step_offset)
            self.writer.add_scalar("AuxLoss/Step", aux_loss.item(), step + step_offset)
        return total_q_loss, total_aux_loss

    def eval_model(self, model: ChessCNN, steps: int, batch_size: int, step: int) -> float:
        """
        Evaluate the model. Currently only evaluates based on auxiliary task accuracy (move legality).
        """
        correct = 0
        total = 0
        aux_val_loss = 0.0
        predictions = []
        targets = []
        for _ in tqdm(range(steps), total=steps, desc="Evaluating"):
            batch = self.experience_buffer.sample_n(batch_size)
            state_batch = torch.stack(
                [exp.state for exp in batch],
                dim=0
            )
            legal_moves_mask_batch = torch.stack(
                [exp.legal_moves_mask for exp in batch],
                dim=0
            )
            with torch.no_grad():
                _, aux_logits = model(state_batch)
            predicted_labels = torch.round(torch.sigmoid(aux_logits))
            correct += (predicted_labels == legal_moves_mask_batch).sum().item()
            total += legal_moves_mask_batch.numel()
            aux_val_loss += F.binary_cross_entropy_with_logits(aux_logits, legal_moves_mask_batch).item()
            predictions.extend(predicted_labels.cpu().numpy().flatten())
            targets.extend(legal_moves_mask_batch.cpu().numpy().flatten())
        accuracy = correct / total
        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)
        self.writer.add_scalar("Validation/AuxAccuracy/Step", accuracy, step)
        self.writer.add_scalar("Validation/AuxPrecision/Step", precision, step)
        self.writer.add_scalar("Validation/AuxRecall/Step", recall, step)
        self.writer.add_scalar("Validation/AuxLoss/Step", aux_val_loss, step)
        return aux_val_loss

    def train_deep_q_network_off_policy(
        self,
        model: ChessCNN,
        episodes: int,
        app_config: AppConfig = AppConfig(),
    ) -> ChessCNN:
        model.to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=app_config.MODEL_LR)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=episodes//(app_config.MODEL_EXPLORE_EPISODES * app_config.MODEL_BATCH_SIZE),
            eta_min=1e-6
        )

        q_loss_fn = F.mse_loss
        aux_loss_fn = F.binary_cross_entropy_with_logits  # Classifying move legality
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

        episode = 0  # episode = 1 game played
        step = 0  # step = 1 batch of moves learned
        last_saved_episode = 0
        last_eval_episode = 0

        while episode < episodes:
            gamma = min(
                app_config.MODEL_INITIAL_GAMMA + gamma_ramp * episode,
                app_config.MODEL_GAMMA,
            )

            new_experiences = self.explore(
                model, app_config.MODEL_EXPLORE_EPISODES, gamma, epsilon, app_config.MODEL_BATCH_SIZE
            )
            self.experience_buffer.extend(new_experiences)
            episode += app_config.MODEL_EXPLORE_EPISODES * app_config.MODEL_BATCH_SIZE

            total_q_loss, total_aux_loss = self.learn(
                model=model,
                optimizer=optimizer,
                q_loss_fn=q_loss_fn,
                aux_loss_fn=aux_loss_fn,
                gamma=gamma,
                steps=app_config.MODEL_LEARN_STEPS,
                batch_size=app_config.MODEL_BATCH_SIZE,
                step_offset=step,
            )
            step += app_config.MODEL_LEARN_STEPS

            # Log metrics to Tensorboard
            self.writer.add_scalar("TotalQLoss/Step", total_q_loss, step)
            self.writer.add_scalar("TotalAuxLoss/Step", total_aux_loss, step)
            self.writer.add_scalar("Epsilon/Step", epsilon, step)
            self.writer.add_scalar("Gamma/Step", gamma, step)
            self.writer.add_scalar("LR/Step", scheduler.get_last_lr()[0], step)

            logger.info(f"Episode {episode}, QLoss: {total_q_loss}, AuxLoss: {total_aux_loss}")

            if episode - last_saved_episode > app_config.APP_SAVE_STEPS:
                self.save_checkpoint(model, optimizer, episode)
                last_saved_episode = episode

            if episode - last_eval_episode > app_config.APP_EVAL_STEPS:
                aux_val_loss = self.eval_model(model, 10, app_config.MODEL_BATCH_SIZE, step)
                logger.info(f"Episode {episode}, Validation Aux Loss: {aux_val_loss}")
                last_eval_episode = episode

            # Epsilon decay
            epsilon = max(
                (app_config.MODEL_DECAY) ** (episode // (app_config.MODEL_EXPLORE_EPISODES * app_config.MODEL_BATCH_SIZE)),
                app_config.MODEL_MIN_EPSILON
            )

            # Learning rate scheduler
            scheduler.step()

        self.writer.close()
        self.save_checkpoint(model, optimizer, episode)
        logger.info("Training complete")
        return model


if __name__ == "__main__":
    app_config = AppConfig()
    model = ChessCNN(num_filters=256, num_residual_blocks=12)
    trainer = CNNTrainer(app_config=app_config)
    trainer.train_deep_q_network_off_policy(
        model, episodes=100000, app_config=app_config
    )
