import os
import pathlib
import datetime
import logging
import pickle
import random
import argparse
import json

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
from rl_chess.modeling.chess_cnn import ChessCNN  # Import the new ChessCNN model
from rl_chess.modeling.utils import (
    board_to_tensor,
    calculate_reward,
    get_legal_moves_mask,
    index_to_move,
)
from rl_chess.modeling.experience_buffer import ExperienceBuffer, ExperienceRecord

app_config = AppConfig()

logging.basicConfig(
    level=app_config.LOG_LEVEL,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train_cnn.log")],
)

logger = logging.getLogger(__name__)

class CNNTrainer:
    def __init__(
        self,
        device: str | None = None,
        app_config: AppConfig = AppConfig(),
        model_timestamp: str | None = None,
    ) -> None:

        if device:
            self.device = torch.device(device)
        else:
            self.device = self.select_device()

        # Load curriculum data
        self.openings_df = pd.read_csv(base_path / "data/openings_fen7.csv")

        self.model_timestamp = model_timestamp or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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

    def save_hparams(self, model: ChessCNN, app_config: AppConfig) -> None:
        hparams = {
            "MODEL_NUM_FILTERS": app_config.MODEL_NUM_FILTERS,
            "MODEL_RESIDUAL_BLOCKS": app_config.MODEL_RESIDUAL_BLOCKS,
            "MODEL_GAMMA": app_config.MODEL_GAMMA,
            "MODEL_INITIAL_GAMMA": app_config.MODEL_INITIAL_GAMMA,
            "MODEL_GAMMA_RAMP_STEPS": app_config.MODEL_GAMMA_RAMP_STEPS,
            "MODEL_LR": app_config.MODEL_LR,
            "MODEL_DECAY": app_config.MODEL_DECAY,
            "MODEL_CLIP_GRAD": app_config.MODEL_CLIP_GRAD,
            "MODEL_BUFFER_SIZE": app_config.MODEL_BUFFER_SIZE,
            "MODEL_MIN_EPSILON": app_config.MODEL_MIN_EPSILON,
            "MODEL_EXPLORE_EPISODES": app_config.MODEL_EXPLORE_EPISODES,
            "MODEL_LEARN_STEPS": app_config.MODEL_LEARN_STEPS,
            "MODEL_TARGET_UPDATE_FREQ": app_config.MODEL_TARGET_UPDATE_FREQ,
        }
        # Save hparams to Tensorboard
        self.writer.add_hparams(
            hparam_dict=hparams,
            metric_dict={},
        )
        with open(base_path / app_config.APP_OUTPUT_DIR / f"hparams_{self.model_timestamp}.json", "w") as f:
            json.dump(hparams, f)

    @staticmethod
    def select_device() -> torch.device:
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
        Select an action based on epsilon-greedy policy.

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
                # Check if the game is over after the opponent move
                opp_dones = torch.tensor([int(board.is_game_over()) for board in boards], device=self.device)
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
                for current_state, legal_mask, action, reward, next_state, next_legal_mask, done, opp_done, predicted_q, target_q_value, pred_q, max_next_q in zip(
                    current_states, legal_moves_mask, actions, rewards, next_states, next_legal_moves_mask, dones, opp_dones, predicted_q, target_q_value, predicted_q_values, max_next_q_values
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
                            opp_done=bool(opp_done),
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
        target_model: ChessCNN,
        optimizer: optim.Optimizer,
        q_loss_fn: torch.nn.Module,
        aux_loss_fn: torch.nn.Module,
        gamma: float,
        steps: int,
        batch_size: int,
        target_update_freq: int,
        step_offset: int = 0,
    ) -> tuple[float, float]:
        total_q_loss = 0.0
        total_aux_loss = 0.0
        for step in tqdm(range(steps), total=steps, desc="Learning"):
            batch = self.experience_buffer.sample_n(batch_size)
            state_batch = torch.stack([exp.state for exp in batch], dim=0)
            legal_moves_mask_batch = torch.stack([exp.legal_moves_mask for exp in batch], dim=0)
            action_batch = torch.stack([exp.action for exp in batch], dim=0)
            reward_batch = torch.tensor([exp.reward for exp in batch], device=self.device)
            next_state_batch = torch.stack([exp.next_state for exp in batch], dim=0)
            next_legal_moves_mask_batch = torch.stack([exp.next_legal_moves_mask for exp in batch], dim=0)
            done_batch = torch.tensor([int(exp.done) for exp in batch], device=self.device)
            opp_done_batch = torch.tensor([int(exp.opp_done) for exp in batch], device=self.device)

            predicted_q_values, aux_logits = model(state_batch)
            predicted_q = predicted_q_values.gather(1, action_batch)

            # Use target network for next Q-values
            with torch.no_grad():
                next_q_values, _ = target_model(next_state_batch)
                masked_next_q_values = next_q_values.masked_fill(next_legal_moves_mask_batch == 0, -1e10)
                max_next_q_values = masked_next_q_values.max(1)[0]
                masked_max_next_q_values = max_next_q_values * (1 - done_batch) * (1 - opp_done_batch)
                discounted_max_next_q_values = gamma * masked_max_next_q_values
                target_q_values = reward_batch + discounted_max_next_q_values

            # Compute loss
            q_loss = q_loss_fn(predicted_q, target_q_values.unsqueeze(1))
            total_q_loss += q_loss.item()
            aux_loss = aux_loss_fn(aux_logits, legal_moves_mask_batch)
            total_aux_loss += aux_loss.item()
            loss = q_loss + aux_loss

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), app_config.MODEL_CLIP_GRAD)
            optimizer.step()
            optimizer.zero_grad()

            self.writer.add_scalar("Loss/Step", loss.item(), step + step_offset)
            self.writer.add_scalar("QLoss/Step", q_loss.item(), step + step_offset)
            self.writer.add_scalar("AuxLoss/Step", aux_loss.item(), step + step_offset)

            # Update target network
            if (step + 1) % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

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

    def fill_experience_buffer(self, model: ChessCNN, epsilon: float, gamma: float, app_config: AppConfig):
        logger.info("Filling experience buffer")
        # Fill the experience buffer
        # We don't count this towards the total number of episodes
        desired_additional_experience = app_config.MODEL_BUFFER_SIZE - len(self.experience_buffer.buffer)
        estimated_experience_rate = app_config.MODEL_BATCH_SIZE * app_config.MODEL_MAX_MOVES # Estimated experience per episode batch, start with max moves per episode

        while len(self.experience_buffer.buffer) < app_config.MODEL_BUFFER_SIZE:
            desired_additional_experience = app_config.MODEL_BUFFER_SIZE - len(self.experience_buffer.buffer)
            est_num_episodes = max(int(desired_additional_experience // estimated_experience_rate), 1)
            new_experiences = self.explore(
                model, est_num_episodes, gamma, epsilon, app_config.MODEL_BATCH_SIZE
            )
            self.experience_buffer.extend(new_experiences)
            # Update estimated experience rate
            estimated_experience_rate = len(new_experiences) / est_num_episodes
            logger.info(f"Experience buffer size: {len(self.experience_buffer.buffer)}/{app_config.MODEL_BUFFER_SIZE}")

    def train_deep_q_network_off_policy(
        self,
        model: ChessCNN,
        optimizer: optim.Optimizer,
        episodes: int,
        app_config: AppConfig = AppConfig(),
        start_episode: int = 0,
        start_step: int = 0,
    ) -> ChessCNN:
        model.to(self.device)
        target_model = ChessCNN(num_filters=model.num_filters, num_residual_blocks=model.num_residual_blocks).to(self.device)
        target_model.load_state_dict(model.state_dict())
        target_model.eval()

        t_max = episodes//(app_config.MODEL_EXPLORE_EPISODES * app_config.MODEL_BATCH_SIZE)  # Number of epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=1e-6
        )

        if start_episode > 0:
            # Convert start_episode to the corresponding epoch
            start_epoch = start_episode // (app_config.MODEL_EXPLORE_EPISODES * app_config.MODEL_BATCH_SIZE)
            for _ in range(start_epoch):
                scheduler.step()

        q_loss_fn = torch.nn.SmoothL1Loss()
        aux_loss_fn = F.binary_cross_entropy_with_logits  # Classifying move legality

        self.save_hparams(model, app_config)

        episode = start_episode  # episode = 1 game played
        step = start_step  # step = 1 batch of moves learned
        last_saved_episode = 0
        last_eval_episode = 0

        gamma_ramp = (
            app_config.MODEL_GAMMA - app_config.MODEL_INITIAL_GAMMA
        ) / app_config.MODEL_GAMMA_RAMP_STEPS
        gamma = min(app_config.MODEL_INITIAL_GAMMA, app_config.MODEL_GAMMA + gamma_ramp * max(episode - app_config.MODEL_GAMMA_STARTUP_STEPS, 0))
        epsilon = max(
            (app_config.MODEL_DECAY) ** (episode // (app_config.MODEL_EXPLORE_EPISODES * app_config.MODEL_BATCH_SIZE)),
            app_config.MODEL_MIN_EPSILON
        )
        self.fill_experience_buffer(model, epsilon, gamma, app_config)

        while episode < episodes:
            gamma = min(
                app_config.MODEL_INITIAL_GAMMA + gamma_ramp * max(episode - app_config.MODEL_GAMMA_STARTUP_STEPS, 0),
                app_config.MODEL_GAMMA,
            )

            new_experiences = self.explore(
                model, app_config.MODEL_EXPLORE_EPISODES, gamma, epsilon, app_config.MODEL_BATCH_SIZE
            )
            self.experience_buffer.extend(new_experiences)
            episode += app_config.MODEL_EXPLORE_EPISODES * app_config.MODEL_BATCH_SIZE

            total_q_loss, total_aux_loss = self.learn(
                model=model,
                target_model=target_model,
                optimizer=optimizer,
                q_loss_fn=q_loss_fn,
                aux_loss_fn=aux_loss_fn,
                gamma=gamma,
                steps=app_config.MODEL_LEARN_STEPS,
                batch_size=app_config.MODEL_BATCH_SIZE,
                target_update_freq=app_config.MODEL_TARGET_UPDATE_FREQ,
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


def find_latest_model_filename(model_timestamp: str | None = None) -> str:
    """
    Find the latest filename for a given model timestamp.
    :param model_timestamp: The timestamp of the model to find the latest file for. If None, the latest model is used.
    :return: The latest filename for the given model timestamp, or None if no matching model files are found
    """

    if model_timestamp is None:
        model_files = list(base_path.glob(f"{app_config.APP_OUTPUT_DIR}/model_*.pt"))
    else:
        model_files = list(
            base_path.glob(f"{app_config.APP_OUTPUT_DIR}/model_{model_timestamp}_e*.pt")
        )
    
    # Sort by episode number
    sorted_models = sorted(model_files, key=lambda x: int(x.stem.split("_e")[-1]))

    if model_timestamp is None:
        # Sort by timestamp second
        # In-place sort means last element is the highest episode number of the newest model
        sorted_models = sorted(sorted_models, key=lambda x: datetime.datetime.strptime(x.stem.split("_")[1], "%Y%m%d-%H%M%S"))
    if not sorted_models:
        return None
    return str(sorted_models[-1])


def load_from_checkpoint(
    model_filename: str, device: torch.device,
) -> tuple[ChessCNN, optim.Optimizer, int, dict, str]:
    # First find the model timestamp from the filename
    model_timestamp = model_filename.split("_e")[0].split("_")[-1]
    # Load hparams from the hparams file
    with open(base_path / app_config.APP_OUTPUT_DIR / f"hparams_{model_timestamp}.json", "rb") as f:
        hparams = json.load(f)

    # Load the model and optimizer
    model = ChessCNN(num_filters=hparams["MODEL_NUM_FILTERS"], num_residual_blocks=hparams["MODEL_RESIDUAL_BLOCKS"]).to(device)
    model.load_state_dict(torch.load(model_filename))

    optimizer = optim.AdamW(model.parameters(), lr=hparams["MODEL_LR"])
    optimizer.load_state_dict(torch.load(model_filename.replace("model", "optimizer")))

    episode = int(model_filename.split("_e")[-1].split(".")[0])
    return model, optimizer, episode, hparams, model_timestamp


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )

    argparser.add_argument(
        "--timestamp",
        type=str,
        help="Timestamp of the model to resume training from. Optional; --resume without timestamp will resume from the latest checkpoint.",
        required=False,
    )

    argparser.add_argument(
        "--episodes",
        type=int,
        help="Number of episodes to train for",
        default=100000,
    )

    args = argparser.parse_args()

    device = CNNTrainer.select_device()
    if args.resume:
        model_filename = find_latest_model_filename(args.timestamp)
        if model_filename is None:
            raise FileNotFoundError("No model files found to resume training from")
        logger.info(f"Resuming training from {model_filename}")
        model, optimizer, start_episode, hparams, model_timestamp = load_from_checkpoint(model_filename, device)
        # Ovewrite the app config with the latest values
        app_config = AppConfig(**hparams)
        # approximate start step based on learn steps, explore steps, and episode number
        start_step = (start_episode // (app_config.MODEL_BATCH_SIZE * app_config.MODEL_EXPLORE_EPISODES)) * app_config.MODEL_LEARN_STEPS
    else:
        app_config = AppConfig()
        model = ChessCNN(num_filters=app_config.MODEL_NUM_FILTERS, num_residual_blocks=app_config.MODEL_RESIDUAL_BLOCKS).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=app_config.MODEL_LR)
        model_timestamp = None
        start_episode = 0
        start_step = 0

    trainer = CNNTrainer(app_config=app_config, model_timestamp=model_timestamp)
    trainer.train_deep_q_network_off_policy(
        model,
        optimizer,
        episodes=args.episodes,
        app_config=app_config,
        start_episode=start_episode,
        start_step=start_step,
    )

