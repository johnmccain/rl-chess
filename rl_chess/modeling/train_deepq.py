import argparse
import datetime
import copy
import json
import logging
import os
import pathlib
import pickle
import random
from collections import defaultdict
import gzip

import chess
import pandas as pd
import torch
from torch import nn
import torch.amp
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rl_chess import base_path
from rl_chess.config.config import AppConfig
from rl_chess.evaluation.stockfish_evaluator import StockfishEvaluator
from rl_chess.modeling.chess_cnn import ChessCNN
from rl_chess.modeling.chess_transformer import ChessTransformer
from rl_chess.modeling.chess_ensemble import EnsembleCNNTransformer
from rl_chess.modeling.experience_buffer import ExperienceBuffer, ExperienceRecord
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
    handlers=[logging.StreamHandler(), logging.FileHandler("train_deepq.log")],
)

logger = logging.getLogger(__name__)


class DeepQTrainer:
    CURRICULUM_BUFFER_PATH = base_path / "data" / "curriculum_buffers"

    def __init__(
        self,
        device: str | None = None,
        app_config: AppConfig = AppConfig(),
        model_timestamp: str | None = None,
        stockfish_evaluator: StockfishEvaluator | None = None,
    ) -> None:

        if device:
            self.device = torch.device(device)
        else:
            self.device = self.select_device()

        # Load curriculum data
        self.openings_df = pd.read_csv(base_path / "data/openings_fen7.csv")

        self.model_timestamp = model_timestamp or datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )

        self.stockfish_evaluator = stockfish_evaluator or StockfishEvaluator(app_config)

        self.writer = SummaryWriter(
            log_dir=base_path
            / app_config.APP_TENSORBOARD_DIR
            / f"qchess_{self.model_timestamp}",
        )

        self.experience_buffer = ExperienceBuffer(app_config.MODEL_BUFFER_SIZE)
        self.buffer_file_list = [
            f
            for f in os.listdir(self.CURRICULUM_BUFFER_PATH)
            if f.endswith(".pkl.gzip") or f.endswith(".pkl") or f.endswith(".pkl.gz")
        ]
        self.app_config = app_config

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
            exp.make_serializeable() for exp in self.experience_buffer.buffer
        ]
        with open(filename, "wb") as f:
            pickle.dump(saveable_experience_buffer, f)

    def augment_experience_buffer(self, path: str | pathlib.Path | None = None) -> None:
        """
        Load an experience buffer from a file and append it to the current experience buffer.

        :param path: The path to the experience buffer file. If None, a random buffer file will be selected.
        """
        if path is None:
            # Use a random buffer file
            if not self.buffer_file_list:
                logger.warning("No buffer files available!")
                return
            idx = random.randint(0, len(self.buffer_file_list) - 1)
            filename = self.buffer_file_list.pop(idx)
            path = self.CURRICULUM_BUFFER_PATH / filename

        logger.info(f"Augmenting experience buffer from {path}")
        if str(path).endswith(".gz") or str(path).endswith(".gzip"):
            with gzip.open(path, "rb") as f:
                experiences = pickle.load(f)
        else:
            with open(path, "rb") as f:
                experiences = pickle.load(f)
        for exp in experiences:
            experience_record = ExperienceRecord.from_serialized(exp)

            experience_record.state = experience_record.state.to(self.device)
            experience_record.next_state = experience_record.next_state.to(self.device)
            experience_record.legal_moves_mask = experience_record.legal_moves_mask.to(
                self.device
            )
            experience_record.next_legal_moves_mask = experience_record.next_legal_moves_mask.to(
                self.device
            )
            experience_record.action = experience_record.action.to(self.device)
            if experience_record.pred_q_values:
                experience_record.pred_q_values = experience_record.pred_q_values.to(
                    self.device
                )
            self.experience_buffer.add(experience_record)

    def save_checkpoint(
        self, model: nn.Module, optimizer: optim.Optimizer, episode: int
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
        optimizer_files = list(
            checkpoint_dir.glob(f"optimizer_{self.model_timestamp}_e*.pt")
        )
        buffer_files = list(
            checkpoint_dir.glob(f"experience_buffer_{self.model_timestamp}_e*.pkl")
        )

        # Sort files by episode number
        def get_episode(filename):
            return int(filename.stem.split("_e")[-1])

        model_files.sort(key=get_episode, reverse=True)
        optimizer_files.sort(key=get_episode, reverse=True)
        buffer_files.sort(key=get_episode, reverse=True)

        # Keep only the last 5 checkpoints
        for file_list in [model_files, optimizer_files, buffer_files]:
            for file in file_list[5:]:
                os.remove(file)
                logger.info(f"Deleted old checkpoint file: {file}")

    def save_hparams(self, model: nn.Module, app_config: AppConfig) -> None:
        hparams = {
            "MODEL_CLASS": model.__class__.__name__,

            "MODEL_CNN_NUM_FILTERS": app_config.MODEL_CNN_NUM_FILTERS,
            "MODEL_CNN_RESIDUAL_BLOCKS": app_config.MODEL_CNN_RESIDUAL_BLOCKS,
            "MODEL_CNN_NEGATIVE_SLOPE": app_config.MODEL_CNN_NEGATIVE_SLOPE,
            "MODEL_CNN_DROPOUT": app_config.MODEL_CNN_DROPOUT,

            "MODEL_TRANSFORMER_NUM_HEADS": app_config.MODEL_TRANSFORMER_NUM_HEADS,
            "MODEL_TRANSFORMER_NUM_LAYERS": app_config.MODEL_TRANSFORMER_NUM_LAYERS,
            "MODEL_TRANSFORMER_D_MODEL": app_config.MODEL_TRANSFORMER_D_MODEL,
            "MODEL_TRANSFORMER_DIM_FEEDFORWARD": app_config.MODEL_TRANSFORMER_DIM_FEEDFORWARD,
            "MODEL_TRANSFORMER_DROPOUT": app_config.MODEL_TRANSFORMER_DROPOUT,
            "MODEL_TRANSFORMER_FREEZE_POS": app_config.MODEL_TRANSFORMER_FREEZE_POS,
            "MODEL_TRANSFORMER_ADD_GLOBAL": app_config.MODEL_TRANSFORMER_ADD_GLOBAL,

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
        with open(
            base_path
            / app_config.APP_OUTPUT_DIR
            / f"hparams_{self.model_timestamp}.json",
            "w",
        ) as f:
            json.dump(hparams, f)

    @staticmethod
    def select_device() -> torch.device:
        if torch.cuda.is_available():
            logger.info("CUDA available, using GPU")
            device = torch.device("cuda")
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
        self,
        model: nn.Module,
        episodes: int,
        gamma: float,
        epsilon: float,
        batch_size: int,
    ) -> list[ExperienceRecord]:
        experience_buffer = []
        for episode in tqdm(range(episodes), total=episodes, desc="Exploring"):

            boards = [
                chess.Board() if random.random() > 0.25 else self.sample_opening_state()
                for _ in range(batch_size)
            ]
            n_moves = 0

            enable_amp = self.device.type == "cuda"

            while (
                boards
                and not all(board.is_game_over() for board in boards)
                and n_moves < app_config.MODEL_MAX_MOVES
            ):
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=enable_amp):
                    current_states = torch.stack(
                        [board_to_tensor(board, board.turn) for board in boards], dim=0
                    ).to(self.device)

                    turns = [board.turn for board in boards]

                    ### Calculate Player move ###

                    logger.debug("START Predicting Q values")
                    # Predict Q-values
                    predicted_q_values, _ = model(current_states)
                    logger.debug("END Predicting Q values")

                    logger.debug("START Masking illegal moves")
                    # Mask illegal moves
                    legal_moves_mask = torch.stack(
                        [get_legal_moves_mask(board) for board in boards], dim=0
                    ).to(self.device)
                    masked_q_values = predicted_q_values.masked_fill(
                        legal_moves_mask == 0, torch.finfo(predicted_q_values.dtype).min
                    )
                    logger.debug("END Masking illegal moves")

                    logger.debug("START Selecting action")
                    actions = torch.stack(
                        [
                            self.select_action(masked_q_values[idx], epsilon, board)
                            for idx, board in enumerate(boards)
                        ],
                        dim=0,
                    )
                    logger.debug("END Selecting action")

                    logger.debug("START Taking action & calculating reward")
                    # Take action and observe reward and next state
                    moves = [
                        index_to_move(actions[idx].item(), board)
                        for idx, board in enumerate(boards)
                    ]
                    if any(move is None for move in moves):
                        logger.warning("Invalid move selected!")
                        # No handling for invalid moves; just break out of the loop
                        break
                    rewards = torch.tensor(
                        [
                            calculate_reward(board, move)
                            for board, move in zip(boards, moves)
                        ]
                    ).to(self.device)
                    # Calculate default reward for opponent move (if opponent can't make a move such as in checkmate or stalemate)
                    default_opp_rewards = [
                        calculate_reward(board, move, flip_perspective=True)
                        for board, move in zip(boards, moves)
                    ]

                    # Take the action
                    for move, board in zip(moves, boards):
                        board.push(move)
                    logger.debug("END Taking action & calculating reward")

                    ### Calculate Opponent Move ###
                    logger.debug("START Preparing Opponent move")
                    opp_next_states = torch.stack(
                        [board_to_tensor(board, board.turn) for board in boards], dim=0
                    ).to(self.device)

                    # To process as a batch, we continue processing games that are over with dummy values, but need to ensure that we don't push moves to the board
                    dones = torch.tensor(
                        [int(board.is_game_over()) for board in boards], device=self.device
                    )
                    logger.debug("END Preparing Opponent move")

                    logger.debug("START Predicting Opponent Q values")
                    # Select opponent next move
                    opp_next_q_values, _ = model(opp_next_states)
                    logger.debug("END Predicting Opponent Q values")

                    logger.debug("START Masking illegal moves for Opponent")
                    opp_next_legal_moves_mask = torch.stack(
                        [get_legal_moves_mask(board) for board in boards], dim=0
                    ).to(self.device)
                    opp_masked_next_q_values = opp_next_q_values.masked_fill(
                        opp_next_legal_moves_mask == 0,
                        torch.finfo(opp_next_q_values.dtype).min,
                    )
                    logger.debug("END Masking illegal moves for Opponent")

                    logger.debug("START Selecting Opponent action")
                    opp_actions = [
                        (
                            self.select_action(
                                opp_masked_next_q_values[idx], epsilon, board
                            )
                            if not done
                            else -1
                        )  # -1 is a placeholder when there is no valid move
                        for idx, (board, done) in enumerate(zip(boards, dones))
                    ]
                    logger.debug("END Selecting Opponent action")

                    logger.debug("START Taking Opponent action & calculating reward")
                    # Take opponent action
                    opp_moves = [
                        index_to_move(opp_action, board) if opp_action != -1 else None
                        for opp_action, board in zip(opp_actions, boards)
                    ]

                    # Calculate reward for opponent move
                    opp_rewards = torch.tensor(
                        [
                            calculate_reward(board, opp_move) if opp_move else default
                            for opp_move, default, board in zip(
                                opp_moves, default_opp_rewards, boards
                            )
                        ]
                    ).to(self.device)
                    for board, opp_move in zip(boards, opp_moves):
                        if opp_move:
                            board.push(opp_move)
                    # Check if the game is over after the opponent move
                    opp_dones = torch.tensor(
                        [int(board.is_game_over()) for board in boards], device=self.device
                    )
                    logger.debug("END Taking Opponent action & calculating reward")

                    ### Calculate next state and reward ###
                    # Compute the next-state max Q-value for active player given the opponent's possible move
                    # next_states for done boards are ignored, but we need to provide a tensor of the correct shape so we use the current state (since no move was pushed)
                    logger.debug("START Preparing next state")
                    next_states = torch.stack(
                        [board_to_tensor(board, board.turn) for board in boards], dim=0
                    ).to(self.device)
                    logger.debug("END Preparing next state")

                    logger.debug("START Predicting next Q values")
                    next_q_values, _ = model(next_states)
                    logger.debug("END Predicting next Q values")

                    logger.debug("START Masking illegal moves for next state")
                    next_legal_moves_mask = torch.stack(
                        [get_legal_moves_mask(board) for board in boards], dim=0
                    ).to(self.device)

                    masked_next_q_values = next_q_values.masked_fill(
                        next_legal_moves_mask == 0,
                        torch.finfo(next_q_values.dtype).min,
                    )
                    max_next_q_values = masked_next_q_values.max(1)[0]
                    max_next_q_values[
                        max_next_q_values == torch.finfo(max_next_q_values.dtype).min,
                    ] = 0.0  # Set Q-value to 0.0 for situations where no legal moves are available
                    logger.debug("END Masking illegal moves for next state")

                    logger.debug("START Calculating target Q value")
                    # Roll back the board state to before the opponent move (if any opponent move was made)
                    for board, opp_move in zip(boards, opp_moves):
                        if opp_move:
                            board.pop()

                    # Handle boards where opponent had no valid moves (game was over)
                    # NOTE: is 0.0 the correct default value for max_next_q_values?
                    max_next_q_values.masked_fill(dones == 1, 0.0)
                    # Compute the target Q-value
                    target_q_value = (
                        rewards - opp_rewards + (gamma * max_next_q_values * (1 - dones))
                    )
                    predicted_q = predicted_q_values.gather(1, actions)
                    logger.debug("END Calculating target Q value")

                logger.debug("START Creating ExperienceRecord")
                # Create separate ExperienceRecord for each board
                for (
                    current_state,
                    legal_mask,
                    action,
                    reward,
                    next_state,
                    next_legal_mask,
                    done,
                    opp_done,
                    predicted_q,
                    target_q_value,
                    pred_q,
                    max_next_q,
                    turn,
                ) in zip(
                    current_states,
                    legal_moves_mask,
                    actions,
                    rewards,
                    next_states,
                    next_legal_moves_mask,
                    dones,
                    opp_dones,
                    predicted_q,
                    target_q_value,
                    predicted_q_values,
                    max_next_q_values,
                    turns,
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
                            max_next_q=max_next_q.item(),
                            color=turn,
                        )
                    )
                logger.debug("END Creating ExperienceRecord")
                # Remove any games that are done
                boards = [board for board in boards if not board.is_game_over()]
                n_moves += 1
        return experience_buffer

    def learn(
        self,
        model: nn.Module,
        target_model: nn.Module,
        optimizer: optim.Optimizer,
        scaler: torch.amp.GradScaler,
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
        model.train()
        enable_amp = self.device.type == "cuda"
        last_logged_loss = 0.0
        last_logged_q_loss = 0.0
        last_logged_aux_loss = 0.0
        last_logged_step = 0
        for step in tqdm(range(steps), total=steps, desc="Learning"):
            optimizer.zero_grad()

            batch = self.experience_buffer.sample_n(batch_size)
            state_batch = torch.stack([exp.state for exp in batch], dim=0)
            legal_moves_mask_batch = torch.stack(
                [exp.legal_moves_mask for exp in batch], dim=0
            )
            action_batch = torch.stack([exp.action for exp in batch], dim=0)
            reward_batch = torch.tensor(
                [exp.reward for exp in batch], device=self.device
            )
            next_state_batch = torch.stack([exp.next_state for exp in batch], dim=0)
            next_legal_moves_mask_batch = torch.stack(
                [exp.next_legal_moves_mask for exp in batch], dim=0
            )
            done_batch = torch.tensor(
                [int(exp.done) for exp in batch], device=self.device
            )
            opp_done_batch = torch.tensor(
                [int(exp.opp_done) for exp in batch], device=self.device
            )

            with torch.amp.autocast("cuda", enabled=enable_amp):
                predicted_q_values, aux_logits = model(state_batch)
                predicted_q = predicted_q_values.gather(1, action_batch)

            # Use target network for next Q-values
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=enable_amp):
                next_q_values, _ = target_model(next_state_batch)
                masked_next_q_values = next_q_values.masked_fill(
                    next_legal_moves_mask_batch == 0, torch.finfo(next_q_values.dtype).min
                )
                max_next_q_values = masked_next_q_values.max(1)[0]
                masked_max_next_q_values = (
                    max_next_q_values * (1 - done_batch) * (1 - opp_done_batch)
                )
                discounted_max_next_q_values = gamma * masked_max_next_q_values
                target_q_values = reward_batch + discounted_max_next_q_values

            # Compute loss
            q_loss = q_loss_fn(predicted_q, target_q_values.unsqueeze(1))
            total_q_loss += q_loss.item()
            aux_loss = aux_loss_fn(aux_logits, legal_moves_mask_batch)
            total_aux_loss += aux_loss.item()
            loss = q_loss + aux_loss

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale before gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), app_config.MODEL_CLIP_GRAD
            )
            scaler.step(optimizer)
            scaler.update()

            if step - last_logged_step >= 100:
                total_loss = total_q_loss + total_aux_loss
                log_loss = (total_loss - last_logged_loss) / (step - last_logged_step)
                log_q_loss = (total_q_loss - last_logged_q_loss) / (
                    step - last_logged_step
                )
                log_aux_loss = (total_aux_loss - last_logged_aux_loss) / (
                    step - last_logged_step
                )

                self.writer.add_scalar("Train/Loss", log_loss, step + step_offset)
                self.writer.add_scalar("Train/QLoss", log_q_loss, step + step_offset)
                self.writer.add_scalar("Train/AuxLoss", log_aux_loss, step + step_offset)

                last_logged_loss = total_loss
                last_logged_q_loss = total_q_loss
                last_logged_aux_loss = total_aux_loss
                last_logged_step = step

            # Update target network
            if (step + 1) % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

        return total_q_loss, total_aux_loss

    def eval_model_aux(
        self, model: nn.Module, steps: int, batch_size: int, step: int
    ) -> float:
        """
        Evaluate the model on the aux task (move legality).
        """
        correct = 0
        total = 0
        aux_val_loss = 0.0
        predictions = []
        targets = []
        for _ in tqdm(range(steps), total=steps, desc="Evaluating"):
            batch = self.experience_buffer.sample_n(batch_size)
            state_batch = torch.stack([exp.state for exp in batch], dim=0)
            legal_moves_mask_batch = torch.stack(
                [exp.legal_moves_mask for exp in batch], dim=0
            )
            with torch.no_grad():
                _, aux_logits = model(state_batch)
            predicted_labels = torch.round(torch.sigmoid(aux_logits))
            correct += (predicted_labels == legal_moves_mask_batch).sum().item()
            total += legal_moves_mask_batch.numel()
            aux_val_loss += F.binary_cross_entropy_with_logits(
                aux_logits, legal_moves_mask_batch
            ).item()
            predictions.extend(predicted_labels.cpu().numpy().flatten())
            targets.extend(legal_moves_mask_batch.cpu().numpy().flatten())
        accuracy = correct / total
        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)
        self.writer.add_scalar("Validation/AuxAccuracy", accuracy, step)
        self.writer.add_scalar("Validation/AuxPrecision", precision, step)
        self.writer.add_scalar("Validation/AuxRecall", recall, step)
        self.writer.add_scalar("Validation/AuxLoss", aux_val_loss, step)
        return aux_val_loss

    def eval_model_sf(self, model: nn.Module, step: int, app_config: AppConfig) -> dict:
        """
        Evaluate the model using a CSV of FEN states and types based on Stockfish evaluations.
        Calculates KPIs overall and by game state type.
        """
        df = pd.read_csv(base_path / "data" / app_config.APP_MOVE_EVAL_DATASET)

        kpis_by_type = defaultdict(lambda: defaultdict(float))
        overall_kpis = defaultdict(float)
        counts_by_type = defaultdict(int)

        model.eval()
        with torch.no_grad():
            for fen, board_type in tqdm(
                zip(df["fen"], df["type"]), total=len(df), desc="Evaluating"
            ):
                board = chess.Board(fen)
                state = board_to_tensor(board, board.turn).unsqueeze(0).to(self.device)
                legal_moves_mask = (
                    get_legal_moves_mask(board).unsqueeze(0).to(self.device)
                )

                # Model predictions
                q_values, _ = model(state)

                counts_by_type[board_type] += 1

                # Move quality evaluation
                masked_q_values = q_values.masked_fill(
                    legal_moves_mask == 0, float("-inf")
                )
                move_index = masked_q_values.argmax().item()
                move = index_to_move(move_index, board)

                if move:
                    move_quality = self.stockfish_evaluator.rate_action(board, move)
                    is_blunder = move_quality < app_config.STOCKFISH_BLUNDER_THRESHOLD
                    stockfish_best_move = self.stockfish_evaluator.move(board)
                    is_optimal = move == stockfish_best_move

                    # Update KPIs
                    for kpi_dict in [kpis_by_type[board_type], overall_kpis]:
                        kpi_dict["move_quality"] += move_quality
                        kpi_dict["blunder_count"] += int(is_blunder)
                        kpi_dict["optimal_count"] += int(is_optimal)
                        kpi_dict["total_moves"] += 1

        # Calculate averages and rates
        for kpi_dict in list(kpis_by_type.values()) + [overall_kpis]:
            total_moves = kpi_dict["total_moves"]
            if total_moves > 0:
                kpi_dict["avg_move_quality"] = kpi_dict["move_quality"] / total_moves
                kpi_dict["blunder_rate"] = kpi_dict["blunder_count"] / total_moves
                kpi_dict["optimal_rate"] = kpi_dict["optimal_count"] / total_moves
            del (
                kpi_dict["move_quality"],
                kpi_dict["blunder_count"],
                kpi_dict["optimal_count"],
                kpi_dict["total_moves"],
            )

        # Log to TensorBoard
        for board_type, kpi_dict in kpis_by_type.items():
            for kpi, value in kpi_dict.items():
                self.writer.add_scalar(
                    f"Validation/{board_type}/{kpi}", value, step
                )

        for kpi, value in overall_kpis.items():
            self.writer.add_scalar(f"Validation/{kpi}", value, step)

        return {"overall": overall_kpis, "by_type": dict(kpis_by_type)}

    def fill_experience_buffer(
        self,
        model: nn.Module,
        epsilon: float,
        gamma: float,
        app_config: AppConfig,
        use_curriculum_experiences: bool = True,
    ) -> None:
        """
        Fill the experience buffer with random exploration or pre-generated experiences.

        :param model: The model to use for exploration
        :param epsilon: The exploration rate
        :param gamma: The discount factor
        :param app_config: The AppConfig object
        :param use_curriculum_experiences: Whether to use pre-generated experiences
        """
        if use_curriculum_experiences:
            # Load curriculum experience buffer
            logger.info("Filling experience buffer with curriculum data")
            while len(self.experience_buffer.buffer) < app_config.MODEL_BUFFER_SIZE and self.buffer_file_list:
                self.augment_experience_buffer()

        if len(self.experience_buffer.buffer) >= app_config.MODEL_BUFFER_SIZE:
            # Buffer is already full from curriculum data
            return

        logger.info("Filling experience buffer with random exploration")
        # Fill the experience buffer
        # We don't count this towards the total number of episodes
        desired_additional_experience = app_config.MODEL_BUFFER_SIZE - len(
            self.experience_buffer.buffer
        )
        estimated_experience_rate = (
            app_config.MODEL_BATCH_SIZE * app_config.MODEL_MAX_MOVES
        )  # Estimated experience per episode batch, start with max moves per episode

        while len(self.experience_buffer.buffer) < app_config.MODEL_BUFFER_SIZE:
            desired_additional_experience = app_config.MODEL_BUFFER_SIZE - len(
                self.experience_buffer.buffer
            )
            est_num_episodes = max(
                int(desired_additional_experience // estimated_experience_rate), 1
            )
            new_experiences = self.explore(
                model, est_num_episodes, gamma, epsilon, app_config.MODEL_BATCH_SIZE
            )
            self.experience_buffer.extend(new_experiences)
            # Update estimated experience rate
            estimated_experience_rate = len(new_experiences) / est_num_episodes
            logger.info(
                f"Experience buffer size: {len(self.experience_buffer.buffer)}/{app_config.MODEL_BUFFER_SIZE}"
            )

    def train_deep_q_network_off_policy(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        episodes: int,
        app_config: AppConfig = AppConfig(),
        start_episode: int = 0,
        start_step: int = 0,
        use_curriculum_experiences: bool = True,
    ) -> nn.Module:
        model.to(self.device)
        target_model = copy.deepcopy(model).to(self.device)
        target_model.load_state_dict(model.state_dict())
        target_model.eval()

        t_max = episodes // (
            app_config.MODEL_EXPLORE_EPISODES * app_config.MODEL_BATCH_SIZE
        )  # Number of epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=1e-6
        )

        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        if start_episode > 0:
            # Convert start_episode to the corresponding epoch
            start_epoch = start_episode // (
                app_config.MODEL_EXPLORE_EPISODES * app_config.MODEL_BATCH_SIZE
            )
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
        gamma = min(
            app_config.MODEL_INITIAL_GAMMA,
            app_config.MODEL_GAMMA
            + gamma_ramp * max(episode - app_config.MODEL_GAMMA_STARTUP_STEPS, 0),
        )
        epsilon = max(
            (app_config.MODEL_DECAY)
            ** (
                episode
                // (app_config.MODEL_EXPLORE_EPISODES * app_config.MODEL_BATCH_SIZE)
            ),
            app_config.MODEL_MIN_EPSILON,
        )
        self.fill_experience_buffer(
            model,
            epsilon,
            gamma,
            app_config,
            use_curriculum_experiences=use_curriculum_experiences,
        )
        experience_counter = 0

        while episode < episodes:
            gamma = min(
                app_config.MODEL_INITIAL_GAMMA
                + gamma_ramp * max(episode - app_config.MODEL_GAMMA_STARTUP_STEPS, 0),
                app_config.MODEL_GAMMA,
            )

            new_experiences = self.explore(
                model,
                app_config.MODEL_EXPLORE_EPISODES,
                gamma,
                epsilon,
                app_config.MODEL_BATCH_SIZE,
            )
            self.experience_buffer.extend(new_experiences)
            experience_counter += len(new_experiences)

            if experience_counter >= app_config.MODEL_BUFFER_SIZE:
                # Experience replay buffer has fully cycled
                experience_counter = 0
                if use_curriculum_experiences and self.buffer_file_list:
                    self.augment_experience_buffer()

            episode += app_config.MODEL_EXPLORE_EPISODES * app_config.MODEL_BATCH_SIZE

            total_q_loss, total_aux_loss = self.learn(
                model=model,
                target_model=target_model,
                optimizer=optimizer,
                scaler=scaler,
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
            self.writer.add_scalar("TotalQLoss", total_q_loss, step)
            self.writer.add_scalar("TotalAuxLoss", total_aux_loss, step)
            self.writer.add_scalar("Epsilon", epsilon, step)
            self.writer.add_scalar("Gamma", gamma, step)
            self.writer.add_scalar("LR", scheduler.get_last_lr()[0], step)

            logger.info(
                f"Episode {episode}, QLoss: {total_q_loss}, AuxLoss: {total_aux_loss}"
            )

            if episode - last_saved_episode > app_config.APP_SAVE_STEPS:
                self.save_checkpoint(model, optimizer, episode)
                last_saved_episode = episode

            if episode - last_eval_episode > app_config.APP_EVAL_STEPS:
                aux_val_loss = self.eval_model_aux(
                    model, 10, app_config.MODEL_BATCH_SIZE, step
                )
                self.eval_model_sf(model, step, app_config=app_config)
                logger.info(f"Episode {episode}, Validation Aux Loss: {aux_val_loss}")
                last_eval_episode = episode

            # Epsilon decay
            epsilon = max(
                (app_config.MODEL_DECAY)
                ** (
                    episode
                    // (app_config.MODEL_EXPLORE_EPISODES * app_config.MODEL_BATCH_SIZE)
                ),
                app_config.MODEL_MIN_EPSILON,
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
        sorted_models = sorted(
            sorted_models,
            key=lambda x: datetime.datetime.strptime(
                x.stem.split("_")[1], "%Y%m%d-%H%M%S"
            ),
        )
    if not sorted_models:
        return None
    return str(sorted_models[-1])


def load_from_checkpoint(
    model_filename: str,
    device: torch.device,
) -> tuple[nn.Module, optim.Optimizer, int, dict, str]:
    # First find the model timestamp from the filename
    model_timestamp = model_filename.split("_e")[0].split("_")[-1]
    # Load hparams from the hparams file
    with open(
        base_path / app_config.APP_OUTPUT_DIR / f"hparams_{model_timestamp}.json", "rb"
    ) as f:
        hparams = json.load(f)

    checkpoint_model_class = hparams["MODEL_CLASS"]

    if checkpoint_model_class == "ChessCNN":
        # Load the model and optimizer
        model = ChessCNN(
            num_filters=hparams["MODEL_CNN_NUM_FILTERS"],
            num_residual_blocks=hparams["MODEL_CNN_RESIDUAL_BLOCKS"],
            negative_slope=hparams["MODEL_CNN_NEGATIVE_SLOPE"],
            dropout=hparams["MODEL_CNN_DROPOUT"],
        ).to(device)
    elif checkpoint_model_class == "ChessTransformer":
        model = ChessTransformer(
            d_model=hparams["MODEL_TRANSFORMER_D_MODEL"],
            nhead=hparams["MODEL_TRANSFORMER_NUM_HEADS"],
            num_layers=hparams["MODEL_TRANSFORMER_NUM_LAYERS"],
            dim_feedforward=hparams["MODEL_TRANSFORMER_DIM_FEEDFORWARD"],
            dropout=hparams["MODEL_TRANSFORMER_DROPOUT"],
            freeze_pos=hparams["MODEL_TRANSFORMER_FREEZE_POS"],
            add_global=hparams["MODEL_TRANSFORMER_ADD_GLOBAL"],
        ).to(device)
    elif checkpoint_model_class == "EnsembleCNNTransformer":
        model = EnsembleCNNTransformer(
            cnn=ChessCNN(
                num_filters=hparams["MODEL_CNN_NUM_FILTERS"],
                num_residual_blocks=hparams["MODEL_CNN_RESIDUAL_BLOCKS"],
                negative_slope=hparams["MODEL_CNN_NEGATIVE_SLOPE"],
                dropout=hparams["MODEL_CNN_DROPOUT"],
            ),
            transformer=ChessTransformer(
                d_model=hparams["MODEL_TRANSFORMER_D_MODEL"],
                nhead=hparams["MODEL_TRANSFORMER_NUM_HEADS"],
                num_layers=hparams["MODEL_TRANSFORMER_NUM_LAYERS"],
                dim_feedforward=hparams["MODEL_TRANSFORMER_DIM_FEEDFORWARD"],
                dropout=hparams["MODEL_TRANSFORMER_DROPOUT"],
                freeze_pos=hparams["MODEL_TRANSFORMER_FREEZE_POS"],
                add_global=hparams["MODEL_TRANSFORMER_ADD_GLOBAL"],
            ),
        ).to(device)

    model.load_state_dict(torch.load(model_filename))

    optimizer = optim.AdamW(model.parameters(), lr=hparams["MODEL_LR"])
    optimizer.load_state_dict(torch.load(model_filename.replace("model", "optimizer")))

    episode = int(model_filename.split("_e")[-1].split(".")[0])
    return model, optimizer, episode, hparams, model_timestamp


def create_model(model_class: str, app_config: AppConfig, device: torch.device) -> nn.Module:
    if model_class == "ChessCNN":
        # Load the model and optimizer
        model = ChessCNN(
            num_filters=app_config.MODEL_CNN_NUM_FILTERS,
            num_residual_blocks=app_config.MODEL_CNN_RESIDUAL_BLOCKS,
            negative_slope=app_config.MODEL_CNN_NEGATIVE_SLOPE,
            dropout=app_config.MODEL_CNN_DROPOUT,
        ).to(device)
    elif model_class == "ChessTransformer":
        model = ChessTransformer(
            d_model=app_config.MODEL_TRANSFORMER_D_MODEL,
            nhead=app_config.MODEL_TRANSFORMER_NUM_HEADS,
            num_layers=app_config.MODEL_TRANSFORMER_NUM_LAYERS,
            dim_feedforward=app_config.MODEL_TRANSFORMER_DIM_FEEDFORWARD,
            dropout=app_config.MODEL_TRANSFORMER_DROPOUT,
            freeze_pos=app_config.MODEL_TRANSFORMER_FREEZE_POS,
            add_global=app_config.MODEL_TRANSFORMER_ADD_GLOBAL,
        ).to(device)
    elif model_class == "EnsembleCNNTransformer":
        model = EnsembleCNNTransformer(
            cnn=ChessCNN(
                num_filters=app_config.MODEL_CNN_NUM_FILTERS,
                num_residual_blocks=app_config.MODEL_CNN_RESIDUAL_BLOCKS,
                negative_slope=app_config.MODEL_CNN_NEGATIVE_SLOPE,
                dropout=app_config.MODEL_CNN_DROPOUT,
            ),
            transformer=ChessTransformer(
                d_model=app_config.MODEL_TRANSFORMER_D_MODEL,
                nhead=app_config.MODEL_TRANSFORMER_NUM_HEADS,
                num_layers=app_config.MODEL_TRANSFORMER_NUM_LAYERS,
                dim_feedforward=app_config.MODEL_TRANSFORMER_DIM_FEEDFORWARD,
                dropout=app_config.MODEL_TRANSFORMER_DROPOUT,
                freeze_pos=app_config.MODEL_TRANSFORMER_FREEZE_POS,
                add_global=app_config.MODEL_TRANSFORMER_ADD_GLOBAL,
            ),
        ).to(device)
    return model


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

    argparser.add_argument(
        "--pretrained",
        type=str,
        help="Path to a pretrained model to load",
        default=None,
    )

    argparser.add_argument(
        "--use-curriculum",
        type=bool,
        help="Whether to use curriculum data for training",
        default=True,
    )

    args = argparser.parse_args()

    device = DeepQTrainer.select_device()
    if args.resume:
        model_filename = find_latest_model_filename(args.timestamp)
        if model_filename is None:
            raise FileNotFoundError("No model files found to resume training from")
        logger.info(f"Resuming training from {model_filename}")
        (
            model,
            optimizer,
            start_episode,
            hparams,
            model_timestamp,
        ) = load_from_checkpoint(model_filename, device)
        # Ovewrite the app config with the latest values
        app_config = AppConfig(**hparams)
        # approximate start step based on learn steps, explore steps, and episode number
        start_step = (
            start_episode
            // (app_config.MODEL_BATCH_SIZE * app_config.MODEL_EXPLORE_EPISODES)
        ) * app_config.MODEL_LEARN_STEPS
    else:
        app_config = AppConfig()
        model = create_model("EnsembleCNNTransformer", app_config, device)
        if args.pretrained:
            logger.info(f"Loading pretrained model from {args.pretrained}")
            with open(base_path/args.pretrained, "rb") as f:
                pretrained_model = pickle.load(f)
            model.load_state_dict(pretrained_model.state_dict())
            del pretrained_model
        optimizer = optim.AdamW(model.parameters(), lr=app_config.MODEL_LR)
        model_timestamp = None
        start_episode = 0
        start_step = 0

    trainer = DeepQTrainer(app_config=app_config, model_timestamp=model_timestamp)
    trainer.train_deep_q_network_off_policy(
        model,
        optimizer,
        episodes=args.episodes,
        app_config=app_config,
        start_episode=start_episode,
        start_step=start_step,
        use_curriculum_experiences=args.use_curriculum,
    )
