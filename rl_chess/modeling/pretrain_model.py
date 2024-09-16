import datetime
import gzip
import json
import os
import pickle
import uuid
from typing import List, Tuple

import torch
import torch.amp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from rl_chess import base_path
from rl_chess.config.config import AppConfig
from rl_chess.modeling.chess_cnn import ChessCNN
from rl_chess.modeling.chess_cnn_transformer import ChessCNNTransformer
from rl_chess.modeling.chess_ensemble import EnsembleCNNTransformer
from rl_chess.modeling.chess_transformer import ChessTransformer
from rl_chess.modeling.experience_buffer import FullEvaluationRecord


class ChessDataset(Dataset):
    def __init__(self, data_dir: str | None = None, file_list: List[str] | None = None):
        if not data_dir and not file_list:
            raise ValueError("Either data_dir or file_list must be provided")
        self.data_dir = data_dir
        if not file_list:
            self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".pkl.gzip")]
        else:
            self.file_list = file_list
        self.data = []
        self._load_data()

    def _load_data(self):
        for file_name in tqdm(self.file_list, desc="Loading data"):
            file_path = os.path.join(self.data_dir, file_name)
            with gzip.open(file_path, "rb") as f:
                records = pickle.load(f)
                for serialized_record in records:
                    record = FullEvaluationRecord.from_serialized(serialized_record)
                    self.data.append(
                        (record.state, record.legal_moves_mask, record.rewards)
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def save_model(model: nn.Module, name: str):
    with open(
        base_path / "out" / f"pretrained_{model.__class__.__name__}_{name}.pkl",
        "wb",
    ) as f:
        pickle.dump(model, f)


def save_hparams(hparams: dict, name: str):
    with open(base_path / "out" / f"pretrained_{name}_hparams.json", "w") as f:
        json.dump(hparams, f)


def validate_model(
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[float, float]:
    model.eval()

    mse_loss = nn.MSELoss(reduction="none")

    q_losses = []
    aux_losses = []

    with torch.no_grad():
        for states, legal_moves_masks, rewards in test_loader:
            states = states.to(device)
            legal_moves_masks = legal_moves_masks.to(device)
            rewards = rewards.to(device)

            predicted_q_values, aux_logits = model(states)

            element_wise_mse = mse_loss(predicted_q_values, rewards)
            masked_mse = element_wise_mse * legal_moves_masks
            q_loss = (
                masked_mse.sum() / legal_moves_masks.sum()
            )  # Normalize by number of legal moves

            aux_loss = nn.BCEWithLogitsLoss()(aux_logits, legal_moves_masks)

            q_losses.append(q_loss.item())
            aux_losses.append(aux_loss.item())

    avg_q_loss = sum(q_losses) / len(q_losses)
    avg_aux_loss = sum(aux_losses) / len(aux_losses)

    return avg_q_loss, avg_aux_loss


def pretrain_model(
    model: nn.Module,
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    timestamp: str | None = None,
) -> Tuple[nn.Module, List[float], List[float]]:
    model.to(device)
    model.train()

    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    writer = SummaryWriter(
        log_dir=base_path
        / app_config.APP_TENSORBOARD_DIR
        / f"pretraining/{model.__class__.__name__}_{timestamp}",
    )

    # Define loss functions
    mse_loss = nn.MSELoss(
        reduction="none"
    )  # Changed to 'none' to allow element-wise multiplication
    bce_loss = nn.BCEWithLogitsLoss()

    # Create dataset and dataloader
    data_dir = base_path / "data" / "full_move_evals"
    files = [f for f in os.listdir(data_dir) if f.endswith(".pkl.gzip")]
    train_files, test_files = train_test_split(files, test_size=0.1)
    train_dataset = ChessDataset(data_dir=str(data_dir), file_list=train_files)
    test_dataset = ChessDataset(data_dir=str(data_dir), file_list=test_files)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=num_epochs * len(train_dataloader),
    )

    # Training loop
    q_losses = []
    aux_losses = []
    step = 0

    scaler = torch.amp.grad_scaler.GradScaler(enabled=(device == "cuda"))

    for epoch in range(num_epochs):
        epoch_q_loss = 0.0
        epoch_aux_loss = 0.0
        last_q_loss = 0.0
        last_aux_loss = 0.0
        last_log_step = step

        for states, legal_moves_masks, rewards in tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            model.train()
            optimizer.zero_grad()

            step += 1
            states = states.to(device)
            legal_moves_masks = legal_moves_masks.to(device)
            rewards = rewards.to(device)

            # Forward pass
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                predicted_q_values, aux_logits = model(states)

                # Calculate losses
                element_wise_mse = mse_loss(predicted_q_values, rewards)
                masked_mse = element_wise_mse * legal_moves_masks
                q_loss = (
                    masked_mse.sum() / legal_moves_masks.sum()
                )  # Normalize by number of legal moves

                aux_loss = bce_loss(aux_logits, legal_moves_masks)
                total_loss = q_loss + aux_loss

            # Backward pass and optimize
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            # Update epoch losses
            epoch_q_loss += q_loss.item()
            epoch_aux_loss += aux_loss.item()

            if step % 1000 == 0:
                writer.add_scalar(
                    "Train/Q-Loss",
                    (epoch_q_loss - last_q_loss) / (step - last_log_step),
                    step,
                )
                writer.add_scalar(
                    "Train/Aux-Loss",
                    (epoch_aux_loss - last_aux_loss) / (step - last_log_step),
                    step,
                )
                writer.add_scalar("LR", lr_scheduler.get_last_lr()[0], step)
                last_q_loss = epoch_q_loss
                last_aux_loss = epoch_aux_loss
                last_log_step = step
            if step % 4000 == 0:
                test_q_loss, test_aux_loss = validate_model(model, test_dataloader, device)
                writer.add_scalar("Validation/Q-Loss", test_q_loss, step)
                writer.add_scalar("Validation/Aux-Loss", test_aux_loss, step)

        # Calculate average losses for the epoch
        avg_q_loss = epoch_q_loss / len(train_dataloader)
        avg_aux_loss = epoch_aux_loss / len(train_dataloader)

        q_losses.append(avg_q_loss)
        aux_losses.append(avg_aux_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Q-Loss: {avg_q_loss:.4f}, Aux-Loss: {avg_aux_loss:.4f}"
        )
        save_model(model, f"{timestamp}_epoch_{epoch+1}")

    return model, q_losses, aux_losses


if __name__ == "__main__":
    app_config = AppConfig()
    # model = ChessCNN(
    #     num_filters=128,
    #     num_residual_blocks=4,
    #     negative_slope=0.0
    # )
    # model = ChessTransformer(
    #     d_model=128,
    #     nhead=4,
    #     num_layers=4,
    #     dim_feedforward=512,
    #     dropout=0.0,
    #     freeze_pos=True,
    #     add_global=False,
    # )
    # model = ChessCNNTransformer(
    #     num_filters=128,
    #     num_residual_blocks=5,
    #     d_model=128,
    #     nhead=4,
    #     num_transformer_layers=4,
    #     dim_feedforward=512,
    #     dropout=0.0,
    # )
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
    )
    hparams = dict(
        MODEL_TRANSFORMER_D_MODEL=app_config.MODEL_TRANSFORMER_D_MODEL,
        MODEL_TRANSFORMER_NUM_HEADS=app_config.MODEL_TRANSFORMER_NUM_HEADS,
        MODEL_TRANSFORMER_NUM_LAYERS=app_config.MODEL_TRANSFORMER_NUM_LAYERS,
        MODEL_TRANSFORMER_DIM_FEEDFORWARD=app_config.MODEL_TRANSFORMER_DIM_FEEDFORWARD,
        MODEL_TRANSFORMER_DROPOUT=app_config.MODEL_TRANSFORMER_DROPOUT,
        MODEL_TRANSFORMER_FREEZE_POS=app_config.MODEL_TRANSFORMER_FREEZE_POS,
        MODEL_TRANSFORMER_ADD_GLOBAL=app_config.MODEL_TRANSFORMER_ADD_GLOBAL,
        MODEL_CNN_NUM_FILTERS=app_config.MODEL_CNN_NUM_FILTERS,
        MODEL_CNN_RESIDUAL_BLOCKS=app_config.MODEL_CNN_RESIDUAL_BLOCKS,
        MODEL_CNN_NEGATIVE_SLOPE=app_config.MODEL_CNN_NEGATIVE_SLOPE,
        MODEL_CNN_DROPOUT=app_config.MODEL_CNN_DROPOUT,
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_hparams(hparams, f"{model.__class__.__name__}_{timestamp}")
    pretrained_model, q_losses, aux_losses = pretrain_model(
        model,
        num_epochs=5,
        batch_size=128,
        learning_rate=4e-4,
        timestamp=timestamp,
    )
    save_model(pretrained_model, f"{timestamp}_final")
