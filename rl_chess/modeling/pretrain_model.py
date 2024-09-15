import datetime
import os
import gzip
import pickle
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import List, Tuple
from rl_chess import base_path
from rl_chess.modeling.chess_cnn import ChessCNN
from rl_chess.modeling.chess_transformer import ChessTransformer
from rl_chess.modeling.chess_cnn_transformer import ChessCNNTransformer
from rl_chess.modeling.chess_ensemble import EnsembleCNNTransformer
from rl_chess.modeling.experience_buffer import FullEvaluationRecord
from rl_chess.config.config import AppConfig


class ChessDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".pkl.gzip")][:4]
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


def pretrain_model(
    model: ChessCNN,
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[ChessCNN, List[float], List[float]]:
    model.to(device)
    model.train()

    # Define loss functions
    mse_loss = nn.MSELoss(
        reduction="none"
    )  # Changed to 'none' to allow element-wise multiplication
    bce_loss = nn.BCEWithLogitsLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create dataset and dataloader
    data_dir = base_path / "data" / "full_move_evals"
    dataset = ChessDataset(str(data_dir))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    q_losses = []
    aux_losses = []

    for epoch in range(num_epochs):
        epoch_q_loss = 0.0
        epoch_aux_loss = 0.0

        for states, legal_moves_masks, rewards in tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            states = states.to(device)
            legal_moves_masks = legal_moves_masks.to(device)
            rewards = rewards.to(device)

            # Forward pass
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
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update epoch losses
            epoch_q_loss += q_loss.item()
            epoch_aux_loss += aux_loss.item()

        # Calculate average losses for the epoch
        avg_q_loss = epoch_q_loss / len(dataloader)
        avg_aux_loss = epoch_aux_loss / len(dataloader)

        q_losses.append(avg_q_loss)
        aux_losses.append(avg_aux_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Q-Loss: {avg_q_loss:.4f}, Aux-Loss: {avg_aux_loss:.4f}"
        )

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
        cnn=ChessCNN(num_filters=128, num_residual_blocks=4, negative_slope=0.0),
        transformer=ChessTransformer(
            d_model=128,
            nhead=4,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.0,
            freeze_pos=False,
            add_global=False,
        ),
    )
    pretrained_model, q_losses, aux_losses = pretrain_model(
        model,
        num_epochs=50,
        learning_rate=1e-4,
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    with open(
        base_path / "out" / f"pretrained_{model.__class__.__name__}_{timestamp}.pkl", "wb"
    ) as f:
        pickle.dump(pretrained_model, f)
