import math

import chess
import numpy as np
import torch
from torch import nn


def get_sinusoidal_embeddings(d_model: int) -> np.ndarray:
    """
    Generate sinusoidal embeddings for each position on the chess board.

    :param d_model: The number of dimensions of the embeddings.

    :returns: A max_len x d_model array of sinusoidal embeddings.
    """
    max_len = 64 # Number of positions on the chess board
    position = np.array([[y, x] for y in range(8) for x in range(8)])
    embeddings = np.zeros((max_len, d_model))

    for dim in range(d_model):
        div_term = np.exp((dim // 2) * -math.log(10000.0) / (d_model // 2))
        if dim % 2 == 0:
            # Apply sin to even indices in the dimensions
            embeddings[:, dim] = np.sin(position[:, dim % 2] * div_term)
        else:
            # Apply cos to odd indices in the dimensions
            embeddings[:, dim] = np.cos(position[:, dim % 2] * div_term)

    return embeddings


class ChessTransformerEmbeddings(nn.Module):

    SEQ_LEN = 64  # Number of positions on the chess board

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        freeze_pos: bool = False,
        add_global: bool = True,
    ) -> None:
        """
        :param vocab_size: The number of unique tokens in the input.
        :param d_model: The number of dimensions of the embeddings.
        :param freeze_pos: Whether to freeze the positional embeddings.
        :param add_global: Whether to add a global token in the input.
        """
        super(ChessTransformerEmbeddings, self).__init__()
        self.add_global = add_global
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
            self.vocab_size if not add_global else self.vocab_size + 1,
            d_model,
        )
        pos_embedding_tensor = torch.tensor(
            get_sinusoidal_embeddings(d_model), dtype=torch.float
        )
        if add_global:
            global_token = torch.zeros(1, d_model)
            pos_embedding_tensor = torch.cat([global_token, pos_embedding_tensor], dim=0)
        self.pos_encoder = nn.Embedding.from_pretrained(
            pos_embedding_tensor,
            freeze=freeze_pos,
        )
        pos_ids = torch.arange(
            self.SEQ_LEN if not add_global else self.SEQ_LEN + 1, dtype=torch.long
        ).unsqueeze(0)
        self.register_buffer("pos_ids", pos_ids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len)
        if self.add_global:
            # Add global token to the input
            global_token = torch.full_like(x[:, 0], self.vocab_size).unsqueeze(1)
            x = torch.cat([global_token, x], dim=1)

        # x: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.embedding(x) + self.pos_encoder(self.pos_ids)
        return x


class ChessTransformer(nn.Module):
    """
    Encoder-only transformer model for chess board evaluation. Deep-Q learning.
    """

    INPUT_DIM = 64  # Number of positions on the chess board
    OUTPUT_DIM = 64  # Number of spaces each piece could move to
    VOCAB_SIZE = 13  # Number of unique tokens in the input

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        freeze_pos: bool = True,
        add_global: bool = True,
    ):
        super(ChessTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.freeze_pos = freeze_pos
        self.add_global = add_global

        self.embedding = ChessTransformerEmbeddings(
            vocab_size=self.VOCAB_SIZE,
            d_model=d_model,
            freeze_pos=freeze_pos,
            add_global=add_global,
        )
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            )
            for _ in range(num_layers)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_dropout = nn.Dropout(dropout)
        self.policy_fc = nn.Linear(d_model, self.OUTPUT_DIM)
        self.auxiliary_fc = nn.Linear(d_model, self.OUTPUT_DIM)

    def get_hparams(self) -> dict:
        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "freeze_pos": self.freeze_pos,
        }

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        # (seq_len + 1 if add_global)
        x = self.embedding(x)

        for layer in self.layers:
            # x: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
            x = layer(x)

        x = self.layer_norm(x)
        x = self.fc_dropout(x)

        # Remove the global token prior to fc layers if it was added
        if self.add_global:
            x = x[:, 1:, :]

        # x: (batch_size, seq_len, d_model) -> (batch_size, seq_len, output_dim)
        policy = self.policy_fc(x)
        # policy: (batch_size, seq_len, output_dim) -> (batch_size, seq_len * output_dim)
        policy = policy.view(policy.size(0), -1)

        # x: (batch_size, seq_len, d_model) -> (batch_size, seq_len, output_dim)
        auxiliary = self.auxiliary_fc(x)
        # auxiliary: (batch_size, seq_len, output_dim) -> (batch_size, seq_len * output_dim)
        auxiliary = auxiliary.view(auxiliary.size(0), -1)
        return policy, auxiliary
