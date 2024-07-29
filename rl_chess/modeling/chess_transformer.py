import math

import chess
import numpy as np
import torch
from torch import nn


def get_sinusoidal_embeddings(d_model: int, max_len: int = 64) -> np.ndarray:
    """
    Generate sinusoidal embeddings for each position on the chess board.

    :param d_model: The number of dimensions of the embeddings.
    :param max_len: The total number of positions (default 64 for 8x8 chessboard).

    :returns: A max_len x d_model array of sinusoidal embeddings.
    """
    assert max_len == 64  # Ensure it's a standard chess board
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

    def __init__(self, vocab_size: int, d_model: int, freeze_pos: bool = True):
        """
        :param vocab_size: The number of unique tokens in the input.
        :param d_model: The number of dimensions of the embeddings.
        :param freeze_pos: Whether to freeze the positional embeddings.
        """
        super(ChessTransformerEmbeddings, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_encoder = nn.Embedding.from_pretrained(
            torch.tensor(
                get_sinusoidal_embeddings(d_model, self.SEQ_LEN), dtype=torch.float
            ),
            freeze=freeze_pos,
        )
        pos_ids = torch.arange(self.SEQ_LEN).unsqueeze(0)
        self.register_buffer("pos_ids", pos_ids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    ):
        super(ChessTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.freeze_pos = freeze_pos

        self.embedding = ChessTransformerEmbeddings(
            vocab_size=self.VOCAB_SIZE, d_model=d_model, freeze_pos=freeze_pos
        )
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        )
        self.output_layer = nn.Linear(d_model, self.OUTPUT_DIM)

    def get_hparams(self) -> dict:
        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "freeze_pos": self.freeze_pos,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len) -> (seq_len, batch_size, d_model)
        x = self.embedding(x).permute(1, 0, 2)

        for layer in self.layers:
            # x: (seq_len, batch_size, d_model) -> (seq_len, batch_size, d_model)
            x = layer(x)

        # x: (seq_len, batch_size, d_model) -> (seq_len, batch_size, output_dim)
        x = self.output_layer(x.permute(1, 0, 2))
        # x: (seq_len, batch_size, output_dim) -> (batch_size, seq_len * output_dim)
        x = x.view(x.size(0), -1)
        return x
