import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoidal_embeddings(d_model: int) -> torch.Tensor:
    """
    Generate sinusoidal embeddings for each position on the chess board.

    :param d_model: The number of dimensions of the embeddings.

    :returns: A max_len x d_model tensor of sinusoidal embeddings.
    """
    max_len = 64  # Number of positions on the chess board
    position = torch.tensor(
        [[y, x] for y in range(8) for x in range(8)], dtype=torch.float
    )
    embeddings = torch.zeros(max_len, d_model)

    for dim in range(d_model):
        div_term = torch.exp(
            torch.tensor(dim // 2 * -math.log(10000.0) / (d_model // 2))
        )
        if dim % 2 == 0:
            # Apply sin to even indices in the dimensions
            embeddings[:, dim] = torch.sin(position[:, dim % 2] * div_term)
        else:
            # Apply cos to odd indices in the dimensions
            embeddings[:, dim] = torch.cos(position[:, dim % 2] * div_term)

    return embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=64):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = get_sinusoidal_embeddings(d_model)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ChessCNNTransformer(nn.Module):
    def __init__(
        self,
        num_filters=128,
        num_residual_blocks=5,
        d_model=128,
        nhead=4,
        num_transformer_layers=4,
        dim_feedforward=512,
        dropout=0.0,
    ):
        super(ChessCNNTransformer, self).__init__()

        # CNN layers
        self.input_conv = nn.Conv2d(12, num_filters, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.input_conv.weight, nonlinearity="relu")
        self.batch_norm = nn.BatchNorm2d(num_filters)

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )

        # Transformer layers
        self.transformer_input_linear = nn.Linear(num_filters, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, batch_first=True
                )
                for _ in range(num_transformer_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)

        # Policy and auxiliary heads
        self.policy_fc = nn.Linear(d_model, 64)
        self.auxiliary_fc = nn.Linear(d_model, 64)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, 64)
        x = self.preprocess_input(x)  # shape: (batch_size, 12, 8, 8)

        x = F.relu(self.batch_norm(self.input_conv(x)))

        for block in self.residual_blocks:
            x = block(x)

        # Prepare for transformer
        # x shape: (batch_size, num_filters, 8, 8)
        x = x.permute(0, 2, 3, 1)  # (batch_size, 8, 8, num_filters)
        x = x.view(x.size(0), -1, x.size(3))  # (batch_size, 64, num_filters)

        # Linear projection to transformer input dimension
        x = self.transformer_input_linear(x)  # (batch_size, 64, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        x = self.layer_norm(x)

        # Policy and auxiliary heads
        policy = self.policy_fc(x)  # (batch_size, 64, 64)
        policy = policy.view(policy.size(0), -1)  # (batch_size, 4096)

        auxiliary = self.auxiliary_fc(x)  # (batch_size, 64, 64)
        auxiliary = auxiliary.view(auxiliary.size(0), -1)  # (batch_size, 4096)

        return policy, auxiliary

    def preprocess_input(self, x):
        # x shape: (batch_size, 64)
        batch_size = x.size(0)
        x = x.long()  # Ensure input is long tensor

        # Create one-hot encoded tensor
        one_hot = torch.zeros(batch_size, 13, 64, device=x.device)
        one_hot.scatter_(1, x.unsqueeze(1), 1)

        # Reshape to 8x8 board
        one_hot = one_hot.view(batch_size, 13, 8, 8)

        # Remove the empty square channel (assuming 0 represents empty)
        return one_hot[:, 1:, :, :]
