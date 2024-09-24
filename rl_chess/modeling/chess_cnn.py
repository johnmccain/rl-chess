import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessCNN(nn.Module):

    def __init__(
        self,
        num_filters: int = 256,
        num_residual_blocks: int = 10,
        negative_slope: float = 0.01,
        dropout: float = 0.1,
        move_count_scale_factor: float = 0.05,
    ):
        super(ChessCNN, self).__init__()

        self.move_count_scale_factor = move_count_scale_factor
        self.negative_slope = negative_slope
        self.activation = (
            nn.LeakyReLU(negative_slope=self.negative_slope)
            if negative_slope > 0
            else nn.ReLU()
        )
        self.activation_str = "leaky_relu" if negative_slope > 0 else "relu"

        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks

        # 13 for 12 piece types + 1 for move count
        self.input_conv = nn.Conv2d(13, num_filters, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(
            self.input_conv.weight,
            nonlinearity=self.activation_str,
            a=self.negative_slope,
        )
        self.batch_norm = nn.BatchNorm2d(num_filters)

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(num_filters, negative_slope=self.negative_slope)
                for _ in range(num_residual_blocks)
            ]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        nn.init.kaiming_normal_(
            self.policy_conv.weight,
            nonlinearity=self.activation_str,
            a=self.negative_slope,
        )
        self.policy_fc = nn.Linear(2 * 64, 4096)

        # Auxiliary head (move legality)
        self.auxiliary_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        nn.init.kaiming_normal_(
            self.auxiliary_conv.weight,
            nonlinearity=self.activation_str,
            a=self.negative_slope,
        )
        self.auxiliary_fc = nn.Linear(2 * 64, 4096)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, move_count: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, 64)
        # move_count shape: (batch_size, 1)
        x = self.preprocess_input(x, move_count)  # shape: (batch_size, 12, 8, 8)

        x = self.activation(self.batch_norm(self.input_conv(x)))

        for block in self.residual_blocks:
            x = block(x)
            x = self.dropout(x)

        policy = self.policy_conv(x)
        policy = policy.view(policy.size(0), -1)
        policy = self.dropout(policy)
        q_values = self.policy_fc(policy)

        auxiliary = self.auxiliary_conv(x)
        auxiliary = auxiliary.view(auxiliary.size(0), -1)
        auxiliary = self.dropout(auxiliary)
        auxiliary_logits = self.auxiliary_fc(auxiliary)

        return q_values, auxiliary_logits

    def preprocess_input(self, x: torch.Tensor, move_count: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 64)
        # move_count shape: (batch_size,)
        x = x.long()

        # Create one-hot encoded tensor
        one_hot = torch.zeros_like(x.unsqueeze(1).expand(-1, 13, -1))
        one_hot.scatter_(1, x.unsqueeze(1), 1)

        # Reshape to 8x8 board
        one_hot = one_hot.view(-1, 13, 8, 8)
        one_hot = one_hot[:, 1:, :, :]  # Remove the empty square channel

        # Reshape and expand move_count
        move_count = move_count.view(-1, 1, 1, 1).expand(-1, 1, 8, 8)
        move_count = move_count * self.move_count_scale_factor

        # Concatenate move count to the input
        one_hot = torch.cat([one_hot, move_count], dim=1)  # shape: (batch_size, 13, 8, 8)

        return one_hot


class ResidualBlock(nn.Module):
    def __init__(self, num_filters, negative_slope=0.01):
        super(ResidualBlock, self).__init__()
        self.negative_slope = negative_slope
        self.activation = (
            nn.LeakyReLU(negative_slope=self.negative_slope)
            if negative_slope > 0
            else nn.ReLU()
        )
        self.activation_str = "leaky_relu" if negative_slope > 0 else "relu"

        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(
            self.conv1.weight, nonlinearity=self.activation_str, a=self.negative_slope
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(
            self.conv2.weight, nonlinearity=self.activation_str, a=self.negative_slope
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(
            self.bn1(self.conv1(x)),
        )
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.activation(
            x,
        )
        return x
