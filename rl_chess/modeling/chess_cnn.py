import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessCNN(nn.Module):
    def __init__(self, num_filters=256, num_residual_blocks=10, negative_slope=0.01):
        super(ChessCNN, self).__init__()

        self.negative_slope = negative_slope

        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks

        self.input_conv = nn.Conv2d(13, num_filters, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(
            self.input_conv.weight, nonlinearity="leaky_relu", a=self.negative_slope
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
            self.policy_conv.weight, nonlinearity="leaky_relu", a=self.negative_slope
        )
        self.policy_fc = nn.Linear(2 * 64, 4096)

        # Auxiliary head (move legality)
        self.auxiliary_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        nn.init.kaiming_normal_(
            self.auxiliary_conv.weight, nonlinearity="leaky_relu", a=self.negative_slope
        )
        self.auxiliary_fc = nn.Linear(2 * 64, 4096)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, 64)
        x = self.preprocess_input(x)  # shape: (batch_size, 12, 8, 8)

        x = F.leaky_relu(self.batch_norm(self.input_conv(x)))

        for block in self.residual_blocks:
            x = block(x)

        policy = self.policy_conv(x)
        policy = policy.view(policy.size(0), -1)
        q_values = self.policy_fc(policy)

        auxiliary = self.auxiliary_conv(x)
        auxiliary = auxiliary.view(auxiliary.size(0), -1)
        auxiliary_logits = self.auxiliary_fc(auxiliary)

        return q_values, auxiliary_logits

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 64)
        batch_size = x.size(0)
        x = x.long()  # Ensure input is long tensor

        # Create one-hot encoded tensor
        one_hot = torch.zeros(batch_size, 13, 64, device=x.device)
        one_hot.scatter_(1, x.unsqueeze(1), 1)

        # Reshape to 8x8 board
        one_hot = one_hot.view(batch_size, 13, 8, 8)

        # Remove the empty square channel (assuming 0 represents empty)
        return one_hot[:, :, :, :]


class ResidualBlock(nn.Module):
    def __init__(self, num_filters, negative_slope=0.01):
        super(ResidualBlock, self).__init__()
        self.negative_slope = negative_slope
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(
            self.conv1.weight, nonlinearity="leaky_relu", a=self.negative_slope
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(
            self.conv2.weight, nonlinearity="leaky_relu", a=self.negative_slope
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.leaky_relu(
            self.bn1(self.conv1(x)),
            negative_slope=self.negative_slope,
        )
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.leaky_relu(
            x,
            negative_slope=self.negative_slope,
        )
        return x
