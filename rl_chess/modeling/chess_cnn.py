import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessCNN(nn.Module):
    def __init__(self, num_filters=256, num_residual_blocks=10):
        super(ChessCNN, self).__init__()

        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks

        self.input_conv = nn.Conv2d(12, num_filters, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_filters)

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 64, 4096)

    def forward(self, x):
        # x shape: (batch_size, 64)
        x = self.preprocess_input(x)  # shape: (batch_size, 12, 8, 8)

        x = F.relu(self.batch_norm(self.input_conv(x)))

        for block in self.residual_blocks:
            x = block(x)

        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        q_values = self.policy_fc(policy)

        return q_values

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


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
