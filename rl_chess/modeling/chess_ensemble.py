import torch
from torch import nn
from rl_chess.modeling.chess_cnn import ChessCNN
from rl_chess.modeling.chess_transformer import ChessTransformer

class EnsembleCNNTransformer(nn.Module):

    def __init__(self, cnn: ChessCNN, transformer: ChessTransformer):
        super(EnsembleCNNTransformer, self).__init__()
        self.cnn = cnn
        self.transformer = transformer
        self.q_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.aux_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x: torch.Tensor, move_count: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, 64)
        # move_count shape: (batch_size,)
        cnn_q_hat, cnn_aux_logits = self.cnn(x, move_count)
        transformer_q_hat, transformer_aux_logits = self.transformer(x, move_count)

        q_hat = cnn_q_hat * self.q_weight + transformer_q_hat * (1 - self.q_weight)
        aux_logits = cnn_aux_logits * self.aux_weight + transformer_aux_logits * (
            1 - self.aux_weight
        )

        return q_hat, aux_logits
