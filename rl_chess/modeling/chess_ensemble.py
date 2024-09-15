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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cnn_q_hat, cnn_aux_logits = self.cnn(x)
        transformer_q_hat, transformer_aux_logits = self.transformer(x)

        q_hat = cnn_q_hat * self.q_weight + transformer_q_hat * (1 - self.q_weight)
        aux_logits = cnn_aux_logits * self.aux_weight + transformer_aux_logits * (
            1 - self.aux_weight
        )

        return q_hat, aux_logits
