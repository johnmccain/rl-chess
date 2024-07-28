import random

import chess
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from rl_chess.modeling.chess_transformer import ChessTransformer
from rl_chess.modeling.utils import (
    board_to_tensor,
    evaluate_fitness,
    get_legal_moves_mask,
    index_to_move,
)


def calculate_reward(board: chess.Board, move: chess.Move) -> float:
    player = board.turn
    board.push(move)
    reward = evaluate_fitness(board, player)
    board.pop()
    return reward


def train_deep_q_network(
    model: nn.Module,
    episodes: int,
    gamma: float = 0.99,
    lr: float = 0.001,
    decay: float = 0.995,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    epsilon = 1.0

    for episode in range(episodes):
        board = chess.Board()
        total_loss = 0

        while not board.is_game_over():
            current_state = board_to_tensor(board, board.turn)
            current_state = current_state.unsqueeze(0)  # Batch size of 1

            # Predict Q-values
            predicted_q_values = model(current_state)

            # Mask illegal moves
            legal_moves_mask = get_legal_moves_mask(board)
            masked_q_values = predicted_q_values.masked_fill(
                legal_moves_mask == 0, float("-inf")
            )

            # Epsilon-greedy action selection
            if random.random() > epsilon:
                action = masked_q_values.max(1)[1].view(1, 1)
            else:
                # Select action randomly with softmax
                action = torch.multinomial(F.softmax(masked_q_values, dim=-1), 1)

            # Take action and observe reward and next state
            move = index_to_move(action, board)  # Convert action index to a chess move
            reward = calculate_reward(
                board, move
            )  # Implement this based on your reward system

            board.push(move)
            next_state = board_to_tensor(board, board.turn)
            next_state = next_state.unsqueeze(0)

            done = torch.tensor([int(board.is_game_over())])

            # Predict next Q-values
            next_q_values = model(next_state)
            max_next_q_values = next_q_values.max(1)[0].detach()

            # Compute the target Q-value
            target_q_values = reward + (gamma * max_next_q_values * (1 - done))

            # Compute loss
            loss = loss_fn(
                predicted_q_values.gather(1, action), target_q_values.unsqueeze(1)
            )
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, Loss: {total_loss}")
            epsilon = max(epsilon * decay, 0.01)  # Decay epsilon

    print("Training complete")


if __name__ == "__main__":
    model = ChessTransformer(
        vocab_size=13,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    )
    train_deep_q_network(model, episodes=5000)
