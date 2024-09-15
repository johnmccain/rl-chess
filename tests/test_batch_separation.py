import torch
import torch.nn as nn
from rl_chess.modeling.chess_transformer import ChessTransformer
from rl_chess.modeling.chess_cnn import ChessCNN
from rl_chess.modeling.chess_cnn_transformer import ChessCNNTransformer


def test_batch_separation(
    model: nn.Module, seq_len: int = 64, vocab_size: int = 13, batch_size: int = 8, trials=64,
):
    """
    Test that the model's output for the first example in a batch remains the same when more random
    examples are added to the batch.

    :param model: The model to test.
    :param seq_len: The length of the input sequence (64 for a chessboard).
    :param vocab_size: The number of unique piece types (13 for chess).
    :param batch_size: The size of the batch to test.
    :param trials: The number of trials to run.
    """
    # Set model to evaluation mode
    model.eval()

    num_passed = 0
    max_diff = 0.0

    for _ in range(trials):
        # Generate a single random board state (a single input example)
        first_example = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.int32)

        # Pass this single example through the model and record the output
        with torch.no_grad():
            single_output = model(first_example)[
                0
            ]  # First element of the tuple (policy prediction)

        # Generate random examples for the rest of the batch
        additional_examples = torch.randint(
            0, vocab_size, (batch_size - 1, seq_len), dtype=torch.int32
        )

        # Combine the first example with the additional random examples to form a batch
        batch_input = torch.cat(
            [first_example, additional_examples], dim=0
        )  # (batch_size, seq_len)

        # Pass the batch through the model
        with torch.no_grad():
            batch_output = model(batch_input)[
                0
            ]  # First element of the tuple (policy prediction)

        # Extract the output corresponding to the first example in the batch
        first_in_batch_output = batch_output[0]  # Output of the first example in the batch

        # Compare the outputs
        if torch.allclose(single_output, first_in_batch_output, atol=1e-5):
            num_passed += 1
        else:
            max_diff = max(max_diff, torch.abs(single_output - first_in_batch_output).max().item())

    if num_passed == trials:
        print(
            f"Test passed: The output for the first example is identical in {num_passed}/{trials} trials."
        )
    else:
        max_diff = max(
            max_diff, torch.abs(single_output - first_in_batch_output).max().item()
        )
        print(
            f"Test failed: The output for the first example changed when additional examples were added in in {trials - num_passed}/{trials} trials."
        )
        print(
            "Biggest difference:",
            max_diff,
        )


if __name__ == "__main__":
    print("Testing batch separation for CNN model:")
    # Create a random CNN model
    cnn_model = ChessCNN(
        num_filters=256,
        num_residual_blocks=4,
        negative_slope=0.01,
    )

    test_batch_separation(cnn_model)

    print("Testing batch separation for Transformer model:")
    # Create a random transformer model
    transformer_model = ChessTransformer(
        d_model=256,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.0,
    )

    test_batch_separation(transformer_model)

    print("Testing batch separation for CNN-Transformer model:")
    # Create a random CNN-Transformer model
    cnn_transformer_model = ChessCNNTransformer(
        num_filters=128,
        num_residual_blocks=2,
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
        dim_feedforward=512,
        dropout=0.0,
    )

    test_batch_separation(cnn_transformer_model)
