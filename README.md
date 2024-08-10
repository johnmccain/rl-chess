# RL Chess

WIP chess agent using reinforcement learning & self play.

## Setup

This project requires git-lfs for storing some data files.

This project uses poetry for dependency management and requires `python>=3.11`.

```bash
poetry install
```

You will also need a stockfish binary for evaluation. You can download stockfish for your system [here](https://stockfishchess.org/download/).

When it is downloaded, you should update the `[STOCKFISH] PATH` config option with the path to your binary.

This project uses tensorboard to track training runs; runs are stored in the `tensorboard` directory.

Pre commit hooks can be installed with `pre-commit install`.

## Configuration

Configuration is managed via Dynaconf and TOML files. `rl_chess/config/default.toml` contains the primary configuration settings.

Environment specific configuration files can be created (ex: `local.toml`) which will be merged with the default.toml and overwrite values selectively. The environment specific config file to load is controlled via the `ENV` environment variable.

Config settings can also be overridden by environment variable using the structure:
**`DYNACONF_{TABLE}_{KEY}`**

## Usage

### Inference

This project has a simple pygame interface for playing against the model. You can run it with:
```bash
python -m rl_chess.main
```
The model filename should be updated in the default.toml according to which checkpoint you wish to play.

### Training

Training duration is measured in *episodes* = number of games played. Concurrent games due to batch size > 1 are counted towards episode count.

```bash
usage: python -m rl_chess.modeling.train_cnn [-h] [--resume] [--timestamp TIMESTAMP] [--episodes EPISODES]

options:
  -h, --help            show this help message and exit
  --resume              Resume training from the latest checkpoint
  --timestamp TIMESTAMP
                        Timestamp of the model to resume training from. Optional; --resume without
                        timestamp will resume from the latest checkpoint.
  --episodes EPISODES   Number of episodes to train for
```

## Methodology

The exact implementation is changing as I experiment, but currently the primary method I'm using is off-policy deep-Q learning with a convolutional network.

The state space is defined as a 8x8 int tensor with ids ranging from 0-12 for each

### CNN

The CNN converts the 8x8 piece id tensor to a 8x8x12 one-hot encoded tensor. The hidden layers of the CNN are a conventional residual network. The action head output of the CNN is a 4096 length float tensor predicting the Q value--this represents picking up a piece on some square and setting down a piece on another square (flattened 8x8x8x8). The CNN also has an auxiliary task head to classify move legality (also a 4096 length tensor). When selecting actions, the output q values must be masked based to remove illegal moves from consideration.

En passant square and castling rights are not currently included in the input representation, though the square-to-square nature of the output space means such moves can still be selected. Underpromotion of pawns is also not included in the output representation, and it is assumed that all pawns are promoted to queens.

### Transformer

The transformer network has received less focus, but the architecture is an encoder-only transformer processing inputs of length 64. The piece ids serve as the "token id" and are embedded with random initialization, and positional embeddings are sinusoidal. The transformer has a similar output/action space to the CNN.

### Training

The training uses off-policy learning with an experience replay buffer. The model goes through phases of exploration where new experiences are added to the buffer during self play and learning where the model predictions are updated. Due to the adversarial nature of chess, the "next state" is the next *player* move, so there is both a selected player move and an unseen opponent move taken between the "current" and "next" states.

Experiences are stored including the board state $s$, the reward $r$, the next board state $s'$, legal move masks for both current and next state $l_n$, $l'$, and flags for completed games with no further legal moves $d_n$ $d'$. The experience replay buffer is randomly sampled during the learning phases (prioritized experience replay is on the list of things to try). The experience replay buffer has a capped size that ejects the least recently observed moves.

Actions are selected using an $\epsilon$-greedy strategy. A decaying value $\epsilon \in [0, 1]$ controls the likelihood of selecting a random move for exploration vs selecting the top rated move based on $\hat{Q}$. Random move selection uses weighted random sampling, so $\hat{Q}$ still has impact on those move selections. Epsilon exponentially decays from a high initial value to a minimum based on a decay hyperparameter.

Rewards are calculated using a fairly simple calculation that includes a material score using conventional piece values normalized by the total material on the board in addition to a bias towards controlling the center board. Checkmates are rewarded with $\pm1$ and stalemates have a reward of $0$.

Target Q-values are computed as: $Q(s, a) = r + \gamma * Q(s', a')$. The $\gamma$ term controls the discounting of future rewards.

$\gamma$ changes on a schedule, starting with a number of warmup episodes with 0 gamma and then a linear ramp up to the target gamma value. The intuition here is that chess is a difficult game to learn, so preparing the network by learning the simple reward function first may improve training stability.

$Q(s', a')$ is predicted using a separate target model which is updated with the policy model weights on a schedule. This mitigates the "moving target" problem of the policy network needing to account for its own future predictions as those change.

### Evaluation

During training, evaluation takes place using
1. Sampling from the experience buffer for move legality classification
2. Single-move evaluation using Stockfish for move quality

The move quality evaluation uses a dataset of board states taken from a stratified random sample of chess games. There are an equal number of board state + move pairs for early game, midgame, endgame, player advantage, player disadvantage, and checkmate. The `scripts/build_fen_dataset.py` script can be used to build such a dataset from a PGN file and store it as a csv including the board FEN, move UCI, and which situation it was sampled from.

Move quality metrics include:
- Average move quality (difference in stockfish evaluation before/after the move)
- Blunder rate (rate of moves resulting in decrease in evaluation beyond a configured level)
- Best move rate (rate of moves matching stockfish's selection)

Stockfish is set to a limited depth for this evaluation for efficiency. There is also a script `scripts/run_move_evaluation.py` that runs the same move evaluation outside of a training run.

#### Stockfish ELO Estimation

There is a basic elo estimation script using stockfish `scripts/run_stockfish_eval.py`. It pits the model against stockfish for a number of games at different ELO levels and finds the ELO that approximates a 50% win rate.

#### MiniMax Evaluation

`scripts/run_minimax_eval.py` pits the model against a simple minimax chess agent with variable depth, similarly to the Stockfish ELO estimation.
