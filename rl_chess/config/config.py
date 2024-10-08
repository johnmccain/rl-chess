import os
import pathlib

from dynaconf import Dynaconf
from pydantic import BaseModel

current_dir = pathlib.Path(__file__).parent

# Load the environment from the `ENV` environment variable, defaulting to `local`.
env = os.environ.get("ENV", "local")

config = Dynaconf(
    envvar_prefix="DYNACONF",  # `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
    settings_files=[
        current_dir / "default.toml",
        current_dir / f"{env}.toml",
        current_dir / ".secrets.toml",
    ],
    merge_enabled=True,
)


class AppConfig(BaseModel):
    APP_OUTPUT_DIR: str = config["app.output_dir"]
    APP_TENSORBOARD_DIR: str = config["app.tensorboard_dir"]
    APP_SAVE_STEPS: int = config["app.save_steps"]
    APP_EVAL_STEPS: int = config["app.eval_steps"]
    APP_MODEL_NAME: str = config["app.model_name"]
    APP_MOVE_EVAL_DATASET: str = config["app.move_eval_dataset"]

    LOG_LEVEL: str = config["log.level"]

    MODEL_CLASS: str = config["model.class"]

    MODEL_LR: float = config["model.lr"]
    MODEL_WARMUP_STEPS: int = config["model.warmup_steps"]
    MODEL_GAMMA_STARTUP_STEPS: int = config["model.gamma_startup_steps"]
    MODEL_GAMMA_RAMP_STEPS: int = config["model.gamma_ramp_steps"]
    MODEL_GAMMA: float = config["model.gamma"]
    MODEL_INITIAL_GAMMA: float = config["model.initial_gamma"]
    MODEL_GRAD_STEPS: int = config["model.grad_steps"]
    MODEL_DECAY: float = config["model.decay"]
    MODEL_MIN_EPSILON: float = config["model.min_epsilon"]
    MODEL_CLIP_GRAD: float = config["model.clip_grad"]
    MODEL_MAX_MOVES: int = config["model.max_moves"]
    MODEL_BUFFER_SIZE: int = config["model.buffer_size"]
    MODEL_BATCH_SIZE: int = config["model.batch_size"]
    MODEL_EXPLORE_EPISODES: int = config["model.explore_episodes"]
    MODEL_LEARN_STEPS: int = config["model.learn_steps"]
    MODEL_TARGET_UPDATE_FREQ: int = config["model.target_update_freq"]

    MODEL_CNN_NUM_FILTERS: int = config["model.cnn.num_filters"]
    MODEL_CNN_RESIDUAL_BLOCKS: int = config["model.cnn.residual_blocks"]
    MODEL_CNN_NEGATIVE_SLOPE: float = config["model.cnn.negative_slope"]
    MODEL_CNN_DROPOUT: float = config["model.cnn.dropout"]

    MODEL_TRANSFORMER_NUM_HEADS: int = config["model.transformer.num_heads"]
    MODEL_TRANSFORMER_NUM_LAYERS: int = config["model.transformer.num_layers"]
    MODEL_TRANSFORMER_D_MODEL: int = config["model.transformer.d_model"]
    MODEL_TRANSFORMER_DIM_FEEDFORWARD: int = config["model.transformer.dim_feedforward"]
    MODEL_TRANSFORMER_DROPOUT: float = config["model.transformer.dropout"]
    MODEL_TRANSFORMER_FREEZE_POS: bool = config["model.transformer.freeze_pos"]
    MODEL_TRANSFORMER_ADD_GLOBAL: bool = config["model.transformer.add_global"]

    STOCKFISH_PATH: str = config["stockfish.path"]
    STOCKFISH_PROB: float = config["stockfish.prob"]
    STOCKFISH_DEPTH: int = config["stockfish.depth"]
    STOCKFISH_BLUNDER_THRESHOLD: int = config["stockfish.blunder_threshold"]
