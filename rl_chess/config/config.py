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
    APP_MODEL_NAME = config["app.model_name"]

    LOG_LEVEL: str = config["log.level"]

    MODEL_LR: float = config["model.lr"]
    MODEL_GAMMA_RAMP_STEPS: int = config["model.gamma_ramp_steps"]
    MODEL_GAMMA: float = config["model.gamma"]
    MODEL_INITIAL_GAMMA: float = config["model.initial_gamma"]
    MODEL_GRAD_STEPS: int = config["model.grad_steps"]
    MODEL_DECAY: float = config["model.decay"]
    MODEL_CLIP_GRAD: float = config["model.clip_grad"]
