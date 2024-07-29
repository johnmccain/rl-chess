import logging

from rl_chess.config.config import AppConfig
from rl_chess.interface.game_interface import GameRunner

app_config = AppConfig()

logging.basicConfig(
    level=app_config.LOG_LEVEL,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
)

if __name__ == "__main__":

    runner = GameRunner()

    runner.run()
