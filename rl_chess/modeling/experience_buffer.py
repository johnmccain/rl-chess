import collections
import logging
import random
from dataclasses import dataclass, field

import chess
import torch

logger = logging.getLogger(__name__)


@dataclass
class FullEvaluationRecord:
    fen: str
    state: torch.Tensor
    legal_moves_mask: torch.Tensor
    rewards: torch.Tensor
    done: bool
    color: chess.Color | None = None
    move_count: int = 0  # move count (measured in half-turns)

    def make_serializeable(self) -> dict:
        """
        Make efficiently serializable by converting tensors to numpy arrays and converting to dictionary.
        """
        return {
            "fen": self.fen,
            "state": self.state.cpu().numpy(),
            "legal_moves_mask": self.legal_moves_mask.cpu().numpy(),  # (4096,) tensor
            "rewards": self.rewards.cpu().numpy(),  # (4096,) tensor
            "done": self.done,
            "color": self.color,
            "move_count": self.move_count,
        }

    @classmethod
    def from_serialized(cls, serialized: dict) -> "FullEvaluationRecord":
        """
        Load a serialized FullEvaluationRecord from a dictionary by converting numpy arrays back to tensors.
        """
        return cls(
            fen=serialized["fen"],
            state=torch.tensor(serialized["state"]),
            legal_moves_mask=torch.tensor(serialized["legal_moves_mask"]),
            rewards=torch.tensor(serialized["rewards"]),
            done=serialized["done"],
            color=serialized["color"],
            move_count=serialized["move_count"],
        )


@dataclass(order=True)
class ExperienceRecord:
    q_diff: float
    state: torch.Tensor = field(compare=False)
    legal_moves_mask: torch.Tensor = field(compare=False)
    action: torch.Tensor = field(compare=False)
    reward: float = field(compare=False)
    next_state: torch.Tensor = field(compare=False)
    next_legal_moves_mask: torch.Tensor = field(compare=False)
    done: bool = field(compare=False)
    opp_done: bool = field(compare=False)
    pred_q_values: torch.Tensor | None = field(default=None, compare=False)
    max_next_q: float | None = field(default=None, compare=False)
    color: chess.Color | None = None
    move_count: int = 0  # move count (measured in half-turns)

    def make_serializeable(self) -> dict:
        """
        Make efficiently serializable by converting tensors to numpy arrays and converting to dictionary.
        """
        return {
            "q_diff": self.q_diff,
            "state": self.state.cpu().numpy(),
            "legal_moves_mask": self.legal_moves_mask.cpu().numpy(),
            "action": self.action.cpu().numpy(),
            "reward": self.reward,
            "next_state": self.next_state.cpu().numpy(),
            "next_legal_moves_mask": self.next_legal_moves_mask.cpu().numpy(),
            "done": self.done,
            "opp_done": self.opp_done,
            "pred_q_values": (
                self.pred_q_values.cpu().numpy()
                if self.pred_q_values is not None
                else None
            ),
            "max_next_q": self.max_next_q,
            "color": self.color,
            "move_count": self.move_count,
        }

    @classmethod
    def from_serialized(cls, serialized: dict) -> "ExperienceRecord":
        """
        Load a serialized ExperienceRecord from a dictionary by converting numpy arrays back to tensors.
        """
        return cls(
            q_diff=serialized["q_diff"],
            state=torch.tensor(serialized["state"]),
            legal_moves_mask=torch.tensor(serialized["legal_moves_mask"]),
            action=torch.tensor(serialized["action"]),
            reward=serialized["reward"],
            next_state=torch.tensor(serialized["next_state"]),
            next_legal_moves_mask=torch.tensor(serialized["next_legal_moves_mask"]),
            done=serialized["done"],
            opp_done=serialized["opp_done"],
            pred_q_values=(
                torch.tensor(serialized["pred_q_values"])
                if serialized["pred_q_values"] is not None
                else None
            ),
            max_next_q=serialized["max_next_q"],
            color=serialized["color"],
            move_count=serialized["move_count"],
        )


class ExperienceBuffer:
    def __init__(self, window_size: int) -> None:
        self.buffer: collections.deque[ExperienceRecord] = collections.deque(
            maxlen=window_size
        )
        self.window_size = window_size

    def add(self, experience: ExperienceRecord) -> None:
        self.buffer.append(experience)

    def sample(self) -> ExperienceRecord:
        """
        Sample a random experience from the buffer and return it.
        """
        if not self.buffer:
            raise IndexError("sample from an empty buffer")
        return random.choice(self.buffer)

    def sample_n(self, n: int) -> list[ExperienceRecord]:
        """
        Sample n random experiences from the buffer without replacement and return them.
        """
        if len(self.buffer) < n:
            raise IndexError("sample from an empty buffer")
        return random.sample(self.buffer, n)

    def extend(self, iterable: list[ExperienceRecord]) -> None:
        self.buffer.extend(iterable)

    def __len__(self) -> int:
        return len(self.buffer)
