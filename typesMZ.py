from typing import Any, Dict, Callable, List, NamedTuple, Tuple, Union, Optional, Sequence
import torch
Player = int
LatentState = List[float]
AfterState = List[float]

Action = Any

Outcome = Any
ActionOrOutcome = Union[Action, Outcome]


VisitSoftmaxTemperatureFn = Callable[[int], float]
# EnvironmentFactory = Callable[[], Environment]
# NetworkFactory = Callable[[], Network]

class SearchStats(NamedTuple):
    search_policy: List[int]
    search_value: float
class State(NamedTuple):
    """Data for a single state."""
    observation: List[float]
    reward: float
    discount: float
    player: Player
    action: Action
    search_stats: SearchStats

Trajectory = Sequence[State]

class KnownBounds(NamedTuple):
    min: float
    max: float


class NetworkOutput(NamedTuple):
    value: torch.Tensor
    float_value: float
    probabilities: List[float]
    reward: torch.Tensor
    float_reward: float

