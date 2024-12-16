from typing import Any, Dict, Callable, List, NamedTuple, Tuple, Union, Optional, Sequence
from agents import Network
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
    search_policy: Dict[Action, int]
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

NetworkFactory = Callable[[], Network]

class NetworkOutput(NamedTuple):
    value: float
    probabilities: Dict[ActionOrOutcome, float]
    reward: Optional[float] = 0.0

