from typing import Optional, Callable
from utils import Environment
from typesMZ import VisitSoftmaxTemperatureFn, NetworkFactory, KnownBounds, Network

EnvironmentFactory = Callable[[], Environment]


class StochasticMuZeroConfig:
    # A factory for the environment.
    environment_factory: EnvironmentFactory
    network_factory: NetworkFactory
    # Self-Play
    num_actors: int
    visit_softmax_temperature_fn: VisitSoftmaxTemperatureFn
    num_simulations: int
    discount: float
    # Root prior exploration noise.
    root_dirichlet_alpha: float
    root_dirichlet_fraction: float
    root_dirichlet_adaptive: bool
    # UCB formula
    pb_c_base: float = 19652
    pb_c_init: float = 1.25
    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
    known_bounds: Optional[KnownBounds] = None
    # Replay buffer.
    num_trajectories_in_buffer: int = int(1e6)
    batch_size: int = int(128)
    num_unroll_steps: int = 5
    td_steps: int = 6
    td_lambda: float = 1.0
    # Alpha and beta parameters for prioritization.
    # By default they are set to 0 which means uniform sampling.
    priority_alpha: float = 0.0
    priority_beta: float = 0.0
    # Training
    training_steps: int = int(1e6)
    export_network_every: int = int(1e3)
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    # The number of chance codes (codebook size).
    # We use a codebook of size 32 for all our experiments.
    codebook_size: int = 32

def twentyfortyeight_config() -> StochasticMuZeroConfig:
    """Returns the config for the game of 2048."""
    def environment_factory():
        # Returns an implementation of 2048.
        return Environment()
    def network_factory():
        # 10 layer fully connected Res V2 network with Layer normalization and size
        # 256.
        return Network()
    
    def visit_softmax_temperature(train_steps: int) -> float:
        if train_steps < 1e5:
            return 1.0
        elif train_steps < 2e5:
            return 0.5
        elif train_steps < 3e5:
            return 0.1
        else:
        # Greedy selection.
            return 0.0
        
    return StochasticMuZeroConfig(
        environment_factory=environment_factory,
        network_factory=network_factory,
        num_actors=1000,
        visit_softmax_temperature=visit_softmax_temperature,
        num_simulations=100,
        discount=0.999,
        root_dirichlet_alpha=0.3,
        root_dirichlet_fraction=0.1,
        root_dirichlet_adaptive=False,
        num_trajectories_in_buffer=int(125e3),
        td_steps=10,
        td_lambda=0.5,
        priority_alpha=1.0,
        priority_beta=1.0,
        training_steps=int(20e6),
        batch_size=1024,
        weight_decay=0.0)