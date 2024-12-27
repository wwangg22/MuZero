from typing import Optional, List, Sequence, Tuple, Union, Any, Callable
from typesMZ import Trajectory, KnownBounds, VisitSoftmaxTemperatureFn, LatentState, NetworkOutput, AfterState
import numpy as np
from encoders import VQCodebook, ResNetV2Tower
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAXIMUM_FLOAT_VALUE = float('inf')

Player = int

Action= Any

Outcome = Any

ActionOrOutcome = Union[Action, Outcome]

    

class Network(nn.Module):
    """An instance of the network used by stochastic MuZero."""
    def __init__(self, observation_dim=496, hidden_dim=256, codebook_size=32, action_space=10, outcome_space=32):
        """
        Args:
            observation_dim: dimension of the flattened/processed observation input.
            hidden_dim: the dimension of the latent representations (256 as specified).
            codebook_size: size of the VQ codebook (32).
            action_space: number of possible actions (assumption).
            outcome_space: number of possible outcomes (matching codebook size).
        """
        super().__init__()
        
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.outcome_space = outcome_space

        # A linear layer to map from observation to an initial hidden state (256-dim)
        self.obs_to_hidden = nn.Linear(observation_dim, hidden_dim).to(device)

        # 10-block ResNet v2 tower
        self.resnet_tower_rep = ResNetV2Tower(input_dim=hidden_dim, num_blocks=10).to(device)
        self.resnet_tower_af_dyn = ResNetV2Tower(input_dim=hidden_dim, num_blocks=10).to(device)
        self.resnet_tower_dyn = ResNetV2Tower(input_dim=hidden_dim, num_blocks=10).to(device)



        # Heads for predictions: from latent state -> value and policy
        # We assume predictions from a latent state include:
        # - value: scalar
        # - policy: probabilities over actions
        self.value_head = nn.Linear(hidden_dim, 601).to(device)
        self.af_value_head = nn.Linear(hidden_dim, 601).to(device)
        self.policy_head = nn.Linear(hidden_dim, self.action_space).to(device)
        self.afterstate_pred = nn.Linear(hidden_dim, 32).to(device)

        # For afterstate predictions, we reuse the same tower and just produce a policy and value (no reward).
        # We can reuse value_head and policy_head with a different forward pass, 
        # or create separate heads. Let's reuse for simplicity.

        # Afterstate dynamics:
        # given a latent state and an action, produce an afterstate.
        # We'll model this as a simple MLP:
        self.afterstate_fc = nn.Linear(hidden_dim + self.action_space, hidden_dim).to(device)

        # Dynamics:
        # given afterstate and outcome, produce next latent state.
        # Similarly, an MLP:
        self.dynamics_fc = nn.Linear(hidden_dim + self.outcome_space, hidden_dim).to(device)

        # Encoder:
        # maps observation to outcome distribution (logits) to be quantized.
        # The snippet suggests encoder producing codebook indices. We'll do a simple linear:
        self.encoder_fc = nn.Linear(observation_dim, codebook_size).to(device)
        self.codebook = VQCodebook(M=codebook_size)

        # Reward head (for state transitions):
        # The snippet suggests that predictions for a latent state may include a reward.
        # The original code returns a reward in NetworkOutput.
        # We'll add a reward head from latent state for completeness:
        self.reward_head = nn.Linear(hidden_dim, 601).to(device)


    def representation(self, observation) -> LatentState:
        """Representation function maps from observation to latent state.
        Here we assume 'observation' is a tensor of shape (batch, observation_dim).
        """
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        observation = observation.to(device)
        # Map observation to hidden dim
        x = self.obs_to_hidden(observation)
        # Pass through ResNet tower
        x = self.resnet_tower_rep(x)
        # x is now a latent state
        return x

    def predictions(self, state: torch.Tensor) -> 'NetworkOutput':
        """Returns the network predictions for a latent state: value, policy, reward.
        returns an output of 601
        """
        state = state.to(device)
        value = self.value_head(state)
        value_prob = F.softmax(value, dim = -1)
        policy_logits = self.policy_head(state)
        policy_probs = F.softmax(policy_logits, dim=-1)
        # For simplicity, pick an example reward from the reward head:
        reward = self.reward_head(state)
        reward_prob = F.softmax(reward, dim=-1)
        # if state.dim() == 1:
        #     probabilities = {a: policy_probs[a].item() for a in range(self.action_space)}
        # else:
        # # Convert probabilities to a dict {action: prob}
        # # Assuming action space is integer from 0 to action_space-1
        #     probabilities = {a: policy_probs[0, a].item() for a in range(self.action_space)}
        support = torch.arange(0, 601).to(device)
        expected_value = torch.sum(value_prob * support, dim=-1)  # shape: (batch_size,)
        expected_reward = torch.sum(reward_prob * support, dim=-1)
        if expected_reward.dim() == 0:
            float_reward  = expected_reward.item()
            float_value = expected_value.item()
        else:
            float_reward = expected_reward
            float_value=expected_value
        # print("Policy probs", policy_probs)

        return NetworkOutput(value=value_prob,float_value=float_value, probabilities=policy_probs, reward=reward_prob, float_reward=float_reward)

    def afterstate_dynamics(self, state: LatentState, action: Action) -> AfterState:
        """Implements the dynamics from latent state and action to afterstate.
        We one-hot encode the action and concatenate with state, then process through a linear layer + tower.
        """
        # print("action", action)
        # if not isinstance(action, int):
        #     # In real code, handle different action types.
        #     # Assume action is int representing action index.
        #     action = int(action)
        if isinstance(action, int):
            action_one_hot = F.one_hot(torch.tensor([action]), num_classes=self.action_space).float()
        else:
            action_one_hot = F.one_hot(action, num_classes=self.action_space).float()
        action_one_hot = action_one_hot.to(device)
        # Expand state to batch dimension if needed
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # (1, hidden_dim)
        state = state.to(device)
        # Concatenate
        x = torch.cat([state, action_one_hot], dim=-1)
        x = self.afterstate_fc(x)
        x = self.resnet_tower_af_dyn(x)  
        return x

    def afterstate_predictions(self, state: AfterState) -> 'NetworkOutput':
        """Returns the network predictions for an afterstate.
        Afterstates differ from latent states in that we do not produce a reward here (assumption from snippet).
        """
        state = state.to(device)
        value = self.af_value_head(state)
        value_prob = F.softmax(value, dim = -1)
        policy_logits = self.afterstate_pred(state)
        policy_probs = F.softmax(policy_logits, dim=-1)
        # if state.dim() == 1:
        #     probabilities = {a: policy_probs[a].item() for a in range(self.action_space)}
        # else:
        #     probabilities = {a: policy_probs[0, a].item() for a in range(self.action_space)}

        support = torch.arange(0, 601).to(device)
        expected_value = torch.sum(value_prob * support, dim=-1)  # shape: (batch_size,)
        if expected_value.dim() == 0:
            float_value = expected_value.item()
        else:
            float_value=expected_value

        # No reward for afterstate transitions as per snippet comment.
        return NetworkOutput(value=value_prob, float_value=float_value, probabilities=policy_probs, reward=0.0, float_reward = 0.0)

    def dynamics(self, state: AfterState, outcome: Outcome) -> LatentState:
        """Implements the dynamics from afterstate and chance outcome to state.
        We one-hot encode the outcome (from the codebook) and combine with afterstate.
        """
        # print("outcome ", outcome)
        # if not isinstance(outcome, int):
        #     # Assume outcome is int index from 0 to outcome_space-1
        #     outcome = int(outcome)
        if not isinstance(outcome, torch.Tensor):
            outcome = int(outcome)
            outcome_one_hot = F.one_hot(torch.tensor([outcome]), num_classes=self.outcome_space).float()
        else:
            # outcome_one_hot = F.one_hot(outcome, num_classes=self.outcome_space).float()
            outcome_one_hot = outcome
        outcome_one_hot = outcome_one_hot.to(device)
        # ensure state is batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        state = state.to(device)
        x = torch.cat([state, outcome_one_hot], dim=-1)
        x = self.dynamics_fc(x)
        x = self.resnet_tower_dyn(x)
        return x

    def encoder(self, observation) -> Outcome:
        """An encoder maps an observation to an outcome.
        Using the VQ-VAE codebook approach, we:
          1) Compute logits from encoder_fc.
          2) Use argmax or Gumbel-softmax to pick a code from the codebook.
        """
        observation = observation.to(device)
         # Pass observations through the encoder
        logits = self.encoder_fc(observation)  # shape: (batch_size, codebook_size)

        # Use codebook (VQ) to get one-hot codes
        code_one_hot = self.codebook(logits, use_gumbel=False)  # (batch_size, codebook_size)

        # For each element in the batch, pick the index of the max one-hot dimension
        # outcome_idx = torch.argmax(code_one_hot, dim=-1)  # (batch_size,)

        return code_one_hot

class NetworkCacher:
    """An object to share the network between the self-play and training
    jobs.
    """
    def __init__(self):
        self._networks = {}
    def save_network(self, step: int, network: Network):
        self._networks[step] = network
    def load_network(self) -> Tuple[int, Network]:
        training_step = max(self._networks.keys())
        return training_step, self._networks[training_step]
    
NetworkFactory = Callable[[], Network]

class Environment:
    """Implements the rules of the environment.
    """
    def apply(self, action: Action):
        """Applies an action or a chance outcome to the environment.
        """
    def observation(self):
        """Returns the observation of the environment to feed to the network.
        """
    def is_terminal(self) -> bool:
        """Returns true if the environment is in a terminal state.
        return False
        """
    def legal_actions(self) -> Sequence[Action]:
        """Returns the legal actions for the current state.
        return []
        """
    def reward(self, player: Player) -> float:
        """Returns the last reward for the player.
        return 0.0
        """
    def to_play(self) -> Player:
        """Returns the current player to play.
        return 0"""


EnvironmentFactory = Callable[[], Environment]

class StochasticMuZeroConfig:
    def __init__(
        self,
        environment_factory: EnvironmentFactory,
        network_factory: NetworkFactory,
        num_actors: int,
        visit_softmax_temperature_fn: VisitSoftmaxTemperatureFn,
        num_simulations: int,
        discount: float,
        root_dirichlet_alpha: float,
        root_dirichlet_fraction: float,
        root_dirichlet_adaptive: bool,
        pb_c_base: float = 19652,
        pb_c_init: float = 1.25,
        known_bounds: Optional[KnownBounds] = None,
        num_trajectories_in_buffer: int = int(1e6),
        batch_size: int = int(128),
        num_unroll_steps: int = 5,
        td_steps: int = 6,
        td_lambda: float = 1.0,
        priority_alpha: float = 0.0,
        priority_beta: float = 0.0,
        training_steps: int = int(1e6),
        export_network_every: int = int(1e3),
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        codebook_size: int = 32,
    ):
        self.environment_factory = environment_factory
        self.network_factory = network_factory
        self.num_actors = num_actors
        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.num_simulations = num_simulations
        self.discount = discount
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_dirichlet_fraction = root_dirichlet_fraction
        self.root_dirichlet_adaptive = root_dirichlet_adaptive
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.known_bounds = known_bounds
        self.num_trajectories_in_buffer = num_trajectories_in_buffer
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        self.td_lambda = td_lambda
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        self.training_steps = training_steps
        self.export_network_every = export_network_every
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.codebook_size = codebook_size

def twentyfortyeight_config() -> StochasticMuZeroConfig:
    """Returns the config for the game of 2048."""
    def environment_factory():
        # Returns an implementation of 2048.
        return twenty48()
    def network_factory():
        # 10 layer fully connected Res V2 network with Layer normalization and size
        # 256.
        return Network(action_space=4)
    
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
        num_actors=1, #1000
        visit_softmax_temperature_fn =visit_softmax_temperature,
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
        batch_size=64,#1024
        weight_decay=0.0)

class Node(object):
    """A Node in the MCTS search tree.
    """
    def __init__(self,
    prior: float,
    is_chance: bool = False):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.is_chance = is_chance
        self.reward = 0
    def expanded(self) -> bool:
        return len(self.children) > 0
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
class ActionOutcomeHistory:
    """Simple history container used inside the search.
    Only used to keep track of the actions and chance outcomes executed.
    """
    def __init__(self,
        player: Player,
        history: Optional[List[ActionOrOutcome]] = None):
        self.initial_player = player
        self.history = list(history or [])
    def clone(self):
        return ActionOutcomeHistory(self.initial_player, self.history)
    def add_action_or_outcome(self, action_or_outcome: ActionOrOutcome):
        self.history.append(action_or_outcome)
    def last_action_or_outcome(self) -> ActionOrOutcome:
        return self.history[-1]
    def to_play(self) -> Player:
        # Returns the next player to play based on the initial player and the
        # history of actions and outcomes. For example for backgammon the two
        # players alternate, while for 2048 it is always the same player.
        return 0
    
class ReplayBuffer:
    """A replay buffer to hold the experience generated by the selfplay.
    """
    def __init__(self, config: StochasticMuZeroConfig):
        self.config = config
        self.data = []
        # self.lock = threading.Lock()
    def save(self, seq: Trajectory):
        # print("saving new traj")

    
        if len(self.data) > self.config.num_trajectories_in_buffer:
        # Remove the oldest sequence from the buffer.
            self.data.pop(0)
        self.data.append(seq)
        # print('length after saving ', len(self.data))
    def sample_trajectory(self) -> Trajectory:
        """Samples a trajectory uniformly or using prioritization.
        return self.data[0]
        """
    
        # print("sampling trajectory")

        if len(self.data) > 0:
            return self.data[0]
        return []
    def sample_index(self, seq: Trajectory) -> int:
        """Samples an index in the trajectory uniformly or using
        prioritization.
        """
        # print("sampling index")

        length = len(self.data[0])
        return np.random.randint(0, length-max(self.config.num_unroll_steps, self.config.td_steps))
    def sample_element(self) -> Trajectory:
        """Samples a single element from the buffer.
        """
        # print("sampling element")
        # Sample a trajectory.
        trajectory = self.sample_trajectory()
        state_idx = self.sample_index(trajectory)
        limit = max([self.config.num_unroll_steps, self.config.td_steps])
        # print("limit", limit)
        # Returns a trajectory of experiment.
        return trajectory[state_idx:state_idx + limit]
    def sample(self) -> Sequence[Trajectory]:
        """Samples a training batch.
        """
        # with self.lock:
        if len(self.data) == 0:
            return []
        # print("length", len(self.data))
        return [self.sample_element() for _ in range(self.config.batch_size)]
    
class MinMaxStats(object):
    """A class that holds the min-max values of the tree.
    """
    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE
    def update(self, value: float):
        # print("val", value)
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class NetworkCacher:
    """An object to share the network between the self-play and training
    jobs.
    """
    def __init__(self):
        self._networks = {}
        self.lock = threading.Lock()
    def save_network(self, step: int, network: Network):
        # print("saving network")
        with self.lock:
            self._networks[step] = network
    def load_network(self) -> Tuple[int, Network]:
        # print("loading netwrok")
        with self.lock:
            training_step = max(self._networks.keys())
            return training_step, self._networks[training_step]

class twenty48(Environment):
    """Implements the rules of the environment.
    """

    def __init__(self):
        self.grid_size = 4
        self.last_reward = 0
        self._reset()

    def _reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)  # Use int
        self._add_new_tile()
        self._add_new_tile()
        return self.grid.flatten()
    
    def _add_new_tile(self):
        empty_cells = list(zip(*np.where(self.grid == 0)))
        if empty_cells:
            row, col = empty_cells[np.random.randint(len(empty_cells))]
            self.grid[row, col] = 2 if np.random.rand() < 0.9 else 4  # Float values

    def _merge(self, grid):
        reward = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                if grid[i, j] == grid[i, j + 1] and grid[i, j] != 0:
                    grid[i, j] *= 2.0  # Float values
                    reward += grid[i, j]
                    grid[i, j + 1] = 0
        return grid, reward
    
    def _compress(self, grid):
        new_grid = np.zeros_like(grid)
        for i in range(self.grid_size):
            pos = 0
            for j in range(self.grid_size):
                if grid[i, j] != 0:
                    new_grid[i, pos] = grid[i, j]
                    pos += 1
        return new_grid
    
    def _reverse(self, grid):
        return np.flip(grid, axis=1)

    def _transpose(self, grid):
        return np.transpose(grid)

    def _move(self, direction):
        if direction == 0:  # Up
            self.grid = self._transpose(self.grid)
            self.grid = self._compress(self.grid)
            self.grid, reward = self._merge(self.grid)
            self.grid = self._compress(self.grid)
            self.grid = self._transpose(self.grid)
        elif direction == 1:  # Down
            self.grid = self._transpose(self.grid)
            self.grid = self._reverse(self.grid)
            self.grid = self._compress(self.grid)
            self.grid, reward = self._merge(self.grid)
            self.grid = self._compress(self.grid)
            self.grid = self._reverse(self.grid)
            self.grid = self._transpose(self.grid)
        elif direction == 2:  # Left
            self.grid = self._compress(self.grid)
            self.grid, reward = self._merge(self.grid)
            self.grid = self._compress(self.grid)
        elif direction == 3:  # Right
            self.grid = self._reverse(self.grid)
            self.grid = self._compress(self.grid)
            self.grid, reward = self._merge(self.grid)
            self.grid = self._compress(self.grid)
            self.grid = self._reverse(self.grid)
        else:
            raise ValueError("Invalid direction! Use 0 (up), 1 (down), 2 (left), or 3 (right).")

        return reward
    
    def _step(self, action):
        initial_grid = self.grid.copy()
        reward = self._move(action)
        if not np.array_equal(initial_grid, self.grid):
            self._add_new_tile()
        
        self.last_reward = reward

        # done = self.is_game_over()
        # if done:
        #     reward += np.max(self.grid)
        # return self.grid.flatten(), reward, done
    def _is_game_over(self):
        if np.any(self.grid == 0):  # Empty cells
            return False
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                if self.grid[i, j] == self.grid[i, j + 1]:  # Adjacent horizontal match
                    return False
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size):
                if self.grid[i, j] == self.grid[i + 1, j]:  # Adjacent vertical match
                    return False
        return True
    def _binary_rep(self):
        flat_board = self.grid.flatten()
        arr = np.asarray(flat_board).ravel() 
        bits_array = (arr[:, None] >> np.arange(30, -1, -1)) & 1

        return bits_array.reshape(-1)


    def apply(self, action: Action):
        """Applies an action or a chance outcome to the environment.
        """
        self._step(action)

    def observation(self):
        """Returns the observation of the environment to feed to the network.
        """
        # print("stepping game engine")
        return torch.from_numpy(self._binary_rep().astype(np.float32))
    def is_terminal(self) -> bool:
        """Returns true if the environment is in a terminal state.
        return False
        """
        return self._is_game_over()
    def legal_actions(self) -> Sequence[Action]:
        """Returns the legal actions for the current state.
        return []
        """
        initial_grid = self.grid.copy()
        legal_actions = []
        for a in range(4):
            self.grid = initial_grid.copy()
            self._move(a)
            if not np.array_equal(initial_grid, self.grid):
                legal_actions.append(a)
        return legal_actions


    def reward(self, player: Player) -> float:
        """Returns the last reward for the player.
        return 0.0
        """
        return self.last_reward
    def to_play(self) -> Player:
        """Returns the current player to play.
        return 0"""
        return 0
