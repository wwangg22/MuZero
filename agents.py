from utils import Network, NetworkCacher, Environment, Node, MinMaxStats, ReplayBuffer, ActionOutcomeHistory
from typesMZ import SearchStats, NetworkOutput, LatentState, AfterState, Action, Outcome, State
from encoders import VQCodebook, ResNetV2Tower
from MCTS import expand_node, backpropagate, add_exploration_noise, run_mcts
from config import StochasticMuZeroConfig
from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Actor():
    """An actor to interact with the environment.
    """
    # @abc.abstractmethod
    def reset(self):
        """Resets the player for a new episode."""
    # @abc.abstractmethod
    def select_action(self, env: Environment) -> Action:
        """Selects an action for the current state of the environment.
        """
    # @abc.abstractmethod
    def stats(self) -> SearchStats:
        """Returns the stats for the player after it has selected an action."""


class StochasticMuZeroActor(Actor):
    def __init__(self,
        config: StochasticMuZeroConfig,
        cacher: NetworkCacher):
        self.config = config
        self.cacher = cacher
        self.training_step = -1
        self.network = None
    def reset(self):
        # Read a network from the cacher for the new episode.
        self.training_step, self.network = self.cacher.load_network()
        self.root = None
    def _mask_illegal_actions(self,
        env: Environment,
        outputs: NetworkOutput) -> NetworkOutput:
        """Masks any actions which are illegal at the root.
        """
        # We mask out and keep only the legal actions.
        masked_policy = {}
        network_policy = outputs.probabilities
        norm = 0
        for action in env.legal_actions():
            if action in network_policy:
                masked_policy[action] = network_policy[action]
            else:
                masked_policy[action] = 0.0
        norm += masked_policy[action]
        # Renormalize the masked policy.
        masked_policy = {a: v / norm for a, v in masked_policy.items()}
        return NetworkOutput(value=outputs.value, probabilities=masked_policy
        )
    def _select_action(self, root: Node):
        """Selects an action given the root node.
        """
        # Get the visit count distribution.
        # actions, visit_counts = zip(*[
        #     (action, node.visit_counts)
        #     for action, node in node.children.items()
        # ])
        actions, visit_counts = zip(*[
            (action, child_node.visit_counts)
            for action, child_node in root.children.items()
        ])
        # Temperature
        temperature = self.config.visit_softmax_temperature_fn(self.
        training_step)
        # Compute the search policy.
        search_policy = [v ** (1. / temperature) for v in visit_counts]
        norm = sum(search_policy)
        search_policy = [v / norm for v in search_policy]
        return np.random.choice(actions, p=search_policy)
    
    def select_action(self, env: Environment) -> Action:
        """Selects an action.
        """
        # New min max stats for the search tree.
        min_max_stats = MinMaxStats(self.config.known_bounds)
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        # Provide the history of observations to the representation network to
        # get the initial latent state.
        latent_state = self.network.representation(env.observation())
        # Compute the predictions.
        outputs = self.network.predictions(latent_state)
        # Keep only the legal actions.
        outputs = self._mask_illegal_actions(env, outputs)
        # Expand the root node.
        expand_node(root, latent_state, outputs, env.to_play(), is_chance=False)
        # Backpropagate the value.
        backpropagate([root], outputs.value, env.to_play(), self.config.discount, min_max_stats)
        # We add exploration noise to the root node.
        add_exploration_noise(self.config, root)
        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(self.config, root, ActionOutcomeHistory(env.to_play()), self.network, min_max_stats)
        # Keep track of the root to return the stats.
        self.root = root
        # Return an action.
        return self._select_action(root)
    
    def stats(self) -> SearchStats:
        """Returns the stats of the latest search.
        """
        if self.root is None:
            raise ValueError('No search was executed.')
        return SearchStats(
            search_policy={
                action: node.visit_counts
                for action, node in self.root.children.items()
            },
            search_value=self.root.value())

# Self-play.
# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces an episode and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: StochasticMuZeroConfig,
                    cacher: NetworkCacher,
                    replay_buffer: ReplayBuffer):
    actor = StochasticMuZeroActor(config, cacher)

    while True:
        # Create a new instance of the environment.
        env = config.environment_factory()
        # Reset the actor.
        actor.reset()
        episode = []
        while not env.is_terminal():
            action = actor.select_action(env)
            state = State(
                observation=env.observation(),
                reward=env.reward(env.to_play()),
                discount=config.discount,
                player=env.to_play(),
                action=action,
                search_stats=actor.stats())
            episode.append(state)
            env.apply(action)
        # Send the episode to the replay.
        replay_buffer.save(episode)


class Network:
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
        self.obs_to_hidden = nn.Linear(observation_dim, hidden_dim)

        # 10-block ResNet v2 tower
        self.resnet_tower_rep = ResNetV2Tower(input_dim=hidden_dim, num_blocks=10)
        self.resnet_tower_af_dyn = ResNetV2Tower(input_dim=hidden_dim, num_blocks=10)
        self.resnet_tower_dyn = ResNetV2Tower(input_dim=hidden_dim, num_blocks=10)



        # Heads for predictions: from latent state -> value and policy
        # We assume predictions from a latent state include:
        # - value: scalar
        # - policy: probabilities over actions
        self.value_head = nn.Linear(hidden_dim, 601)
        self.policy_head = nn.Linear(hidden_dim, self.action_space)

        # For afterstate predictions, we reuse the same tower and just produce a policy and value (no reward).
        # We can reuse value_head and policy_head with a different forward pass, 
        # or create separate heads. Let's reuse for simplicity.

        # Afterstate dynamics:
        # given a latent state and an action, produce an afterstate.
        # We'll model this as a simple MLP:
        self.afterstate_fc = nn.Linear(hidden_dim + self.action_space, hidden_dim)

        # Dynamics:
        # given afterstate and outcome, produce next latent state.
        # Similarly, an MLP:
        self.dynamics_fc = nn.Linear(hidden_dim + self.outcome_space, hidden_dim)

        # Encoder:
        # maps observation to outcome distribution (logits) to be quantized.
        # The snippet suggests encoder producing codebook indices. We'll do a simple linear:
        self.encoder_fc = nn.Linear(observation_dim, codebook_size)
        self.codebook = VQCodebook(M=codebook_size)

        # Reward head (for state transitions):
        # The snippet suggests that predictions for a latent state may include a reward.
        # The original code returns a reward in NetworkOutput.
        # We'll add a reward head from latent state for completeness:
        self.reward_head = nn.Linear(hidden_dim, 601)


    def representation(self, observation) -> LatentState:
        """Representation function maps from observation to latent state.
        Here we assume 'observation' is a tensor of shape (batch, observation_dim).
        """
        # Map observation to hidden dim
        x = self.obs_to_hidden(observation)
        # Pass through ResNet tower
        x = self.resnet_tower_rep(x)
        # x is now a latent state
        return x

    def predictions(self, state: LatentState) -> 'NetworkOutput':
        """Returns the network predictions for a latent state: value, policy, reward.
        returns an output of 601
        """
        value = self.value_head(state)
        value_prob = F.softmax(value, dim = -1)
        policy_logits = self.policy_head(state)
        policy_probs = F.softmax(policy_logits, dim=-1)
        # For simplicity, pick an example reward from the reward head:
        reward = self.reward_head(state)
        reward_prob = F.softmax(reward, dim=-1)
        # Convert probabilities to a dict {action: prob}
        # Assuming action space is integer from 0 to action_space-1
        probabilities = {a: policy_probs[0, a].item() for a in range(self.action_space)}

        return NetworkOutput(value=value_prob, probabilities=probabilities, reward=reward_prob)

    def afterstate_dynamics(self, state: LatentState, action: Action) -> AfterState:
        """Implements the dynamics from latent state and action to afterstate.
        We one-hot encode the action and concatenate with state, then process through a linear layer + tower.
        """
        if not isinstance(action, int):
            # In real code, handle different action types.
            # Assume action is int representing action index.
            action = int(action)

        action_one_hot = F.one_hot(torch.tensor([action]), num_classes=self.action_space).float()
        # Expand state to batch dimension if needed
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # (1, hidden_dim)
        # Concatenate
        x = torch.cat([state, action_one_hot], dim=-1)
        x = self.afterstate_fc(x)
        x = self.resnet_tower_ad_dyn(x)  
        return x

    def afterstate_predictions(self, state: AfterState) -> 'NetworkOutput':
        """Returns the network predictions for an afterstate.
        Afterstates differ from latent states in that we do not produce a reward here (assumption from snippet).
        """
        value = self.value_head(state).squeeze(-1).item()
        policy_logits = self.policy_head(state)
        policy_probs = F.softmax(policy_logits, dim=-1)
        probabilities = {a: policy_probs[0, a].item() for a in range(self.action_space)}

        # No reward for afterstate transitions as per snippet comment.
        return NetworkOutput(value=value, probabilities=probabilities, reward=0.0)

    def dynamics(self, state: AfterState, outcome: Outcome) -> LatentState:
        """Implements the dynamics from afterstate and chance outcome to state.
        We one-hot encode the outcome (from the codebook) and combine with afterstate.
        """
        if not isinstance(outcome, int):
            # Assume outcome is int index from 0 to outcome_space-1
            outcome = int(outcome)

        outcome_one_hot = F.one_hot(torch.tensor([outcome]), num_classes=self.outcome_space).float()
        # ensure state is batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
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
        logits = self.encoder_fc(observation)
        # Let's say during inference we just use argmax (no gumbel).
        code_one_hot = self.codebook(logits, use_gumbel=False)
        # Convert the one-hot vector back to an integer index outcome:
        outcome_idx = torch.argmax(code_one_hot, dim=-1).item()
        return outcome_idx

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
    
