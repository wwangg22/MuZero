from utils import NetworkCacher, Environment, Node, MinMaxStats, ReplayBuffer, ActionOutcomeHistory, StochasticMuZeroConfig
from typesMZ import SearchStats, NetworkOutput, LatentState, AfterState, Action, Outcome, State
from encoders import VQCodebook, ResNetV2Tower
from learner import StochasticMuZeroLearner
from typing import Callable
from MCTS import expand_node, backpropagate, add_exploration_noise, run_mcts
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
        masked_policy = [0.0] * len(outputs.probabilities)

        network_policy = outputs.probabilities
        # print("netwrok policy", network_policy)
        norm = 0
        for action in env.legal_actions():
            masked_policy[action] = network_policy[action].item()
        # print("masked policy", masked_policy)
        norm += masked_policy[action]
        # Renormalize the masked policy.
        masked_policy = [a / (norm + 1e-5) for a in masked_policy]
        # print("masked_policy", masked_policy)
        return NetworkOutput(value=outputs.value, float_value=outputs.float_value, probabilities=torch.tensor(masked_policy), reward=0.0, float_reward=0.0)
    def _select_action(self, root: Node):
        """Selects an action given the root node.
        """
        # Get the visit count distribution.
        # actions, visit_counts = zip(*[
        #     (action, node.visit_counts)
        #     for action, node in node.children.items()
        # ])
        actions, visit_counts = zip(*[
            (action, child_node.visit_count)
            for action, child_node in root.children.items()
        ])
        # Temperature
        temperature = self.config.visit_softmax_temperature_fn(self.
        training_step)
        # Compute the search policy.
        if temperature == 0:
            maxi = max(visit_counts)
            search_policy = [ v if v == maxi else 0 for v in visit_counts]
        else:
            search_policy = [v ** (1. / (temperature + 1e-6)) for v in visit_counts]
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
        # print("outputs", outputs.probabilities)
        outputs = self._mask_illegal_actions(env, outputs)
        # Expand the root node.
        # print("outpus", outputs.probabilities)

        expand_node(root, latent_state, outputs, env.to_play(), is_chance=False)
        # Backpropagate the value.
        # print("probsss", outputs.probabilities)

        # print('float val', outputs.float_value)
        backpropagate([root], outputs.float_value, env.to_play(), self.config.discount, min_max_stats)
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
        search_policy = [0] * 4 # Initialize a list with zeros
        for action, node in self.root.children.items():
            # print(action)
            search_policy[action] = node.visit_count
        return SearchStats(
            search_policy=search_policy,
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
        print(f"ReplayBuffer ID: {id(replay_buffer)}")

        while not env.is_terminal():
            # print("hello")
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

def run_selfplay_w_update(config: StochasticMuZeroConfig,
                    cacher: NetworkCacher,
                    replay_buffer: ReplayBuffer,
                    learner: StochasticMuZeroLearner):
    actor = StochasticMuZeroActor(config, cacher)
    step = 0

    while True:
        # Create a new instance of the environment.
        env = config.environment_factory()
        # Reset the actor.
        actor.reset()
        episode = []
        print(f"ReplayBuffer ID: {id(replay_buffer)}")

        while not env.is_terminal():
            # print("hello")
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
        learner.learn()
        cacher.save_network(step, learner.export())
        step +=1




