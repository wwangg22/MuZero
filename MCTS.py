from config import StochasticMuZeroConfig
from utils import ActionOutcomeHistory, Network, MinMaxStats, Node, NetworkOutput
import numpy as np
import math
from typing import Union, List


Player = int

LatentState = List[float]
AfterState = List[float]

def run_mcts(config: StochasticMuZeroConfig, root: Node,
    action_outcome_history: ActionOutcomeHistory, network:
    Network,
    min_max_stats: MinMaxStats):
    for _ in range(config.num_simulations):
        history = action_outcome_history.clone()
        node = root
        search_path = [node]
        while node.expanded():
            action_or_outcome, node = select_child(config, node, min_max_stats)
            history.add_action(action_or_outcome)
            search_path.append(node)
        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        if parent.is_chance:
            # The parent is a chance node, afterstate to latent state transition.
            # The last action or outcome is a chance outcome.
            child_state = network_output.dynamics(parent.state,
            history.last_action_or_outcome())
            network_output = network_output.predictions(child_state)
            # This child is a decision node.
            is_child_chance = False
        else:
            # The parent is a decision node, latent state to afterstate transition.
            # The last action or outcome is an action.
            child_state = network_output.afterstate_dynamics(
            parent.state, history.last_action_or_outcome())
            network_output = network_output.afterstate_predictions(child_state)
            # The child is a chance node.
            is_child_chance = True
            # Expand the node.
        expand_node(node, child_state, network_output, history.to_play(),
            is_child_chance)
        # Backpropagate the value up the tree.
        backpropagate(search_path, network_output.value, history.to_play(),
                config.discount, min_max_stats)

# Select the child with the highest UCB score.
def select_child(config: StochasticMuZeroConfig, node: Node,
    min_max_stats: MinMaxStats):
    if node.is_chance:
        # If the node is chance we sample from the prior.
        outcomes, probs = zip(*[(o, n.prob) for o, n in node.children.items()
        ])
        outcome = np.random.choice(outcomes, p=probs)
        return outcome, node.children[outcome]
        # For decision nodes we use the pUCT formula.
    _, action, child = max(
    (ucb_score(config, node, child, min_max_stats), action, child)
    for action, child in node.children.items())
    return action, child

# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: StochasticMuZeroConfig, parent: Node, child: Node,
    min_max_stats: MinMaxStats) -> float:

    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(child.reward +
        config.discount * child.value()
        )
    else:
        value_score = 0
    return prior_score + value_score

# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(node: Node, state: Union[LatentState, AfterState],
    network_output: NetworkOutput, player: Player, is_chance:
    bool):
    node.to_play = player
    node.state = state
    node.is_chance = is_chance
    node.reward = network_output.reward
    for action, prob in network_output.probabilities.items():
        node.children[action] = Node(prob)

# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play: Player,
    discount: float, min_max_stats: MinMaxStats):
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())
        value = node.reward + discount * value

# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: StochasticMuZeroConfig, node: Node):
    actions = list(node.children.keys())
    dir_alpha = config.root_dirichlet_alpha
    if config.root_dirichlet_adaptive:
        dir_alpha = 1.0 / np.sqrt(len(actions))
        noise = np.random.dirichlet([dir_alpha] * len(actions))
        frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac