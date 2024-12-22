from config import StochasticMuZeroConfig
from utils import ReplayBuffer, Network, NetworkCacher
import torch
import torch.nn as nn
import torch.optim as optim


mse_loss = nn.MSELoss()
kl_div = nn.KLDivLoss()

# Constants
NUM_BINS = 601
MAX_VALUE = 600.0
EPS = 0.001  # Epsilon used in the transform

def muzero_transform(x, eps=EPS):
    # Apply the MuZero transformation h(x)
    # h(x) = sign(x)*(sqrt(|x|+1)-1 + eps*|x|)
    # For 2048, x is non-negative (0 to large values), so sign(x)=1.
    # If negative values are possible, you might consider sign.
    return (torch.sqrt(x.abs() + 1) - 1.0) + eps * x.abs()

def inverse_muzero_transform(y, eps=EPS):
    # Inverse transform to go back from h(x) to x if needed:
    # x = sign(y)*(( (1 + something...) )^2 - 1)
    # For simplicity, assuming x>=0 in 2048:
    # y = sqrt(x+1)-1 + eps*x
    # Let’s solve for x:
    # sqrt(x+1) = y + 1 - eps*x
    # This is more complex; we might not need inverse for training.
    # We won't implement inverse here as it's not required for loss calculation.
    pass

def value_to_distribution(value, num_bins=NUM_BINS, max_value=MAX_VALUE):
    # value is a scalar after transform, which should be in [0, max_value].
    # If not, clamp it.
    value_clamped = torch.clamp(value, 0.0, max_value)
    
    # The bin width is 1.0 in this setup ([0, 600]) => 601 bins.
    # Find the lower and upper bins for the value.
    # Example: If value=10.3, bin_lower=10, bin_upper=11,
    # weight for bin_lower = 0.7, bin_upper = 0.3
    bin_lower = value_clamped.floor().long()
    bin_upper = torch.clamp(bin_lower + 1, max=num_bins - 1)
    
    # Weight for upper bin
    upper_weight = value_clamped - bin_lower.float()
    lower_weight = 1.0 - upper_weight
    
    # Create a zero distribution and fill in the two adjacent bins
    dist = torch.zeros((value.size(0), num_bins), device=value.device)
    dist.scatter_(1, bin_lower.unsqueeze(1), lower_weight.unsqueeze(1))
    dist.scatter_(1, bin_upper.unsqueeze(1), upper_weight.unsqueeze(1))
    
    return dist

class Learner():
    """An learner to update the network weights based.
    """
    # @abc.abstractmethod
    def learn(self):
        """Single training step of the learner.
        """
    # @abc.abstractmethod
    def export(self) -> Network:
        """Exports the network.
        """

def policy_loss(predictions, labels):
    """Minimizes the KL-divergence of the predictions and labels."""
    loss = kl_div(predictions, labels)
    return loss

def compute_td_target(td_steps, td_lambda, trajectory):
    """Computes the TD lambda targets given a trajectory for the 0th
    element.
    Args:
    td_steps: The number n of the n-step returns.
    td_lambda: The lambda in TD(lambda).
    trajectory: A sequence of states. 

    class State(NamedTuple):
    observation: List[float]
    reward: float
    discount: float
    player: Player
    action: Action
    search_stats: SearchStats

    class SearchStats(NamedTuple):
    search_policy: Dict[Action, int]
    search_value: float

    Returns:
    The n-step return.
    """
    rewards = [state.reward for state in trajectory]
    discounts = [state.discount for state in trajectory]
    values = [state.search_stats.search_value for state in trajectory]

    # Length of the trajectory
    T = len(trajectory)

    # Compute the weighted sum of n-step returns
    td_lambda_target = 0.0

    # (1 - λ) factor for the weighted combination
    one_minus_lambda = 1.0 - td_lambda

    for n in range(1, td_steps + 1):
        # Compute G^(n)
        # G^(n) = sum_{k=0}^{n-1} (prod_{j=0}^{k-1} discounts[j]) * rewards[k]
        #         + (prod_{j=0}^{n-1} discounts[j]) * values[n] (if n < T)
        G_n = 0.0
        cum_discount = 1.0
        steps = min(n, T)  # Number of steps we can actually use from the trajectory

        # Sum discounted rewards
        for k in range(steps):
            G_n += cum_discount * rewards[k]
            if k < steps - 1:  # Update discount only if we haven't reached n-th step
                cum_discount *= discounts[k]

        # If we still have a value to bootstrap from (i.e., n < T),
        # include discounted value of the n-th successor state.
        if n < T:
            G_n += cum_discount * discounts[n-1] * values[n]

        # Weight this n-step return by TD(lambda) weights
        weight = one_minus_lambda * (td_lambda ** (n - 1))
        td_lambda_target += weight * G_n

    # Return as a PyTorch tensor
    return torch.tensor(td_lambda_target, dtype=torch.float32)

def value_or_reward_loss(prediction, target):
    """Implements the value or reward loss for Stochastic MuZero.
    For backgammon this is implemented as an MSE loss of scalars.
    For 2048, we use the two hot representation proposed in
    MuZero, and this loss is implemented as a KL divergence between the
    value
    and value target representations.
    For 2048 we also apply a hyperbolic transformation to the target (see
    paper
    for more information).
    Args:
    prediction: The reward or value output of the network.
    target: The reward or value target.
    Returns:
    The loss to minimize."""

    # 1. Apply transform to target
    transformed_target = muzero_transform(target)
    
    # Scale transformed_target into the bin range [0, MAX_VALUE]
    # The transform can produce values that may exceed range if targets are large.
    # In the snippet provided, they mention representing values between [0, 600],
    # so let's clamp/scale if needed.
    
    # If the transform doesn't guarantee [0, MAX_VALUE], we must map:
    # The transform h(x) for large x grows approximately like sqrt(x), 
    # to keep it simple, we assume the target is already in [0, MAX_VALUE]
    # or clamp it:
    transformed_target = torch.clamp(transformed_target, 0.0, MAX_VALUE)
    
    # 2. Convert to distribution
    target_dist = value_to_distribution(transformed_target)
    
    # 3. prediction -> predicted_dist via softmax
    # predicted_dist = F.log_softmax(prediction, dim=1)  # log probabilities
    # We can use cross_entropy style:
    # target_dist is probabilities, predicted_dist is log probs
    # KL divergence: sum target_dist * (log target_dist - log predicted_dist)
    # For training, cross-entropy is often used:
    # Cross-entropy: - sum target_dist * log_predicted_dist
    # If we want KL, we need both target_dist and predicted_dist.
    # Usually, MuZero uses a cross-entropy style loss since target_dist is fixed.
    
    # Cross-entropy loss with soft targets:
    # loss = -torch.sum(target_dist * predicted_dist, dim=1).mean()
    loss = kl_div(target_dist, prediction)
    return loss
    


class StochasticMuZeroLearner(Learner):
    """Implements the learning for Stochastic MuZero.
    """
    def __init__(self,
        config: StochasticMuZeroConfig,
        replay_buffer: ReplayBuffer):
        self.config = config
        self.replay_buffer = replay_buffer
        # Instantiate the network.
        self.network = config.network_factory()
        self.network_optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
    def transpose_to_time(self, batch):
        """Transposes the data so the leading dimension is time instead of
        batch.
        """
        return batch
    def learn(self):
        """Applies a single training step.
        batch = self.replay_buffer.sample()
        """
        # Transpose batch to make time the leading dimension.
        batch = self.transpose_to_time(batch)
        # Compute the initial step loss.
        latent_state = self.network.representation(batch[0].observation)
        predictions = self.network.predictions(latent_state)
        # Computes the td target for the 0th position.
        value_target = compute_td_target(self.config.td_steps,
        self.config.td_lambda,
        batch)
        # Train the network value towards the td target.
        total_loss = value_or_reward_loss(predictions.value, value_target)
        # Train the network policy towards the MCTS policy.
        total_loss += policy_loss(predictions.probabilities,
        batch[0].search_stats.search_policy)
        # Unroll the model for k steps.
        for t in range(1, self.config.num_unroll_steps + 1):
            # Condition the afterstate on the previous action.
            afterstate = self.network.afterstate_dynamics(
            latent_state, batch[t - 1].action)
            afterstate_predictions = self.network.afterstate_predictions(
            afterstate)
            # Call the encoder on the next observation.
            # The encoder returns the chance code which is a discrete one hot code.
            # The gradients flow to the encoder using a straight through estimator.
            chance_code = self.network.encoder(batch[t].observation)
            # The afterstate value is trained towards the previous value target
            # but conditioned on the selected action to obtain a Q-estimate.
            total_loss += value_or_reward_loss(
            afterstate_predictions.value, value_target)

            # The afterstate distribution is trained to predict the chance code
            # generated by the encoder.

            total_loss += policy_loss(afterstate_predictions.probabilities,
            chance_code)
            # Get the dynamic predictions.
            latent_state = self.network.dynamics(afterstate, chance_code)
            predictions = self.network.predictions(latent_state)
            # Compute the new value target.
            value_target = compute_td_target(self.config.td_steps,
            self.config.td_lambda,
            batch[t:])
            # The reward loss for the dynamics network.
            total_loss += value_or_reward_loss(predictions.reward, batch[t].
            reward)
            total_loss += value_or_reward_loss(predictions.value, value_target)
            total_loss += policy_loss(predictions.probabilities,
            batch[t].search_stats.search_policy)
        self.network_optimizer.zero_grad()
        total_loss.backward()
        self.network_optimizer.step()
        # minimize_with_adam_and_weight_decay(total_loss,
        #         learning_rate=self.config.
        #         learning_rate,
        #         weight_decay=self.config.
        #         weight_decay)
        
    def export(self) -> Network:
        return self.network
    
def train_stochastic_muzero(config: StochasticMuZeroConfig,
    cacher: NetworkCacher,
    replay_buffer: ReplayBuffer):
    learner = StochasticMuZeroLearner(config, replay_buffer)
    # Export the network so the actors can start generating experience.
    cacher.save_network(0, learner.export())
    for step in range(config.training_steps):
        # Single learning step.
        learner.learn()
        if step > 0 and step % config.export_network_every == 0:
            cacher.save_network(step, learner.export())