from utils import ReplayBuffer, NetworkCacher, StochasticMuZeroConfig, Network
from typesMZ import SearchStats, State
import numpy as np
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
     # 1) Convert input to a tensor if it's not already
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=torch.float)

    # 2) If value is scalar (0D), unsqueeze to make it [1]
    if value.dim() == 0:
        value = value.unsqueeze(0)

    # 3) Now value is shape [N]. Clamp it to [0, max_value]
    value_clamped = torch.clamp(value, min=0.0, max=max_value)

    # 4) Identify bin indices (lower and upper)
    #    e.g. if value=10.3 => bin_lower=10, bin_upper=11
    bin_lower = value_clamped.floor().long()
    bin_upper = torch.clamp(bin_lower + 1, max=num_bins - 1)

    # 5) Calculate weights for how much goes into the lower or upper bin
    #    e.g. value=10.3 => lower_weight=0.7, upper_weight=0.3
    upper_weight = value_clamped - bin_lower.float()
    lower_weight = 1.0 - upper_weight

    # 6) Create a zero distribution of shape [N, num_bins]
    dist = torch.zeros((value_clamped.size(0), num_bins), device=value.device)

    # 7) Scatter weights into correct bins
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
    """Minimizes the2 KL-divergence of the predictions and labels."""
    # print(predictions, labels)
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
    rewards = torch.stack([state.reward for state in trajectory], dim=1)
    discounts = torch.stack([state.discount for state in trajectory], dim=1)
    values = torch.stack([state.search_stats.search_value for state in trajectory],dim=1)
    print("rewards shape", rewards.shape)
    print("discotuns shape", discounts.shape)
    print("value shape", values.shape)

    # Length of the trajectory
    T = len(trajectory)

    # Compute the weighted sum of n-step returns
    td_lambda_target = torch.zeros(len(rewards), dtype=torch.float32)

    one_minus_lambda = 1.0 - td_lambda  # (1 - λ) factor

    for n in range(1, td_steps + 1):
        # G_n will be shape [batch_size]
        G_n = torch.zeros_like(td_lambda_target)
        # cum_discount also shape [batch_size], starts at 1.0
        cum_discount = torch.ones_like(td_lambda_target)

        steps = min(n, T)

        # 1) Sum of discounted rewards
        #    G^(n) = sum_{k=0}^{n-1} (prod_{j=0}^{k-1} discount[j]) * rewards[k]
        for k in range(steps):
            G_n += cum_discount * rewards[:, k]
            if k < steps - 1:
                cum_discount *= discounts[:, k]

        # 2) If n < T, bootstrap from the value at time step n
        #    G^(n) += (prod_{j=0}^{n-1} discount[j]) * values[n]
        #    (Note that we multiply once more by discounts[:, n-1]
        #     because we updated cum_discount up to k < steps-1)
        if n < T:
            G_n += cum_discount * discounts[:, n - 1] * values[:, n]

        # 3) Weight this n-step return by (1-λ)*(λ^(n-1))
        weight = one_minus_lambda * (td_lambda ** (n - 1))

        # 4) Accumulate into the TD(lambda) target
        td_lambda_target += weight * G_n

    # Return as a PyTorch tensor
    # print("td _lambda _target", td_lambda_target)

    return td_lambda_target


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
    # print("prediciton 1", prediction.shape)
    # print(" target : ", target.shape)
    # print("prediction shape", prediction.sum(dim=1))

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
    # print("target dist ", target_dist.shape)
    
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
    # print("value or reward loss", loss)
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
        # print(batch[0][0].observation)
        transposed = [list(row) for row in zip(*batch)]
        new_arr = []
        for row in transposed:
            combined_observation = torch.stack([state.observation for state in row])
            combined_rewards = torch.tensor([state.reward for state in row])
            combined_discount = torch.tensor([state.discount for state in row])
            combined_player = torch.tensor([state.player for state in row])
            combined_action = torch.tensor([state.action for state in row])
            combined_search = torch.stack([
                torch.tensor(state.search_stats.search_policy) / sum(state.search_stats.search_policy)
                for state in row
            ])
            combined_value = torch.tensor([state.search_stats.search_value for state in row])

            new_search = SearchStats(search_policy=combined_search, search_value=combined_value)
            new_state = State(observation=combined_observation, reward=combined_rewards, discount = combined_discount, player=combined_player, action = combined_action, search_stats=new_search)
            new_arr.append(new_state)
        return new_arr

        return batch
    def learn(self):
        """Applies a single training step.
        batch = self.replay_buffer.sample()
        """
        batch = self.replay_buffer.sample()

        print(len(batch), len(batch[0]))
        if len(batch) == 0:
            # print("length too low, trying again")
            return
        print('running update loop!')
        # Transpose batch to make time the leading dimension.
        batch = self.transpose_to_time(batch) #doesnt do anythin yet
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
        print("total loss 1 ", total_loss)
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
            print("total loss 2 ", total_loss)

        
            # The afterstate distribution is trained to predict the chance code
            # generated by the encoder.

            total_loss += policy_loss(afterstate_predictions.probabilities,
            chance_code)
            print("total loss 3 ", total_loss)

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
            print("total loss 4 ", total_loss)

            total_loss += value_or_reward_loss(predictions.value, value_target)
            print("total loss 5 ", total_loss)

            total_loss += policy_loss(predictions.probabilities,
            batch[t].search_stats.search_policy)
            print("total loss 6 ", total_loss)

        self.network_optimizer.zero_grad()
        total_loss.backward()
        self.network_optimizer.step()
        print("total_losee", total_loss)
        # minimize_with_adam_and_weight_decay(total_loss,
        #         learning_rate=self.config.
        #         learning_rate,
        #         weight_decay=self.config.
        #         weight_decay)
        
    def export(self) -> Network:
        return self.network
    
def train_stochastic_muzero(config: StochasticMuZeroConfig,
    cacher: NetworkCacher,
    replay_buffer: ReplayBuffer,
    learner: StochasticMuZeroLearner):
    print(f"ReplayBuffer ID: {id(replay_buffer)}")

    
    for step in range(config.training_steps):
        # Single learning step.
        # print(f"ReplayBuffer ID: {id(replay_buffer)}")

        learner.learn()
        if step > 0 and step % config.export_network_every == 0:
            cacher.save_network(step, learner.export())