from utils import ReplayBuffer, NetworkCacher
from learner import train_stochastic_muzero
from agents import run_selfplay
from config import StochasticMuZeroConfig

def launch_stochastic_muzero(config: StochasticMuZeroConfig):
    """Full RL loop for stochastic MuZero.
    """
    replay_buffer = ReplayBuffer(config)
    cacher = NetworkCacher()
    # Launch a learner job.
    launch_job(lambda: train_stochastic_muzero(config, cacher,
    replay_buffer))
    # Launch the actors.
    for _ in range(config.num_actors):
        launch_job(lambda: run_selfplay(config, cacher, replay_buffer))
    # Stubs to make the typechecker happy.
def softmax_sample(distribution, temperature: float):
    return 0, 0

def minimize_with_sgd(loss, learning_rate):
    """Minimizes the loss using SGD.
    """
def minimize_with_adam_and_weight_decay(loss, learning_rate, weight_decay
    ):
    """Minimizes the loss using Adam with weight decay.
    """
def launch_job(f):
    """Launches a job to run remotely.
    return f()
    """