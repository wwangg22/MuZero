from utils import ReplayBuffer, NetworkCacher, StochasticMuZeroConfig, twentyfortyeight_config
from learner import train_stochastic_muzero, StochasticMuZeroLearner
from agents import run_selfplay, run_selfplay_w_update
import threading

def launch_stochastic_muzero(config: StochasticMuZeroConfig):
    """Full RL loop for stochastic MuZero.
    """
    replay_buffer = ReplayBuffer(config)
    cacher = NetworkCacher()
    learner = StochasticMuZeroLearner(config, replay_buffer)
    # Export the network so the actors can start generating experience.
    cacher.save_network(0, learner.export())
    # Launch a learner job.
    # launch_job(lambda: train_stochastic_muzero(config, cacher,
    # replay_buffer))
    # learner_thread = threading.Thread(
    #     target=train_stochastic_muzero, 
    #     args=(config, cacher, replay_buffer, learner),
    # )
    # learner_thread.start()
    # Launch the actors.
    # for _ in range(config.num_actors):
    #     launch_job(lambda: run_selfplay(config, cacher, replay_buffer))

    # actor_threads = []
    # for _ in range(config.num_actors):
    #     th = threading.Thread(
    #         target=run_selfplay, 
    #         args=(config, cacher, replay_buffer),
    #     )
    #     th.start()
    #     actor_threads.append(th)
    run_selfplay_w_update(config=config, cacher=cacher, replay_buffer=replay_buffer, learner=learner)
    # train_stochastic_muzero(config=config, cacher=cacher, replay_buffer=replay_buffer, learner=learner)

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
    """
    Launches a job to run remotely (or locally). 
    This placeholder just runs it synchronously in the current process.
    
    In a real, distributed environment, you might do:
      - Launch on a separate process/thread
      - Submit to a cluster
      - Use Ray, etc.

    For now, we just call f() directly.
    """
    return f()

if __name__ == "__main__":
    launch_stochastic_muzero(twentyfortyeight_config())