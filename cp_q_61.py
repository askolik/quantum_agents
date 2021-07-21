import tensorflow as tf
from src.utils.limit_thread_usage import set_thread_usage_limit
set_thread_usage_limit(10, tf)

from config import BASE_PATH, ALICE_BASE_PATH, Envs, EncType
from parallelize import parallelize_cp_q


hyperparams = {
    'episodes': [5000],
    'batch_size': [64],
    'epsilon': [1],
    'epsilon_decay': [0.99],
    'epsilon_min': [0.01],
    'gamma': [0.99],
    'update_after': [1],
    'update_target_after': [1],
    'learning_rate': [0.001],
    'learning_rate_in': [0.001],
    'learning_rate_out': [0.1],
    'circuit_depth': [12],
    'epsilon_schedule': ['fast'],
    'reps': 10,
    'env': Envs.CARTPOLE,
    'save': True,
    'test': False
}


if __name__ == '__main__':
    parallelize_cp_q(hyperparams, BASE_PATH + 'cartpole/depth_scaling/')
