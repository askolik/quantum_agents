import tensorflow as tf
from src.utils.limit_thread_usage import set_thread_usage_limit
set_thread_usage_limit(10, tf)

from config import BASE_PATH, Envs
from parallelize import parallelize_cp_c

hyperparams = {
    'episodes': [5000],
    'batch_size': [32],
    'epsilon': [1],
    'epsilon_decay': [0.99],
    'epsilon_min': [0.01],
    'gamma': [0.99],
    'update_after': [1],
    'update_target_after': [1],
    'learning_rate': [0.001],
    'epsilon_schedule': ['fast'],
    'n_hidden_layers': [2],
    'hidden_layer_config': [[24, 24]],
    'reps': 10,
    'env': Envs.CARTPOLE,
    'save': True,
    'test': False
}


if __name__ == '__main__':
    parallelize_cp_c(hyperparams, BASE_PATH + 'cartpole_classical/params_770/')
