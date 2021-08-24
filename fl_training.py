from config import Envs, BASE_PATH
from src.quantum.training import QLearningFL


hyperparams = {
    'episodes': 5000,
    'max_steps': 200,
    'fixed_memory': False,
    'batch_size': 11,
    'gamma': 0.8,
    'circuit_depth': 15,
    'encoding_depth': 1,
    'multiply_output_by': 1,
    'update_after': 1,
    'update_target_after': 10,
    'vector_form': True,
    'memory_length': 10000,
    'train_readout': False,
    'epsilon': 1,
    'epsilon_schedule': 'fast',
    'epsilon_min': 0.01,
    'epsilon_decay': 0.99,
    'learning_rate': 0.001,
    'env_solved_at': [],
    'task': 'q'
}

qlearning = QLearningFL(
    hyperparams,
    Envs.FROZENLAKE,
    save=False,
    save_as='dummy',
    path=BASE_PATH + 'frozen_lake/fl_q_learning/tfq/',
    slippery=False,
    test=True)

qlearning.perform_episodes()
