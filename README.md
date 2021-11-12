# Quantum agents in the Gym: a variational quantum algorithm for deep Q-learning

Code accompanying the paper: https://arxiv.org/abs/2103.15084

To train a quantum agent on the CartPole environment, run *run_quantum.py* and set the hyperparameters in the file:

    hyperparams = {
        'episodes': [5000],
        'batch_size': [16],
        'epsilon': [1],
        'epsilon_decay': [0.99],
        'epsilon_min': [0.01],
        'gamma': [0.99],
        'update_after': [1],
        'update_target_after': [1],
        'learning_rate': [0.001],
        'learning_rate_in': [0.001],
        'learning_rate_out': [0.1],
        'circuit_depth': [5],
        'epsilon_schedule': ['fast'],
        'use_reuploading': True,
        'trainable_scaling': True,
        'trainable_output': True,
        'output_factor': 1,
        'reps': 10,
        'env': Envs.CARTPOLE,
        'save': True,
        'test': False
    }