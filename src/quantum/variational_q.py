
import multiprocessing
import numpy as np
import random
import gym
import cirq
import sympy
import time
import tensorflow as tf
import tensorflow_quantum as tfq

from collections import deque
from config import BASE_PATH, Envs
from src.quantum.model import state_to_circuit, create_q_circuit, construct_readout_ops, perform_action, q_val, \
    add_to_memory, build_models
from src.quantum.training import perform_episodes
from src.utils.storage import save_data, clean_after_test


def parallelize_runs(hyperparams, path=BASE_PATH):
    experiments = ([
        hyperparams,
        path,
        episodes,
        max_steps,
        batch_size,
        epsilon,
        epsilon_decay,
        epsilon_min,
        gamma,
        update_target_after,
        learning_rate,
        circuit_depth,
        encoding_depth,
        multiply_output_by,
        epsilon_schedule,
        model_update_prob,
        vector_form,
        alpha,
        fixed_update_after]

        for episodes in hyperparams['episodes']
        for max_steps in hyperparams['max_steps']
        for batch_size in hyperparams['batch_size']
        for epsilon in hyperparams['epsilon']
        for epsilon_decay in hyperparams['epsilon_decay']
        for epsilon_min in hyperparams['epsilon_min']
        for gamma in hyperparams['gamma']
        for update_target_after in hyperparams['update_target_after']
        for learning_rate in hyperparams['learning_rate']
        for circuit_depth in hyperparams['circuit_depth']
        for encoding_depth in hyperparams['encoding_depth']
        for multiply_output_by in hyperparams['multiply_output_by']
        for epsilon_schedule in hyperparams.get('epsilon_schedule', ['fast'])
        for model_update_prob in hyperparams.get('model_update_prob', [None])
        for vector_form in hyperparams.get('vector_form', [False])
        for alpha in hyperparams.get('alpha', [0.01])
        for fixed_update_after in hyperparams.get('fixed_update_after', [None]))

    pool = multiprocessing.Pool()
    pool.map(run_qnet, experiments)
    pool.close()
    pool.join()
    return None


def run_qnet(args, test=False):
    (
        hyperparams,
        path,
        episodes,
        max_steps,
        batch_size,
        epsilon,
        epsilon_decay,
        epsilon_min,
        gamma,
        update_target_after,
        learning_rate,
        circuit_depth,
        encoding_depth,
        multiply_output_by,
        epsilon_schedule,
        model_update_prob,
        vector_form,
        alpha,
        fixed_update_after) = args

    fixed_memory = False
    epochs = 1
    load_model = None
    if hyperparams is not None:
        fixed_memory = hyperparams['fixed_memory']
        epochs = hyperparams['epochs']
        load_model = hyperparams.get('load_model')

    print("\nUpdate target after: {},\nLearning rate: {},"
          "\nCircuit/encoding depth: {}/{}\nFixed memory: {}"
          "\nEpochs: {}\nVector form: {}\nalpha: {}".format(
        update_target_after,
        learning_rate,
        circuit_depth,
        encoding_depth,
        fixed_memory,
        epochs,
        vector_form,
        alpha))

    env_type = hyperparams.get('env')
    env = gym.make(env_type.value)

    # np.random.seed(123)
    # env.seed(123)

    action_space = env.action_space.n

    if env_type == Envs.FROZENLAKE:
        observation_space = 4
        loss = tf.keras.losses.mse
        n_qubits = observation_space
        qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
    elif env_type == Envs.CARTPOLE:
        observation_space = env.observation_space.shape[0]
        loss = None
        n_qubits = observation_space
        qubits = [cirq.GridQubit(0, i) for i in range(n_qubits + 1)]

    readout_op = construct_readout_ops(qubits, env_type)
    print("Readout:", readout_op, "\n")

    circuit, symbols = create_q_circuit(observation_space, circuit_depth)
    print(circuit, "\n")

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model, target_model = build_models(circuit, readout_op, opt, loss)

    #### Load weights if specified ####

    if load_model is not None:
        model.load_weights(load_model)
        target_model.load_weights(load_model)

    #### Training ####

    timestamp = time.localtime()

    path_prefix = ''
    if not test:
        save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(random.randint(0, 1000))
    else:
        save_as = 'dummy'
        # path_prefix = '../../'

    meta = {
        'episodes': episodes,
        'max_steps': max_steps,
        'batch_size': batch_size,
        'epsilon': epsilon,
        'epsilon_decay': epsilon_decay,
        'epsilon_min': epsilon_min,
        'gamma': gamma,
        'update_target_after': update_target_after,
        'learning_rate': learning_rate,
        'circuit_depth': circuit_depth,
        'encoding_depth': encoding_depth,
        'multiply_output_by': multiply_output_by,
        'epsilon_schedule': epsilon_schedule,
        'model_update_prob': model_update_prob,
        'vector_form': vector_form,
        'alpha': alpha,
        'fixed_update_after': fixed_update_after,
        'env_name': env_type.value}

    plot_title = "episodes: {}, steps: {}, bs: {},\nuta: {}, lr: {}, cd: {}, ed: {}, mo: {}\neps: {}, mup: {}".format(
        episodes, max_steps, batch_size, update_target_after, learning_rate,
        circuit_depth, encoding_depth, multiply_output_by, epsilon_schedule, model_update_prob or fixed_update_after)

    scores, loss_history, epsilons = perform_episodes(
        episodes,
        max_steps,
        env,
        env_type,
        model,
        target_model,
        circuit,
        symbols,
        readout_op,
        epsilon,
        fixed_memory,
        batch_size,
        model_update_prob,
        gamma,
        action_space,
        encoding_depth,
        opt,
        multiply_output_by,
        update_target_after,
        epsilon_schedule,
        epsilon_min,
        epsilon_decay,
        save_as,
        meta,
        plot_title,
        path,
        vector_form,
        alpha,
        fixed_update_after)

    save_data(save_as, meta, model, plot_title, scores, loss_history, epsilons, path=path)

    if test:
        exit()
