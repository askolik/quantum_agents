import multiprocessing
import random

import time
from copy import copy

from config import BASE_PATH, Envs
from src.quantum.training import RegressionFL, QLearning, QLearningFL, QLearningCartpole, QLearningCartpoleClassical


def parallelize_fl_regression(hyperparams, path=BASE_PATH):
    experiments = ([
        hyperparams,
        path,
        learning_rate,
        gamma,
        output_factor,
        circuit_depth,
        encoding_depth]

        for learning_rate in hyperparams.get('learning_rate')
        for gamma in hyperparams.get('gamma')
        for output_factor in hyperparams.get('output_factor')
        for circuit_depth in hyperparams.get('circuit_depth')
        for encoding_depth in hyperparams.get('encoding_depth')
    )

    pool = multiprocessing.Pool()
    pool.map(_run_fl_reg, experiments)
    pool.close()
    pool.join()
    return None


def _run_fl_reg(args):
    (
        hyperparams,
        path,
        learning_rate,
        gamma,
        output_factor,
        circuit_depth,
        encoding_depth) = args

    iterations = hyperparams.get('iterations')
    save = hyperparams.get('save', True)
    save_as = hyperparams.get('save_as')
    test = hyperparams.get('test', False)

    if save and save_as is None:
        timestamp = time.localtime()
        save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(random.randint(0, 1000))

    if test:
        save_as = 'dummy'

    rfl = RegressionFL(
        circuit_depth, encoding_depth, gamma, learning_rate, output_factor,
        save=save, save_as=save_as, test=test, path=path)
    rfl.regress(iterations)


def parallelize_fl_q(hyperparams, path=BASE_PATH):
    experiments = ([
        hyperparams,
        path,
        episodes,
        max_steps,
        fixed_memory,
        batch_size,
        gamma,
        circuit_depth,
        encoding_depth,
        multiply_output_by,
        update_after,
        update_target_after,
        vector_form,
        memory_length,
        train_readout,
        epsilon,
        epsilon_schedule,
        epsilon_min,
        epsilon_decay,
        learning_rate,
        task]

        for episodes in hyperparams.get('episodes')
        for max_steps in hyperparams.get('max_steps')
        for fixed_memory in hyperparams.get('fixed_memory')
        for batch_size in hyperparams.get('batch_size')
        for gamma in hyperparams.get('gamma')
        for circuit_depth in hyperparams.get('circuit_depth')
        for encoding_depth in hyperparams.get('encoding_depth')
        for multiply_output_by in hyperparams.get('multiply_output_by')
        for update_after in hyperparams.get('update_after')
        for update_target_after in hyperparams.get('update_target_after')
        for vector_form in hyperparams.get('vector_form')
        for memory_length in hyperparams.get('memory_length')
        for train_readout in hyperparams.get('train_readout')
        for epsilon in hyperparams.get('epsilon')
        for epsilon_schedule in hyperparams.get('epsilon_schedule')
        for epsilon_min in hyperparams.get('epsilon_min')
        for epsilon_decay in hyperparams.get('epsilon_decay')
        for learning_rate in hyperparams.get('learning_rate')
        for task in hyperparams.get('task')
    )

    pool = multiprocessing.Pool()
    pool.map(_run_fl_q, experiments)
    pool.close()
    pool.join()
    return None


def _run_fl_q(args):
    (
        hyperparams,
        path,
        episodes,
        max_steps,
        fixed_memory,
        batch_size,
        gamma,
        circuit_depth,
        encoding_depth,
        multiply_output_by,
        update_after,
        update_target_after,
        vector_form,
        memory_length,
        train_readout,
        epsilon,
        epsilon_schedule,
        epsilon_min,
        epsilon_decay,
        learning_rate,
        task) = args

    save = hyperparams.get('save', True)
    save_as = hyperparams.get('save_as')
    test = hyperparams.get('test', False)

    if save and save_as is None:
        timestamp = time.localtime()
        save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(random.randint(0, 1000))

    if test:
        save_as = 'dummy'

    hps = {
        'episodes': episodes,
        'max_steps': max_steps,
        'fixed_memory': fixed_memory,
        'batch_size': batch_size,
        'gamma': gamma,
        'circuit_depth': circuit_depth,
        'encoding_depth': encoding_depth,
        'multiply_output_by': multiply_output_by,
        'update_after': update_after,
        'update_target_after': update_target_after,
        'vector_form': vector_form,
        'memory_length': memory_length,
        'train_readout': train_readout,
        'epsilon': epsilon,
        'epsilon_schedule': epsilon_schedule,
        'epsilon_min': epsilon_min,
        'epsilon_decay': epsilon_decay,
        'learning_rate': learning_rate,
        'task': task
    }

    slippery = hyperparams.get('slippery', False)

    flq = QLearningFL(
        hyperparams=hps,
        env_name=Envs.FROZENLAKE,
        save=save,
        save_as=save_as,
        path=path,
        slippery=slippery,
        test=test)

    flq.perform_episodes()


def parallelize_cp_q(hyperparams, path=BASE_PATH):
    experiments = ([
        hyperparams,
        path,
        episodes,
        batch_size,
        gamma,
        circuit_depth,
        update_after,
        update_target_after,
        epsilon,
        epsilon_schedule,
        epsilon_min,
        epsilon_decay,
        learning_rate,
        learning_rate_in,
        learning_rate_out]

        for episodes in hyperparams.get('episodes')
        for batch_size in hyperparams.get('batch_size')
        for gamma in hyperparams.get('gamma')
        for circuit_depth in hyperparams.get('circuit_depth')
        for update_after in hyperparams.get('update_after')
        for update_target_after in hyperparams.get('update_target_after')
        for epsilon in hyperparams.get('epsilon')
        for epsilon_schedule in hyperparams.get('epsilon_schedule')
        for epsilon_min in hyperparams.get('epsilon_min')
        for epsilon_decay in hyperparams.get('epsilon_decay')
        for learning_rate in hyperparams.get('learning_rate')
        for learning_rate_in in hyperparams.get('learning_rate_in')
        for learning_rate_out in hyperparams.get('learning_rate_out')
    )

    # pool = multiprocessing.Pool()
    # pool.map(_run_cp_q, experiments)
    # pool.close()
    # pool.join()

    _run_cp_q(list(experiments)[0])

    return None


def _run_cp_q(args):
    (
        hyperparams,
        path,
        episodes,
        batch_size,
        gamma,
        circuit_depth,
        update_after,
        update_target_after,
        epsilon,
        epsilon_schedule,
        epsilon_min,
        epsilon_decay,
        learning_rate,
        learning_rate_in,
        learning_rate_out) = args

    save = hyperparams.get('save', True)
    save_as = hyperparams.get('save_as')
    test = hyperparams.get('test', False)

    if save_as is None:
        timestamp = time.localtime()
        save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(random.randint(0, 1000))

    if test:
        save_as = 'dummy'

    hps = {
        'episodes': episodes,
        'batch_size': batch_size,
        'gamma': gamma,
        'circuit_depth': circuit_depth,
        'update_after': update_after,
        'update_target_after': update_target_after,
        'epsilon': epsilon,
        'epsilon_schedule': epsilon_schedule,
        'epsilon_min': epsilon_min,
        'epsilon_decay': epsilon_decay,
        'learning_rate': learning_rate,
        'learning_rate_in': learning_rate_in,
        'learning_rate_out': learning_rate_out,
        'use_negative_rewards': hyperparams.get('use_negative_rewards', False)
    }

    for i in range(hyperparams.get('reps', 1)):
        save_as_instance = copy(save_as)
        if hyperparams.get('reps', 1) > 1:
            save_as_instance += f'_{i}'

        cpq = QLearningCartpole(
            hyperparams=hps,
            env_name=Envs.CARTPOLE,
            save=save,
            save_as=save_as_instance,
            path=path,
            test=test)

        cpq.perform_episodes()


def parallelize_cp_c(hyperparams, path=BASE_PATH):
    experiments = ([
        hyperparams,
        path,
        episodes,
        batch_size,
        gamma,
        update_after,
        update_target_after,
        epsilon,
        epsilon_schedule,
        epsilon_min,
        epsilon_decay,
        learning_rate,
        n_hidden_layers,
        hidden_layer_config]

        for episodes in hyperparams.get('episodes')
        for batch_size in hyperparams.get('batch_size')
        for gamma in hyperparams.get('gamma')
        for update_after in hyperparams.get('update_after')
        for update_target_after in hyperparams.get('update_target_after')
        for epsilon in hyperparams.get('epsilon')
        for epsilon_schedule in hyperparams.get('epsilon_schedule')
        for epsilon_min in hyperparams.get('epsilon_min')
        for epsilon_decay in hyperparams.get('epsilon_decay')
        for learning_rate in hyperparams.get('learning_rate')
        for n_hidden_layers in hyperparams.get('n_hidden_layers')
        for hidden_layer_config in hyperparams.get('hidden_layer_config')
    )

    # pool = multiprocessing.Pool()
    # pool.map(_run_cp_q, experiments)
    # pool.close()
    # pool.join()

    _run_cp_c(list(experiments)[0])

    return None


def _run_cp_c(args):
    (
        hyperparams,
        path,
        episodes,
        batch_size,
        gamma,
        update_after,
        update_target_after,
        epsilon,
        epsilon_schedule,
        epsilon_min,
        epsilon_decay,
        learning_rate,
        n_hidden_layers,
        hidden_layer_config) = args

    save = hyperparams.get('save', True)
    save_as = hyperparams.get('save_as')
    test = hyperparams.get('test', False)

    if save_as is None:
        timestamp = time.localtime()
        save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(random.randint(0, 1000))

    if test:
        save_as = 'dummy'

    hps = {
        'episodes': episodes,
        'batch_size': batch_size,
        'gamma': gamma,
        'update_after': update_after,
        'update_target_after': update_target_after,
        'epsilon': epsilon,
        'epsilon_schedule': epsilon_schedule,
        'epsilon_min': epsilon_min,
        'epsilon_decay': epsilon_decay,
        'learning_rate': learning_rate,
        'n_hidden_layers': n_hidden_layers,
        'hidden_layer_config': hidden_layer_config,
        'use_negative_rewards': hyperparams.get('use_negative_rewards', False)
    }

    for i in range(hyperparams.get('reps', 1)):
        save_as_instance = copy(save_as)
        if hyperparams.get('reps', 1) > 1:
            save_as_instance += f'_{i}'

        cpc = QLearningCartpoleClassical(
            hyperparams=hps,
            env_name=Envs.CARTPOLE,
            save=save,
            save_as=save_as_instance,
            path=path,
            test=test)

        cpc.perform_episodes()
