import random
import copy
import pickle
from collections import deque, namedtuple

import cirq
import gym
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

from config import Envs, BASE_PATH
from src.envs.frozenlake.fl_create_opt_training_data import get_all_transitions
from src.quantum.model import q_val, state_to_circuit, perform_action, add_to_memory, create_q_circuit, build_models, \
    generate_circuit, generate_model, empty_circuits
from src.utils.storage import save_data
from src.utils.utils import get_frozen_lake_true_q_vals


class QLearning:
    def __init__(self, hyperparams, env_name, save=True, save_as=None, path=BASE_PATH, test=False):
        self.env_name = env_name
        self.save = save
        self.save_as = save_as
        self.path = path
        self.test = test

        self.episodes = hyperparams.get('episodes')
        self.max_steps = hyperparams.get('max_steps')
        self.fixed_memory = hyperparams.get('fixed_memory')
        self.batch_size = hyperparams.get('batch_size')
        self.gamma = hyperparams.get('gamma')
        self.circuit_depth = hyperparams.get('circuit_depth')
        self.encoding_depth = hyperparams.get('encoding_depth')
        self.multiply_output_by = hyperparams.get('multiply_output_by')
        self.update_after = hyperparams.get('update_after')
        self.update_target_after = hyperparams.get('update_target_after')
        self.vector_form = hyperparams.get('vector_form')
        self.memory_length = hyperparams.get('memory_length')
        self.train_readout = hyperparams.get('train_readout')

        self.epsilon = hyperparams.get('epsilon')
        self.epsilon_schedule = hyperparams.get('epsilon_schedule')
        self.epsilon_min = hyperparams.get('epsilon_min')
        self.epsilon_decay = hyperparams.get('epsilon_decay')

        self.learning_rate = hyperparams.get('learning_rate')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, amsgrad=True)

        self.memory = self.initialize_memory()

    def initialize_memory(self):
        memory = deque(maxlen=self.memory_length)
        return memory

    def initialize_models(self):
        circuit, symbols = create_q_circuit(self.observation_space, self.circuit_depth)
        model, target_model = build_models(
            circuit, self.readout_op, self.optimizer, tf.keras.losses.mse, self.train_readout)

        return model, target_model, circuit, symbols

    def add_to_memory(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_length:
            if not self.fixed_memory:
                random_ix = np.random.randint(0, self.memory_length)
                self.memory[random_ix] = (state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state))

    def train_step(self, inputs, targets, action_masks):
        with tf.GradientTape() as tape:
            preds = self.model(inputs)
            scaled_preds = tf.divide(tf.add(preds, 1), 2)
            larger_vals = tf.multiply(
                scaled_preds, np.asarray(
                    [self.multiply_output_by for _ in range(self.action_space)]))

            if not self.vector_form:
                action_preds = tf.reduce_sum(larger_vals * action_masks, axis=1)
            else:
                action_preds = larger_vals

            loss = tf.losses.mse(targets, action_preds)
            if self.vector_form:
                loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss.numpy()

    def perform_action(self, state):
        action_type = 'random'
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            q = self.q_vals(state)
            # print("\tQ values:", q)
            action = np.argmax(q)
            action_type = 'argmax'

        return action, action_type

    def save_data(self, meta, scores, loss_history=None):
        with open(self.path + '{}_meta.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(meta, file)

        with open(self.path + '{}_scores.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(scores, file)

        if loss_history:
            with open(self.path + '{}_losses.pickle'.format(self.save_as), 'wb') as file:
                pickle.dump(loss_history, file)

        self.model.save_weights(self.path + '{}_model.h5'.format(self.save_as))

    def train_model(self):
        """
        Perform an update step of the model. Override in environment specific subclass.
        """
        pass

    def perform_episodes(self):
        """
        Perform episodes in environment. Override in environment specific subclass.
        """
        pass

    def q_vals(self, state):
        """
        Compute Q-values for state. Override in environment specific subclass.
        """
        pass


class QLearningFL(QLearning):
    def __init__(
            self,
            hyperparams,
            env_name,
            slippery=False,
            save=True,
            save_as=None,
            test=False,
            path=BASE_PATH):

        super(QLearningFL, self).__init__(hyperparams, env_name, save, save_as, path, test)

        self.env = gym.make(env_name.value, is_slippery=slippery)
        self.action_space = self.env.action_space.n
        self.observation_space = 4
        self.readout_op = self.initialize_readout()

        # 'regression' to use exact Q-values, 'q' for Bellman updates
        self.task = hyperparams.get('task')

        self.all_transitions = get_all_transitions()
        self.dead_states = [5, 7, 11, 12]

        self.model, self.target_model, self.circuit, self.symbols = self.initialize_models()

    def initialize_readout(self):
        qubits = [cirq.GridQubit(0, i) for i in range(self.observation_space)]
        readout_op = [cirq.PauliString(cirq.Z(qubit)) for qubit in qubits]
        return readout_op

    def compute_all_qvals(self):
        alive_states = [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]
        q_vals = []
        for state in alive_states:
            state_circ = state_to_circuit(state, self.encoding_depth, self.env_name)
            in_state = tfq.convert_to_tensor([state_circ])
            model_prediction = self.model(in_state)
            scaled_preds = tf.divide(tf.add(model_prediction, 1), 2)
            action_preds = tf.multiply(
                scaled_preds,
                np.asarray([
                    self.multiply_output_by,
                    self.multiply_output_by,
                    self.multiply_output_by,
                    self.multiply_output_by]))

            q_vals.append(list(action_preds.numpy()[0]))

        return q_vals

    def q_vals(self, state):
        state_circ = state_to_circuit(state, self.encoding_depth, self.env_name)
        in_state = tfq.convert_to_tensor([state_circ])
        model_prediction = self.model(in_state)
        scaled_preds = tf.divide(tf.add(model_prediction, 1), 2)
        action_preds = tf.multiply(
            scaled_preds,
            np.asarray([
                self.multiply_output_by,
                self.multiply_output_by,
                self.multiply_output_by,
                self.multiply_output_by]))

        output = action_preds.numpy()[0]

        # params = self.model.trainable_variables
        # state_circ = state_to_circuit(state, self.encoding_depth, self.env_name)
        # sample_circuit = state_circ + self.circuit
        #
        # expectation_layer = tfq.layers.Expectation()
        # expectation_output = expectation_layer(
        #     sample_circuit, symbol_names=self.symbols,
        #     symbol_values=params, operators=self.readout_op)
        #
        # output = ((expectation_output.numpy()[0] + 1) / 2) * self.multiply_output_by
        return output

    def save_env_data(self, transition_history, state_history, qval_history):
        with open(self.path + '{}_qvals.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(qval_history, file)

        with open(self.path + '{}_transitions.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(transition_history, file)

        with open(self.path + '{}_states.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(state_history, file)

    def train_model(self):
        samples = random.sample(self.memory, self.batch_size)

        batch_states = []
        batch_targets = []
        action_masks = []
        action_matrix = np.eye(self.action_space)
        train_transitions = []

        if self.task == 'regression':
            alive_states = [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]
            for state in alive_states:
                batch_states.append(
                    state_to_circuit(state, self.encoding_depth, env_name=self.env_name))
                batch_targets.append(get_frozen_lake_true_q_vals(state, self.gamma))
        elif self.task == 'q':
            for old_state, action, reward, state in samples:
                try:
                    train_transitions.append(self.all_transitions.index((old_state, action, reward, state)))
                except ValueError:
                    train_transitions.append(None)

                old_q_vals = self.q_vals(old_state)
                if not self.vector_form:
                    target = reward
                else:
                    target = copy.deepcopy(old_q_vals)
                    target[action] = reward

                if state not in self.dead_states:
                    q_vals = self.q_vals(state)
                    next_action = np.argmax(q_vals)

                    if not self.vector_form:
                        target += self.gamma * q_vals[next_action]
                    else:
                        target[action] += self.gamma * q_vals[next_action]

                batch_states.append(
                    state_to_circuit(old_state, self.encoding_depth, env_name=self.env_name))
                batch_targets.append(target)

                if not self.vector_form:
                    action_masks.append(action_matrix[action])

        states_tensor = tfq.convert_to_tensor(batch_states)
        targets_tensor = np.array(batch_targets)
        step_loss = self.train_step(states_tensor, targets_tensor, action_masks)

        return step_loss, train_transitions

    def perform_episodes(self):
        meta = {
            'episodes': self.episodes,
            'max_steps': self.max_steps,
            'fixed_memory': self.fixed_memory,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'circuit_depth': self.circuit_depth,
            'encoding_depth': self.encoding_depth,
            'multiply_output_by': self.multiply_output_by,
            'update_after': self.update_after,
            'update_target_after': self.update_target_after,
            'vector_form': self.vector_form,
            'memory_length': self.memory_length,
            'train_readout': self.train_readout,
            'last_epsilon': self.epsilon,
            'epsilon': self.epsilon,
            'epsilon_schedule': self.epsilon_schedule,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'learning_rate': self.learning_rate,
            'env_solved_at': [],
            'task': self.task
        }

        scores = [0]
        train_loss_history = []
        transition_history = []
        state_history = []
        qval_history = []

        if self.epsilon_schedule == 'linear':
            eps_values = list(np.linspace(self.epsilon, self.epsilon_min, self.episodes)[::-1])

        solved = False
        for episode in range(self.episodes):
            episode_losses = []
            episode_transitions = []
            episode_states = []
            episode_qvals = []

            state = self.env.reset()

            if solved:
                break

            for iteration in range(self.max_steps):
                episode_states.append(state)
                old_state = state

                action, action_type = self.perform_action(state)
                state, reward, done, _ = self.env.step(action)

                self.add_to_memory(old_state, action, reward, state)

                if done:
                    # 1 if goal reached, 0 if died
                    scores.append(int(state == 15))
                    meta['last_epsilon'] = self.epsilon
                    state_history.append(episode_states)

                    if sum(scores[-100:]) >= 100:
                        meta['env_solved_at'].append(episode)
                        print("Environment solved in {} episodes!".format(episode + 1))
                        solved = True

                    break
                elif iteration == self.max_steps:
                    scores.append(0)
                    meta['last_epsilon'] = self.epsilon
                    state_history.append(episode_states)
                    break

                if len(self.memory) >= self.batch_size and iteration % self.update_after == 0:
                    train_loss, train_transitions = self.train_model()
                    episode_losses.append(train_loss)
                    episode_transitions.append(train_transitions)

                    qvals = self.compute_all_qvals()
                    episode_qvals.append(qvals)

                    if self.test:
                        print("\tIteration: {}, Loss: {}".format(iteration, train_loss))

                if iteration % self.update_target_after == 0:
                    self.target_model.set_weights(self.model.get_weights())

            train_loss_history.append(episode_losses)
            transition_history.append(episode_transitions)
            qval_history.append(episode_qvals)

            if self.test:
                print("Episode {}, score {}/{}".format(episode, sum(scores[-100:]), sum(scores)))

            if self.epsilon_schedule == 'fast':
                self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
            elif self.epsilon_schedule == 'linear':
                self.epsilon = eps_values.pop()

            if self.save:
                self.save_data(meta, scores, train_loss_history)
                self.save_env_data(transition_history, state_history, qval_history)

        if self.save:
            self.save_data(meta, scores, train_loss_history)
            self.save_env_data(transition_history, state_history, qval_history)


class QLearningCartpole(QLearning):
    def __init__(
            self,
            hyperparams,
            env_name,
            save=True,
            save_as=None,
            test=False,
            path=BASE_PATH):

        super(QLearningCartpole, self).__init__(hyperparams, env_name, save, save_as, path, test)

        self.env = gym.make(env_name.value)
        self.max_steps = 200
        self.action_space = self.env.action_space.n
        self.observation_space = self.env.observation_space.shape[0]
        self.readout_op = self.initialize_readout()
        self.qubits = [cirq.GridQubit(0, i) for i in range(self.observation_space)]

        self.interaction = namedtuple('interaction', ('state', 'action', 'reward', 'next_state', 'done'))

        self.learning_rate_in = hyperparams.get('learning_rate_in')
        self.learning_rate_out = hyperparams.get('learning_rate_out')
        self.optimizer_input = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate_in, amsgrad=True)
        self.optimizer_output = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate_out, amsgrad=True)
        self.loss_fun = tf.keras.losses.Huber()

        self.w_input, self.w_var, self.w_output = 1, 0, 2
        self.model, self.target_model, self.circuit = self.initialize_models()

        if ('lambdas' not in self.model.trainable_variables[self.w_input].name) or (
                'thetas' not in self.model.trainable_variables[self.w_var].name) or (
                'obs-weights' not in self.model.trainable_variables[self.w_output].name):
            raise ValueError("Wrong indexing of optimizers parameters.")

    def initialize_readout(self):
        qubits = [cirq.GridQubit(0, i) for i in range(self.observation_space)]
        return [
            cirq.Z(qubits[0]) * cirq.Z(qubits[1]),
            cirq.Z(qubits[2]) * cirq.Z(qubits[3])]

    def initialize_models(self):
        circuit, param_dim, param_symbols, input_symbols = self.create_circuit()
        model = generate_model(
            self.observation_space, self.circuit_depth, circuit,
            param_dim, param_symbols, input_symbols, self.readout_op, False)
        target_model = generate_model(
            self.observation_space, self.circuit_depth, circuit,
            param_dim, param_symbols, input_symbols, self.readout_op, True)
        target_model.set_weights(model.get_weights())

        return model, target_model, circuit

    def create_circuit(self):
        circuit, param_dim, param_symbols, input_symbols = generate_circuit(
            self.observation_space, self.circuit_depth, self.qubits)
        return circuit, param_dim, param_symbols, input_symbols

    def save_env_data(self, state_history):
        with open(self.path + '{}_states.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(state_history, file)

    def add_to_memory(self, state, action, reward, next_state, done):
        transition = self.interaction(
            state, action, reward, next_state, float(done))
        self.memory.append(transition)

    def perform_action(self, state):
        action_type = 'random'
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            q_vals = self.model([empty_circuits(1), state])
            action = int(tf.argmax(q_vals[0]).numpy())
            action_type = 'argmax'

        return action, action_type

    def train_step(self):
        batch = random.choices(self.memory, k=self.batch_size)
        batch = self.interaction(*zip(*batch))

        future_rewards = self.target_model.predict(
            [empty_circuits(self.batch_size), tf.constant(batch.next_state)])
        target_q_values = tf.constant(batch.reward) + self.gamma * tf.reduce_max(future_rewards, axis=1) * (
                1 - tf.constant(batch.done))
        masks = tf.one_hot(batch.action, self.action_space)

        with tf.GradientTape() as tape:
            q_values = self.model([empty_circuits(self.batch_size), tf.constant(batch.state)])
            q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self.loss_fun(target_q_values, q_values_masked)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer_input.apply_gradients(
            [(grads[self.w_input], self.model.trainable_variables[self.w_input])])
        self.optimizer.apply_gradients(
            [(grads[self.w_var], self.model.trainable_variables[self.w_var])])
        self.optimizer_output.apply_gradients(
            [(grads[self.w_output], self.model.trainable_variables[self.w_output])])

    # @tf.function
    def train_step_tutorial(self):
        # print("Update step")
        # it seems like for some reason the update step is only performed
        # twice at the beginning and never again when tf.function decorator is added?
        training_batch = random.choices(self.memory, k=self.batch_size)
        training_batch = self.interaction(*zip(*training_batch))

        states = np.asarray([x for x in training_batch.state])
        rewards = np.asarray([x for x in training_batch.reward], dtype=np.float32)
        next_states = np.asarray([x for x in training_batch.next_state])
        done = np.asarray([x for x in training_batch.done], dtype=np.float32)

        states = tf.convert_to_tensor(states)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        done = tf.convert_to_tensor(done)

        # Compute their target q_values and the masks on sampled actions
        future_rewards = self.target_model([empty_circuits(self.batch_size), next_states])
        target_q_values = rewards + (
                self.gamma * tf.reduce_max(future_rewards, axis=1) * (1.0 - done))
        masks = tf.one_hot(training_batch.action, self.action_space)

        # Train the model on the states and target Q-values
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            q_values = self.model([empty_circuits(self.batch_size), states])
            q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = tf.keras.losses.Huber()(target_q_values, q_values_masked)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        for optimizer, w in zip(
                [self.optimizer_input, self.optimizer, self.optimizer_output],
                [self.w_input, self.w_var, self.w_output]):
            optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])

    def perform_episodes(self):
        meta = {
            'episodes': self.episodes,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'circuit_depth': self.circuit_depth,
            'update_after': self.update_after,
            'update_target_after': self.update_target_after,
            'last_epsilon': self.epsilon,
            'epsilon': self.epsilon,
            'epsilon_schedule': self.epsilon_schedule,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'learning_rate': self.learning_rate,
            'learning_rate_in': self.learning_rate_in,
            'learning_rate_out': self.learning_rate_out,
            'env_solved_at': []
        }

        scores = []
        recent_scores = []

        if self.epsilon_schedule == 'linear':
            eps_values = list(np.linspace(self.epsilon, self.epsilon_min, self.episodes)[::-1])

        solved = False
        for episode in range(self.episodes):
            if solved:
                break

            state = self.env.reset()
            for iteration in range(self.max_steps):
                # self.env.render()

                old_state = state
                action, action_type = self.perform_action(state)
                # print("Action:", action)

                state, reward, done, _ = self.env.step(action)

                self.add_to_memory(old_state, action, reward, state, done)

                if done:
                    scores.append(iteration + 1)
                    meta['last_epsilon'] = self.epsilon

                    if len(scores) > 100:
                        recent_scores = scores[-100:]

                    avg_score = np.mean(recent_scores) if recent_scores else np.mean(scores)
                    print(
                        "\rEpisode {:03d} , epsilon={:.4f}, action type={}, score={:03d}, avg score={:.3f}".format(
                            episode, self.epsilon, action_type, iteration + 1, avg_score))

                    break

                if len(self.memory) >= self.batch_size and iteration % self.update_after == 0:
                    # print("Train step outer")
                    self.train_step()

                if iteration % self.update_target_after == 0:
                    self.target_model.set_weights(self.model.get_weights())

            if np.mean(recent_scores) >= 195:
                print("\nEnvironment solved in {} episodes.".format(episode), end="")
                meta['env_solved_at'].append(episode)
                solved = True

            if self.epsilon_schedule == 'fast':
                self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
            elif self.epsilon_schedule == 'linear':
                self.epsilon = eps_values.pop()

            if self.save:
                self.save_data(meta, scores)


class RegressionFL:
    def __init__(
            self,
            circuit_depth,
            encoding_depth,
            gamma,
            learning_rate,
            output_factor,
            save,
            save_as,
            test,
            path=BASE_PATH):

        self.circuit_depth = circuit_depth
        self.encoding_depth = encoding_depth
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.output_factor = output_factor

        self.model, self.optimizer = self.initialize_model()

        self.loss_history = []

        self.save = save
        self.save_as = save_as
        self.test = test
        self.path = path

    def initialize_model(self):
        observation_space = 4
        qubits = [cirq.GridQubit(0, i) for i in range(observation_space)]
        readout_op = [cirq.PauliString(cirq.Z(qubit)) for qubit in qubits]
        circuit, symbols = create_q_circuit(observation_space, self.circuit_depth)

        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model, target_model = build_models(circuit, readout_op, opt, tf.keras.losses.mse)

        return model, opt

    def regress(self, iterations):
        if self.save:
            meta = {
                'circuit_depth': self.circuit_depth,
                'encoding_depth': self.encoding_depth,
                'gamma': self.gamma,
                'learning_rate': self.learning_rate,
                'output_factor': self.output_factor
            }

            with open(self.path + '{}_meta.pickle'.format(self.save_as), 'wb') as file:
                pickle.dump(meta, file)

        for i in range(iterations):
            train_loss = self.train()
            self.loss_history.append(train_loss)

            if self.test:
                print("Iteration {}: {}".format(i, train_loss))

            if self.save:
                self.save_data()

    def train(self):
        alive_states = [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]
        batch_states = []
        batch_targets = []
        for old_state in alive_states:
            batch_states.append(state_to_circuit(old_state, self.encoding_depth, env_name=Envs.FROZENLAKE))
            batch_targets.append(get_frozen_lake_true_q_vals(old_state, self.gamma))

        # print("True Q values:", batch_targets)

        states_tensor = tfq.convert_to_tensor(batch_states)
        targets_tensor = np.array(batch_targets)
        step_loss = train_step_frozen_lake_regression(
            states_tensor, targets_tensor, self.model, self.optimizer, self.output_factor)

        return step_loss

    def save_data(self):
        with open(self.path + '{}_loss.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(self.loss_history, file)

        self.model.save_weights(self.path + '{}_model.h5'.format(self.save_as))


class RegressionCP:
    def __init__(
            self,
            circuit_depth,
            encoding_depth,
            gamma,
            batch_size,
            learning_rate,
            output_factor,
            enc_type,
            save,
            save_as,
            test,
            path=BASE_PATH):

        self.circuit_depth = circuit_depth
        self.encoding_depth = encoding_depth
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_factor = output_factor
        self.enc_type = enc_type

        self.model, self.optimizer = self.initialize_model()

        self.loss_history = []

        self.save = save
        self.save_as = save_as
        self.test = test
        self.path = path

        self.train_data = []

    def initialize_model(self):
        observation_space = 4
        qubits = [cirq.GridQubit(0, i) for i in range(observation_space)]
        readout_op = [
            cirq.PauliString(cirq.Z(qubits[0])*cirq.Z(qubits[1])),
            cirq.PauliString(cirq.Z(qubits[2])*cirq.Z(qubits[2]))]
        circuit, symbols = create_q_circuit(observation_space, self.circuit_depth)

        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model, target_model = build_models(circuit, readout_op, opt, tf.keras.losses.mse)

        return model, opt

    def load_data(self, path):
        with open(path, 'rb') as file:
            [states, q_vals] = pickle.load(file)

        for i in range(len(states)):
            self.train_data.append([states[i], q_vals[i]])

    def sample_batch(self):
        rnd_ixs = np.random.randint(0, len(self.train_data), size=self.batch_size)
        batch = []
        for ix in rnd_ixs:
            batch.append(self.train_data[ix])
        return batch

    def regress(self, iterations):
        if self.save:
            meta = {
                'circuit_depth': self.circuit_depth,
                'encoding_depth': self.encoding_depth,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'output_factor': self.output_factor
            }

            with open(self.path + '{}_meta.pickle'.format(self.save_as), 'wb') as file:
                pickle.dump(meta, file)

        for i in range(iterations):
            train_loss = self.train()
            self.loss_history.append(train_loss)

            if self.test:
                print("Iteration {}: {}".format(i, train_loss))

            if self.save:
                self.save_data()

    def train(self):
        train_batch = self.sample_batch()
        batch_states = []
        batch_targets = []
        for state, q_vals in train_batch:
            batch_states.append(
                state_to_circuit(
                    state, self.encoding_depth, env_name=Envs.CARTPOLE, enc_type=self.enc_type))
            batch_targets.append(q_vals)

        # print("True Q values:", batch_targets)

        states_tensor = tfq.convert_to_tensor(batch_states)
        targets_tensor = np.array(batch_targets)
        step_loss = self.train_step(states_tensor, targets_tensor)

        return step_loss

    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            preds = self.model(inputs)
            scaled_preds = tf.divide(tf.add(preds, 1), 2)
            action_preds = tf.multiply(
                scaled_preds,
                np.asarray([self.output_factor, self.output_factor]))

            loss = tf.reduce_mean(tf.losses.mse(targets, action_preds))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss.numpy()

    def save_data(self):
        with open(self.path + '{}_loss.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(self.loss_history, file)

        self.model.save_weights(self.path + '{}_model.h5'.format(self.save_as))


def train_model(
        model,
        target_model,
        batch_size,
        memory,
        circuit,
        symbols,
        ops,
        gamma,
        action_space,
        encoding_depth,
        opt,
        multiply_output_by,
        vector_form,
        alpha,
        env_name):
    if env_name == Envs.CARTPOLE:
        step_loss = train_model_cartpole(
            model,
            target_model,
            batch_size,
            memory,
            circuit,
            symbols,
            ops,
            gamma,
            action_space,
            encoding_depth,
            opt,
            multiply_output_by,
            vector_form,
            alpha,
            env_name)

    elif env_name == Envs.FROZENLAKE:
        step_loss = train_model_frozenlake(
            model,
            target_model,
            batch_size,
            memory,
            circuit,
            symbols,
            ops,
            gamma,
            action_space,
            encoding_depth,
            opt,
            multiply_output_by,
            vector_form,
            alpha,
            env_name)

    return step_loss


def train_model_frozenlake(
        model,
        target_model,
        batch_size,
        memory,
        circuit,
        symbols,
        ops,
        gamma,
        action_space,
        encoding_depth,
        opt,
        multiply_output_by,
        vector_form,
        alpha,
        env_name):
    alive_states = [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]
    samples = random.sample(memory, batch_size)

    batch_states = []
    batch_targets = []
    action_masks = []
    action_matrix = np.eye(action_space)
    for old_state, action, reward, state in samples:
        if old_state in alive_states:
            old_q_vals = q_val(old_state, model, circuit, symbols, ops, encoding_depth, multiply_output_by, env_name)
            if not vector_form:
                target = reward
            else:
                target = copy.deepcopy(old_q_vals)
                target[action] = reward

            # add future reward for non-final states
            if state is not None:
                q_vals = q_val(state, target_model, circuit, symbols, ops, encoding_depth, multiply_output_by, env_name)
                next_action = np.argmax(q_vals)

                if not vector_form:
                    target += gamma * q_vals[next_action]
                else:
                    target[action] += gamma * q_vals[next_action]

            # print("Target:", target)
            batch_states.append(state_to_circuit(old_state, encoding_depth, env_name=env_name))
            batch_targets.append(target)

            if not vector_form:
                action_masks.append(action_matrix[action])

    states_tensor = tfq.convert_to_tensor(batch_states)
    targets_tensor = np.array(batch_targets)
    step_loss = train_step(
        states_tensor, targets_tensor, action_masks, model, opt, multiply_output_by, vector_form)

    # batch_states = []
    # batch_targets = []
    # for old_state in alive_states:
    #     batch_states.append(state_to_circuit(old_state, encoding_depth, env_name=env_name))
    #     batch_targets.append(get_frozen_lake_true_q_vals(old_state, gamma))
    #
    # # print("True Q values:", batch_targets)
    #
    # states_tensor = tfq.convert_to_tensor(batch_states)
    # targets_tensor = np.array(batch_targets)
    # step_loss = train_step_frozen_lake_regression(
    #     states_tensor, targets_tensor, model, opt, multiply_output_by)

    return step_loss


def train_model_cartpole(
        model,
        target_model,
        batch_size,
        memory,
        circuit,
        symbols,
        ops,
        gamma,
        action_space,
        encoding_depth,
        opt,
        multiply_output_by,
        vector_form,
        alpha,
        env_name):
    samples = random.sample(memory, batch_size)
    random.shuffle(samples)

    batch_states = []
    batch_targets = []
    action_masks = []
    action_matrix = np.eye(action_space)
    for old_state, action, reward, state in samples:
        old_q_vals = q_val(old_state, model, circuit, symbols, ops, encoding_depth, multiply_output_by, env_name)
        if not vector_form:
            target = reward
        else:
            target = copy.deepcopy(old_q_vals)
            target[action] = reward

        # add future reward for non-final states
        if state is not None:
            q_vals = q_val(state, target_model, circuit, symbols, ops, encoding_depth, multiply_output_by, env_name)
            next_action = np.argmax(q_vals)

            if not vector_form:
                target += gamma * q_vals[next_action]
            else:
                target[action] += gamma * q_vals[next_action]

        # print("Target:", target)
        batch_states.append(state_to_circuit(old_state, encoding_depth, env_name=env_name))
        batch_targets.append(target)

        if not vector_form:
            action_masks.append(action_matrix[action])

    states_tensor = tfq.convert_to_tensor(batch_states)
    targets_tensor = np.array(batch_targets)
    step_loss = train_step(
        states_tensor, targets_tensor, action_masks, model, opt, multiply_output_by, vector_form)

    return step_loss


def train_step(inputs, targets, action_masks, model, opt, multiply_output_by, vector_form):
    with tf.GradientTape() as tape:
        preds = model(inputs)
        scaled_preds = tf.divide(tf.add(preds, 1), 2)
        larger_vals = tf.multiply(
            scaled_preds, np.asarray(
                [multiply_output_by, multiply_output_by]))

        if not vector_form:
            action_preds = tf.reduce_sum(larger_vals * action_masks, axis=1)
        else:
            action_preds = larger_vals

        loss = tf.losses.mse(targets, action_preds)
        if vector_form:
            loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return loss.numpy()


def train_step_frozen_lake_regression(inputs, targets, model, opt, multiply_output_by):
    with tf.GradientTape() as tape:
        preds = model(inputs)
        scaled_preds = tf.divide(tf.add(preds, 1), 2)
        action_preds = tf.multiply(
            scaled_preds, np.asarray([multiply_output_by, multiply_output_by, multiply_output_by, multiply_output_by]))

        # print(action_preds)
        # print(targets)
        # print(tf.losses.mse(targets, action_preds))
        loss = tf.reduce_mean(tf.losses.mse(targets, action_preds))

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return loss.numpy()


def perform_episodes(
        episodes=None, max_steps=None, env=None, env_name=None, model=None, target_model=None,
        circuit=None, symbols=None, readout_op=None, epsilon=None, fixed_memory=None,
        batch_size=None, model_update_prob=None, gamma=None, action_space=None, encoding_depth=None,
        opt=None, multiply_output_by=None, update_target_after=None, epsilon_schedule=None,
        epsilon_min=None, epsilon_decay=None, save_as=None, meta=None, plot_title=None, path=None,
        vector_form=None, alpha=None, fixed_update_after=None, prev_scores=None, prev_losses=None,
        prev_epsilons=None, prev_memory=None, save_every=10):
    scores = prev_scores or [0]
    eps_schedule = list(np.linspace(epsilon, epsilon_min, episodes)[::-1])
    epsilons = prev_epsilons or []
    loss_history = prev_losses or []

    memory_len = 10000
    memory = deque(maxlen=memory_len)

    if prev_memory:
        with open(path + prev_memory, 'rb') as file:
            memory = pickle.load(file)

    if env_name == Envs.CARTPOLE:
        scores, loss_history, epsilons = perform_episodes_cartpole(
            episodes=episodes,
            max_steps=max_steps,
            env=env,
            env_name=env_name,
            model=model,
            target_model=target_model,
            circuit=circuit,
            symbols=symbols,
            readout_op=readout_op,
            epsilon=epsilon,
            fixed_memory=fixed_memory,
            batch_size=batch_size,
            model_update_prob=model_update_prob,
            gamma=gamma,
            action_space=action_space,
            encoding_depth=encoding_depth,
            opt=opt,
            multiply_output_by=multiply_output_by,
            update_target_after=update_target_after,
            epsilon_schedule=epsilon_schedule,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            save_as=save_as,
            meta=meta,
            plot_title=plot_title,
            path=path,
            vector_form=vector_form,
            alpha=alpha,
            fixed_update_after=fixed_update_after,
            save_every=save_every,
            scores=scores,
            epsilons=epsilons,
            loss_history=loss_history,
            memory_len=memory_len,
            eps_schedule=eps_schedule,
            memory=memory)

    elif env_name == Envs.FROZENLAKE:
        scores, loss_history, epsilons = perform_episodes_frozenlake(
            episodes=episodes,
            max_steps=max_steps,
            env=env,
            env_name=env_name,
            model=model,
            target_model=target_model,
            circuit=circuit,
            symbols=symbols,
            readout_op=readout_op,
            epsilon=epsilon,
            fixed_memory=fixed_memory,
            batch_size=batch_size,
            model_update_prob=model_update_prob,
            gamma=gamma,
            action_space=action_space,
            encoding_depth=encoding_depth,
            opt=opt,
            multiply_output_by=multiply_output_by,
            update_target_after=update_target_after,
            epsilon_schedule=epsilon_schedule,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            save_as=save_as,
            meta=meta,
            plot_title=plot_title,
            path=path,
            vector_form=vector_form,
            alpha=alpha,
            fixed_update_after=fixed_update_after,
            save_every=save_every,
            scores=scores,
            epsilons=epsilons,
            loss_history=loss_history,
            memory_len=memory_len,
            eps_schedule=eps_schedule,
            memory=memory)

    return scores, loss_history, epsilons


def perform_episodes_cartpole(
        episodes=None,
        max_steps=None,
        env=None,
        env_name=None,
        model=None,
        target_model=None,
        circuit=None,
        symbols=None,
        readout_op=None,
        epsilon=None,
        fixed_memory=None,
        batch_size=None,
        model_update_prob=None,
        gamma=None,
        action_space=None,
        encoding_depth=None,
        opt=None,
        multiply_output_by=None,
        update_target_after=None,
        epsilon_schedule=None,
        epsilon_min=None,
        epsilon_decay=None,
        save_as=None,
        meta=None,
        plot_title=None,
        path=None,
        vector_form=None,
        alpha=None,
        fixed_update_after=None,
        save_every=10,
        scores=[],
        epsilons=[],
        loss_history=[],
        memory_len=10000,
        eps_schedule=[],
        memory=None):
    recent_scores = [0]
    train_loss = None

    for episode in range(episodes):
        state = env.reset()
        for iteration in range(max_steps):
            old_state = state

            action, action_type = perform_action(
                state, model, circuit, symbols, readout_op, env,
                encoding_depth, multiply_output_by, epsilon, env_name, gamma)

            state, reward, done, _ = env.step(action)

            if done:
                scores.append(iteration + 1)
                epsilons.append(epsilon)

                if train_loss:
                    loss_history.append(train_loss)

                if len(scores) > 100:
                    recent_scores = scores[-100:]
                print(
                    "\rEpisode {:03d} , epsilon = {:.4f}, action type = {}, score = {:03d}".format(
                        episode, epsilon, action_type, iteration + 1))

                if iteration != 199:
                    reward = -1
                if iteration == 199:
                    reward = 2

                memory = add_to_memory(
                    memory, old_state, action, reward, None, memory_len, fixed_memory)
                break

            # Add the observation to replay memory
            memory = add_to_memory(
                memory, old_state, action, reward, state, memory_len, fixed_memory)

            train_now = False
            if fixed_update_after and iteration % fixed_update_after == 0:
                train_now = True
            elif model_update_prob and np.random.random() < model_update_prob:
                train_now = True

            if len(memory) >= batch_size and train_now and len(scores) >= 1 and scores[-1] < 200:
                train_loss = train_model(
                    model,
                    target_model,
                    batch_size,
                    memory,
                    circuit,
                    symbols,
                    readout_op,
                    gamma,
                    action_space,
                    encoding_depth,
                    opt,
                    multiply_output_by,
                    vector_form,
                    alpha,
                    env_name)

                print("\tIteration: {}, Loss: {}".format(iteration, train_loss))

            if iteration % update_target_after == 0:
                target_model.set_weights(model.get_weights())

        # If mean over the last 100 Games is >195, then success
        if np.mean(recent_scores) > 195 and iteration > 195:
            print("\nEnvironment solved in {} episodes.".format(episode), end="")
            break

        if epsilon_schedule == 'fast':
            epsilon = max(epsilon_min, epsilon_decay * epsilon)
        elif epsilon_schedule == 'linear':
            epsilon = eps_schedule.pop()

        if save_as and episode % save_every == 0:
            save_data(save_as, meta, model, plot_title, scores, loss_history, epsilons, path=path)

    return scores, loss_history, epsilons


def perform_regression_frozenlake(
        max_steps=None,
        env_name=None,
        model=None,
        target_model=None,
        circuit=None,
        symbols=None,
        readout_op=None,
        batch_size=None,
        gamma=None,
        action_space=None,
        encoding_depth=None,
        opt=None,
        multiply_output_by=None,
        save_as=None,
        meta=None,
        plot_title=None,
        path=None,
        vector_form=None,
        alpha=None,
        save_every=10,
        scores=[],
        epsilons=[],
        loss_history=[],
        memory=None):
    for i in range(max_steps):
        train_loss = train_model(
            model,
            target_model,
            batch_size,
            memory,
            circuit,
            symbols,
            readout_op,
            gamma,
            action_space,
            encoding_depth,
            opt,
            multiply_output_by,
            vector_form,
            alpha,
            env_name)

        loss_history.append(train_loss)
        print("Iteration {}: {}".format(i, train_loss))

        if save_every > 0:
            save_data(save_as, meta, model, plot_title, None, loss_history, epsilons, path=path)

    return scores, loss_history, epsilons


def perform_episodes_frozenlake(
        episodes=None, max_steps=None, env=None, env_name=None, model=None, target_model=None, circuit=None,
        symbols=None, readout_op=None, epsilon=None, fixed_memory=None, batch_size=None, model_update_prob=None,
        gamma=None, action_space=None, encoding_depth=None, opt=None, multiply_output_by=None,
        update_target_after=None, epsilon_schedule=None, epsilon_min=None, epsilon_decay=None, save_as=None,
        meta=None, plot_title=None, path=None, vector_form=None, alpha=None, fixed_update_after=None, save_every=10,
        scores=[], epsilons=[], loss_history=[], memory_len=10000, eps_schedule=[], memory=None):
    scores = [0]
    train_loss_history = []

    solved = False
    for episode in range(episodes):
        state = env.reset()

        if solved:
            break

        for iteration in range(max_steps):
            old_state = state

            action, action_type = perform_action(
                state, model, circuit, symbols, readout_op, env,
                encoding_depth, multiply_output_by, epsilon, env_name, gamma)

            state, reward, done, _ = env.step(action)

            print(old_state, action, state)

            memory = add_to_memory(
                memory, old_state, action, reward, state, memory_len, fixed_memory)

            if done:
                # 1 if goal reached, 0 if died
                scores.append(int(state == 15))

                if sum(scores[-100:]) >= 100:
                    meta['env_solved_at'].append(episode)
                    print("Environment solved in {} episodes!".format(episode + 1))
                    solved = True

                break
            elif iteration == max_steps:
                scores.append(0)
                break

            train_now = False
            if fixed_update_after and iteration % fixed_update_after == 0:
                train_now = True
            elif model_update_prob and np.random.random() < model_update_prob:
                train_now = True

            if len(memory) >= batch_size and train_now:
                train_loss = train_model(
                    model,
                    target_model,
                    batch_size,
                    memory,
                    circuit,
                    symbols,
                    readout_op,
                    gamma,
                    action_space,
                    encoding_depth,
                    opt,
                    multiply_output_by,
                    vector_form,
                    alpha,
                    env_name)

                train_loss_history.append(train_loss)

                print("\tIteration: {}, Loss: {}".format(iteration, train_loss))

            if iteration % update_target_after == 0:
                target_model.set_weights(model.get_weights())

        print("Episode {}, score {}/{}".format(episode, sum(scores[-100:]), sum(scores)))

        if epsilon_schedule == 'fast':
            epsilon = max(epsilon_min, epsilon_decay * epsilon)
        elif epsilon_schedule == 'linear':
            epsilon = eps_schedule.pop()

        if save_as and episode % save_every == 0:
            save_data(save_as, meta, model, plot_title, scores, loss_history, epsilons, path=path)

    return scores, loss_history, epsilons
