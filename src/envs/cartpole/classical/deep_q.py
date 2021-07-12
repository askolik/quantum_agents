import gym
import sys
import keras
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from random import shuffle

import time
import tensorflow as tf
from collections import deque

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


class Agent():
    def __init__(
            self,
            obs_space,
            action_space,
            l1_units=10,
            l2_units=0,
            learning_rate=0.001,
            batch_size=16,
            update_after=25,
            update_target_after=50):

        self.obs_space = obs_space
        self.action_space = action_space
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.batch_size = batch_size
        self.update_after = update_after
        self.update_target_after = update_target_after
        self.learning_rate = learning_rate
        self.model, self.target_model = self._build_model(l1_units, l2_units)

        self.training_history = []

    def remember(self, observation, action, reward, next_observation):
        if len(self.memory) > self.memory.maxlen:
            if np.random.random() < 0.5:
                shuffle(self.memory)
            self.memory.popleft()
        self.memory.append((observation, action, reward, next_observation))

    def get_q(self, observation):
        np_obs = np.reshape(observation, [-1, self.obs_space])
        return self.model.predict(np_obs)

    def get_q_target(self, observation):
        np_obs = np.reshape(observation, [-1, self.obs_space])
        return self.target_model.predict(np_obs)

    def _build_model(self, l1_units, l2_units):
        print(l1_units, l2_units)

        model = Sequential()

        model.add(Dense(l1_units, input_shape=(self.obs_space,), activation='relu'))

        if l2_units > 0:
            model.add(Dense(l2_units, activation='relu'))

        model.add(Dense(self.action_space, activation='linear'))

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse', metrics=[])

        model.build()
        weights = model.get_weights()
        print(model.summary())

        target_model = Sequential()
        target_model.add(Dense(l1_units, input_shape=(self.obs_space,), activation='relu'))

        if l2_units > 0:
            target_model.add(Dense(l2_units, activation='relu'))

        target_model.add(Dense(self.action_space, activation='linear'))

        target_model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='mse', metrics=[])
        target_model.build()
        target_model.set_weights(weights)

        return model, target_model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_action(self, action_model, target_model):
        sample_transitions = random.sample(self.memory, self.batch_size)
        random.shuffle(sample_transitions)
        batch_observations = []
        batch_targets = []

        for old_observation, action, reward, observation in sample_transitions:
            # Reshape targets to output dimension(=2)
            targets = np.reshape(
                self.get_q_target(old_observation),
                self.action_space)
            targets[action] = reward  # Set Target Value
            if observation is not None:
                # If the old state is not a final state, also consider the
                # discounted future reward
                predictions = self.get_q_target(observation)
                new_action = np.argmax(predictions)
                targets[action] += self.gamma * predictions[0, new_action]

            # print("Targets:", targets)

            # Add Old State to observations batch
            batch_observations.append(old_observation)
            batch_targets.append(targets)  # Add target to targets batch

        # Update the model using Observations and their corresponding Targets
        np_obs = np.reshape(batch_observations, [-1, self.obs_space])
        np_targets = np.reshape(batch_targets, [-1, self.action_space])
        history = self.model.fit(np_obs, np_targets, epochs=1, verbose=0)
        return history

    def save(self, path):
        pass #self.model.save_weights(path)

    def load(self, path):
        pass #self.model.load_weights(path)


def train(l1_units, l2_units, learning_rate=None, batch_size=None, update_after=None, update_target_after=None):
    env = gym.make('CartPole-v0')
    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]
    agent = Agent(
        observation_space, action_space, l1_units, l2_units,
        learning_rate, batch_size, update_after, update_target_after)

    episodes = 5000  # Games played in training phase
    max_steps = 200
    epsilon = 1
    epsilon_decay = 0.99
    epsilon_min = 0.01
    scores = []  # A list of all game scores
    recent_scores = []  # List that hold most recent 100 game scores

    meta = {
        'l1_units': l1_units,
        'l2_units': l2_units,
        'num_params': agent.model.count_params(),
        'update_target_after': update_target_after,
        'epsilon_schedule': 'fast',
        'batch_size': agent.batch_size,
        'learning_rate': agent.learning_rate,
        'update_after': agent.update_after,
        'update_target_after': agent.update_target_after
    }

    history = None

    for episode in range(episodes):
        observation = env.reset()
        for iteration in range(max_steps):
            old_observation = observation

            if np.random.random() < epsilon:
                # Take random action (explore)
                action = np.random.choice(range(action_space))
            else:
                # Query the model and get Q-values for possible actions
                q_values = agent.get_q(observation)
                action = np.argmax(q_values)
            # Take the selected action and observe next state
            observation, reward, done, _ = env.step(action)
            if done:
                scores.append(iteration + 1)  # Append final score

                if history:
                    agent.training_history.append(history.history['loss'][0])
                # Calculate recent scores
                if len(scores) > 100:
                    recent_scores = scores[-100:]
                # Print end-of-game information
                print(
                    "\rEpisode {:03d} , epsilon = {:.4f}, score = {:03d}".format(
                        episode, epsilon, iteration))  # , end="")
                agent.remember(old_observation, action, reward, None)
                break

            agent.remember(old_observation, action, reward, observation)

            if len(agent.memory) >= agent.batch_size and iteration % agent.update_after == 0:
                history = agent.update_action(agent.model, agent.model)

            if iteration % agent.update_target_after == 0:
                agent.update_target_network()

        if np.mean(recent_scores) > 195 and iteration > 195:
            print("\nEnvironment solved in {} episodes.".format(episode), end="")
            break
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

    # Saving the model

    # path = '../../../../data/cartpole_classical/softmax/'
    # timestamp = time.localtime()
    # save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(random.randint(0, 1000))
    # agent.model.save(path + '{}_model.h5'.format(save_as))
    #
    # with open(path + '{}_meta.pickle'.format(save_as), 'wb') as file:
    #     pickle.dump(meta, file)
    #
    # with open(path + '{}_scores.pickle'.format(save_as), 'wb') as file:
    #     pickle.dump(scores, file)
    #
    # with open(path + '{}_loss.pickle'.format(save_as), 'wb') as file:
    #     pickle.dump(agent.training_history, file)

    # plt.plot(agent.training_history, label='Loss')
    #
    # # # Plotting
    # plt.plot(scores, label='Scores')
    # # plt.title('Training Phase')
    # # plt.ylabel('Time Steps')
    # # plt.ylim(ymax=510)
    # plt.xlabel('Trial')
    # plt.legend()
    # # # plt.savefig('results/CartPoleTraining.png', bbox_inches='tight')
    # plt.show()

    return agent


def test(agent):
    env = gym.make("CartPole-v1")
    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]
    # agent = Agent(observation_space, action_space)
    # agent.load("models/cartpole_model.h5")
    scores = []

    # Playing 100 games
    for _ in range(100):
        obs = env.reset()
        episode_reward = 0
        while True:
            q_values = agent.get_q(obs)
            # print("Q values:", q_values)
            action = np.argmax(q_values)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        scores.append(episode_reward)

    # Plot the Performance
    plt.plot(scores)
    plt.title('Testing Phase')
    plt.ylabel('Time Steps')
    plt.ylim(ymax=510)
    plt.xlabel('Trial')
    # plt.savefig('results/CartPoleTesting.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # l1_units_list = [5]
    # l2_units_list = [0, 2, 3, 4]
    # for i in range(5):
    #     for l1_units in l1_units_list:
    #         for l2_units in l2_units_list:
    #             agent = train(l1_units, l2_units)
    # for i in range(5):
    #     for lr in [0.01, 0.001]:
    #         for batch_size in [16]:
    #             for update_after in [5, 10]:
    #                 for update_target_after in [10, 20, 50]:
    #                     train(4, 5, lr, batch_size, update_after, update_target_after)

    for i in range(1):
        train(9, 10, learning_rate=0.01, batch_size=16, update_after=5, update_target_after=10)
