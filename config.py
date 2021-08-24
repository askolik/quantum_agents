from enum import Enum

BASE_PATH = 'data/'


class Envs(Enum):
    CARTPOLE = 'CartPole-v0'
    FROZENLAKE = 'FrozenLake-v0'


class EncType(Enum):
    HIDDEN_SHIFT = 'hidden_shift'
    CONT_X = 'cont_x'
