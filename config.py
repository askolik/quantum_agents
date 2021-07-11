from enum import Enum

BASE_PATH = 'data/'
ALICE_BASE_PATH = '/home/skolika/vql/data/'
BAK_PATH = 'C:/Users/VWEVU4W/Desktop/Promotion/Leiden/vql/data/'

FL_BATCHES_BASE_PATH = '../../../' + BASE_PATH + 'frozen_lake/fl_regression/'
FL_OPT_TAR_BASE_PATH = '../../../' + BASE_PATH + 'frozen_lake/fl_optimal_target/'
FL_OPT_DAT_BASE_PATH = '../../../' + BASE_PATH + 'frozen_lake/fl_optimal_data/'
FL_Q_LEARN_BASE_PATH = '../../../' + BASE_PATH + 'frozen_lake/fl_q_learning/'
FL_SMOOTH_BASE_PATH = '../../../' + BASE_PATH + 'frozen_lake/fl_smoothing/'

MC_REGRESSION = '../../../' + BASE_PATH + 'mountain_car/mc_regression/'


class Envs(Enum):
    CARTPOLE = 'CartPole-v0'
    FROZENLAKE = 'FrozenLake-v0'
    MOUNTAINCAR = 'MountainCar-v0'


class EncType(Enum):
    HIDDEN_SHIFT = 'hidden_shift'
    CONT_X = 'cont_x'
    CONT_YZ = 'cont_yz'
    BASIS = 'basis'
