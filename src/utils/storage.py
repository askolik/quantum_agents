
import pickle
import os
import matplotlib.pyplot as plt

from config import BASE_PATH


def save_data(save_as, meta, model, title, scores, loss_history, epsilons, path):
    with open(path + '{}_meta.pickle'.format(save_as), 'wb') as file:
        pickle.dump(meta, file)

    with open(path + '{}_scores.pickle'.format(save_as), 'wb') as file:
        pickle.dump(scores, file)

    with open(path + '{}_losses.pickle'.format(save_as), 'wb') as file:
        pickle.dump(loss_history, file)

    with open(path + '{}_epsilons.pickle'.format(save_as), 'wb') as file:
        pickle.dump(epsilons, file)

    model.save_weights(path + '{}_model.h5'.format(save_as))


def load_file(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def save_file(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def clean_after_test(path, file_name='dummy'):
    dirs = ['losses', 'meta', 'scores', 'epsilons']
    for dir in dirs:
        os.remove(path + dir + '/{}.pickle'.format(file_name))
    os.remove(path + 'weights/{}_model.h5'.format(file_name))
    os.remove(path + 'figures/{}.png'.format(file_name))
    os.remove(path + 'figures/{}_loss.png'.format(file_name))
