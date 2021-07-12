
import os
import pickle
from pprint import pprint
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import sem

from config import BASE_PATH


def plot_avg_score():
    ...


def plot_val_by_dir(val):
    # path = '../../../../' + BASE_PATH + 'cartpole/no_arctan/'
    # path = '/home/andrea/BAK/vql/' + BASE_PATH + 'cartpole/cirq/train_weights/'
    path = '../../../../' + BASE_PATH + 'cartpole/fixed_range_90/'

    for file_name in os.listdir(path):
        if file_name[-13:] == 'scores.pickle':
            try:
                with open(path + file_name.replace('scores', 'meta'), 'rb') as file:
                    meta = pickle.load(file)
                    pprint(meta)

                learning_rate = meta.get('learning_rate')
                update_after = meta.get('update_after')
                update_target_after = meta.get('update_target_after')
                # epsilon_schedule = meta.get('epsilon_schedule')
                # memory_length = meta.get('memory_length')
                # batch_size = meta.get('batch_size')
                # data_reuploading = meta.get('data_reuploading')
                # circuit_depth = meta.get('n_layers')
                #
                l1u = meta.get('l1_units')
                l2u = meta.get('l2_units')

                # if str(meta.get('env_solved_at')) != '[]':
                #     pprint(meta)
                #     print(file_name)

                if True:
                    with open(path + file_name.replace('scores', val), 'rb') as file:
                        data = pickle.load(file)
                        # print(file_name)
                        # print(data)

                        plt.plot(data)
                        plt.title("Solved at episode {}".format(str(meta.get('env_solved_at')).replace('[]', '(not solved)')))
                        plt.xlabel('Episode')
                        plt.ylabel('Score')
                        plt.show()

                    # for dn in ['scaling']:  # ['loss', 'params', 'weights']:
                    #     with open(path + file_name.replace('scores', dn), 'rb') as file:
                    #             data = pickle.load(file)
                    #             print(data)
                    #             plt.plot(data)
                    #             plt.show()
            except Exception as e:
                print("Error with model", file_name)
                print(e)


def plot_avg_vals(val, min_val, avg_over, path, label, color, hyperparams, plot_to=None):
    all_vals = []
    for file_name in os.listdir(path):
        if file_name[-13:] == 'scores.pickle' and file_name[:5] != 'dummy':
            try:
                with open(path + file_name.replace('scores', 'meta'), 'rb') as file:
                    meta = pickle.load(file)
                    print(meta)

                include_agent = True
                for hp, value in hyperparams.items():
                    if meta.get(hp) != value:
                        include_agent = False
                        break

                if include_agent:
                    # print(meta)
                    with open(path + file_name.replace('scores', val), 'rb') as file:
                        data = pickle.load(file)
                        # pprint(meta['env_solved_at'])
                        # print(file_name)

                        if isinstance(data[0], list):
                            concatenated = []
                            for element in data:
                                concatenated += element
                            data = concatenated

                        filled_vals = np.ones(shape=min_val) * data[-1]
                        filled_vals[:len(data)] = data

                        if len(all_vals) < avg_over:
                            all_vals.append(filled_vals)
                        else:
                            break
            except Exception as e:
                print("Error in file", file_name)
                print(e)

    print(len(all_vals))

    clipped_vals = [x[:min_val] for x in all_vals]
    all_vals = clipped_vals
    mean_vals = np.mean(all_vals, axis=0)
    error = np.std(all_vals, axis=0)
    # error = sem(all_vals)

    fill_low = np.clip(np.asarray(mean_vals) - np.asarray(error), 0, None)
    fill_high = np.clip(np.asarray(mean_vals) + np.asarray(error), None, 200)

    sb.set_style("whitegrid")

    if plot_to is not None:
        sb.lineplot(list(range(plot_to)), mean_vals[:plot_to], color=color, label=label)
        plt.fill_between(range(plot_to), fill_low[:plot_to], fill_high[:plot_to], color=color, lw=0, alpha=0.3)
    else:
        sb.lineplot(list(range(len(mean_vals))), mean_vals, color=color, label=label)
        plt.fill_between(range(len(error)), fill_low, fill_high, color=color, lw=0, alpha=0.3)

    # plt.xlabel("Episode")
    # plt.ylabel("Score")
    # plt.ylim(ymax=200)
    # plt.show()


def plot_avg_score():
    path = '../../../../' + BASE_PATH + 'cartpole/hp_search/'

    for file_name in os.listdir(path):
        if file_name[-13:] == 'losses.pickle':
            with open(path + file_name.replace('losses', 'meta'), 'rb') as file:
                meta = pickle.load(file)
                pprint(meta)

            learning_rate = meta.get('learning_rate')
            update_after = meta.get('update_after')
            update_target_after = meta.get('update_target_after')
            epsilon_schedule = meta.get('epsilon_schedule')
            memory_length = meta.get('memory_length')
            batch_size = meta.get('batch_size')

            # if memory_length == 100000 and learning_rate == 0.0001 and update_after == 30 and update_target_after == 30:
            with open(path + file_name.replace('losses', 'scores'), 'rb') as file:
                data = pickle.load(file)
                # print(file_name)
                # print(data)

                average_scores = []
                window_size = 100
                num_wins = int(len(data)/window_size)
                for i in range(len(data)-window_size):
                    win_score = np.mean(data[i:i+window_size])
                    average_scores.append(win_score)

                plt.plot(average_scores)
                plt.ylim(ymax=200)
                plt.ylabel('Average score')
                plt.xlabel('Episode')
                plt.show()


def plot_by_model_name(model_name, val):
    path = '../../../../' + BASE_PATH + 'cartpole/'
    with open(path + model_name + '_' + val + '.pickle', 'rb') as file:
        data = pickle.load(file)
        print(data)

        # plt.plot(data)
        # plt.title("Final {}: {}".format(val, data[-1]))
        # plt.show()


# plot_val_by_dir('scores')
# exit()

# plot_avg_vals(
#     'scores', 5000, 10,
#     '../../../../' + BASE_PATH + 'cartpole/fixed_output/', 'fixed-range output', 'b',
#     {'data_reuploading': True, 'n_layers': 5, 'train_weights': False, 'train_data_scaling': True})

# plot_avg_vals(
#     'scores', 5000, 10,
#     '/home/andrea/BAK/vql/data/' + 'cartpole/cirq/good_hps/', 'w/o data re-uploading', 'royalblue',
#     {'data_reuploading': False, 'n_layers': 5, 'train_weights': True, 'train_data_scaling': True})

# plot_avg_vals(
#     'scores', 5000, 10,
#     '/home/andrea/BAK/vql/data/' + 'cartpole/cirq/good_hps/', 'w/ data re-uploading', 'g',
#     {'data_reuploading': True, 'n_layers': 5, 'train_weights': True, 'train_data_scaling': True})




# plot_avg_vals(
#     'scores', 5000, 10,
#     '/home/andrea/BAK/vql/data/' + 'cartpole/cirq/data_scaling_only/', 'fixed range [0, 1]', 'purple',
#     {'data_reuploading': True, 'n_layers': 5, 'train_weights': False, 'train_data_scaling': True})
#
# plot_avg_vals(
#     'scores', 5000, 10,
#     '../../../../' + BASE_PATH + 'cartpole/fixed_range_90/', 'fixed range [0, 90]', 'gold',
#     {'data_reuploading': True, 'n_layers': 5, 'train_weights': False, 'train_data_scaling': True})
#
# plot_avg_vals(
#     'scores', 5000, 10,
#     '/home/andrea/BAK/vql/data/' + 'cartpole/fixed_output/', 'fixed range [0, 180]', 'magenta',
#     {'data_reuploading': True, 'n_layers': 5, 'train_weights': False, 'train_data_scaling': True})
#
# plot_avg_vals(
#     'scores', 5000, 10,
#     '/home/andrea/BAK/vql/data/' + 'cartpole/cirq/train_weights/', 'w/o trainable scaling', 'crimson',
#     {'data_reuploading': None, 'n_layers': 5, 'train_weights': True, 'train_data_scaling': False})

# # plot_avg_vals(
# #     'scores', 5000, 100,
# #     '/home/andrea/BAK/vql/data/' + 'cartpole/cirq/train_weights/', 'output only, 10 layers', 'orange',
# #     {'data_reuploading': None, 'n_layers': 10, 'train_weights': True, 'train_data_scaling': False})
#
#
plot_avg_vals(
    'scores', 5000, 10,
    '/home/andrea/BAK/vql/data/' + 'cartpole/cirq/good_hps/', '56 params', 'g',  # both, 56 params, PQC, 56 params
    {'data_reuploading': True, 'n_layers': 5, 'train_weights': True, 'train_data_scaling': True})


# # plot_avg_vals(
# #     'scores', 5000, 10,
# #     '../../../../' + BASE_PATH + 'cartpole/cirq/train_weights/', '5 layers', 'g',
# #     {'data_reuploading': None, 'n_layers': 5, 'train_weights': True, 'train_data_scaling': False})
# #
# # plot_avg_vals(
# #     'scores', 5000, 10,
# #     '../../../../' + BASE_PATH + 'cartpole/cirq/train_weights/', '10 layers', 'b',
# #     {'data_reuploading': None, 'n_layers': 10, 'train_weights': True, 'train_data_scaling': False})
#
#
#
#
# plot_avg_vals(
#     'scores', 5000, 10,
#     '/home/andrea/BAK/vql/data/' + 'cartpole_classical/equal_params/', 'NN, 57 params', 'orange',
#     {'l1_units': 4, 'l2_units': 5})
#
# plot_avg_vals(
#     'scores', 5000, 10,
#     '/home/andrea/BAK/vql/data/' + 'cartpole_classical/equal_params/', 'NN, 75  params', 'orange',
#     {'l1_units': 5, 'l2_units': 6})

# plot_avg_vals(
#     'scores', 5000, 10,
#     '../../../../' + BASE_PATH + 'cartpole_classical/hp_search/', 'NN, 57 params', 'orange',
#     {'l1_units': 4, 'l2_units': 5, 'update_target_after': 10, 'update_after': 5, 'batch_size': 16, 'learning_rate': 0.01})

# plot_avg_vals(
#     'scores', 5000, 10,
#     '../../../../' + BASE_PATH + 'cartpole_classical/real_rewards/', 'NN, 75  params', 'darkblue',
#     {'l1_units': 5, 'l2_units': 6})
#
# plot_avg_vals(
#     'scores', 5000, 10,
#     '../../../../' + BASE_PATH + 'cartpole_classical/real_rewards/', 'NN, 95  params', 'orange',
#     {'l1_units': 6, 'l2_units': 7})

# plot_avg_vals(
#     'scores', 5000, 10,
#     '../../../../' + BASE_PATH + 'cartpole_classical/hp_search/', 'NN, 167  params', 'orchid',
#     {'l1_units': 9, 'l2_units': 10, 'update_target_after': 10, 'update_after': 5, 'batch_size': 16,
#      'learning_rate': 0.01})
#
# plot_avg_vals(
#     'scores', 5000, 10,
#     '../../../../' + BASE_PATH + 'cartpole_classical/softmax/', 'NN, softmax', 'slategrey',
#     {'l1_units': 9, 'l2_units': 10, 'update_target_after': 10, 'update_after': 5, 'batch_size': 16,
#      'learning_rate': 0.01})

# plot_avg_vals(
#     'scores', 5000, 100,
#     '../../../../' + BASE_PATH + 'cartpole_classical/hp_search/', 'NN', 'red',
#     {'l1_units': 8, 'l2_units': 9, 'update_target_after': 50, 'update_after': 5, 'batch_size': 16, 'learning_rate': 0.001})

# plot_avg_vals(
#     'scores', 5000, 10,
#     '../../../../' + BASE_PATH + 'cartpole_classical/hp_search/', 'NN', 'red',
#     {'l1_units': 9, 'l2_units': 10, 'update_target_after': 20, 'update_after': 5, 'batch_size': 16, 'learning_rate': 0.01})


# ############# deeper circuits #############################
plot_avg_vals(
    'scores', 5000, 10,
    '../../../../' + BASE_PATH + 'cartpole/depth_10/', 'depth 10', 'b',  # both, 56 params, PQC, 56 params
    {'circuit_depth': 10})

plot_avg_vals(
    'scores', 5000, 10,
    '../../../../' + BASE_PATH + 'cartpole/depth_10_orig/', 'depth 10', 'r',  # both, 56 params, PQC, 56 params
    {'circuit_depth': 10})


plt.xlabel("Episode")
plt.ylabel("Score")
# plt.title("CartPole v0, averaged over 10 agents each")
# plt.ylim(ymax=200)
plt.legend()  # loc='lower right')
plt.show()

# plot_avg_score()
# plot_by_model_name('2021-01-11_11-47-49_329', 'states')
