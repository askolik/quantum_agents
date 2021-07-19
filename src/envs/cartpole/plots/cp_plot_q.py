
import os
import pickle
from pprint import pprint
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import sem

from config import BASE_PATH
from src.utils.plots import plot_avg_vals


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


bak_path = '/home/andrea/BAK/vql/data/'
path = '../../../../' + BASE_PATH
plot_to = 3000


plot_avg_vals(
    'scores', 5000, 10,
    '/home/andrea/BAK/vql/data/' + 'cartpole/cirq/good_hps/', '5 layers', 'g',
    {'data_reuploading': True, 'n_layers': 5, 'train_weights': True, 'train_data_scaling': True}, plot_to=plot_to)

depth = 10
plot_avg_vals(
    'scores', 5000, 10,
    bak_path + f'cartpole/depth_{depth}_mse/', '10 layers', 'b',
    {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
     'batch_size': 64}, plot_to=plot_to)

depth = 15
plot_avg_vals(
    'scores', 5000, 10,
    bak_path + f'cartpole/depth_{depth}_mse/', '15 layers', 'orange',
    {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
     'batch_size': 32}, plot_to=plot_to)

depth = 20
plot_avg_vals(
    'scores', 5000, 10,
    bak_path + f'cartpole/depth_{depth}_mse/', '20 layers', 'red',
    {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
     'batch_size': 16}, plot_to=plot_to)


plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("PQCs with varying depth (preliminary)")
# plt.ylim(ymax=200)
plt.legend()  # loc='lower right')
plt.show()

# plot_avg_score()
# plot_by_model_name('2021-01-11_11-47-49_329', 'states')
