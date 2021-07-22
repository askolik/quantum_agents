
import matplotlib.pyplot as plt
from config import BASE_PATH
from src.utils.plots import plot_avg_vals


bak_path = '/home/andrea/BAK/vql/data/'
path = '../../../../' + BASE_PATH
plot_to = 5000


# hps = {
#     'learning_rate': 0.001,
#     'update_after': 10, 'update_target_after': 20, 'batch_size': 64, 'use_negative_rewards': None}
# plot_avg_vals(
#     'scores', 5000, 10,
#      bak_path + f'cartpole_classical/params_{1729}/', 'NN, 1729 params', 'orange', hps, plot_to=plot_to)
#
#
# hps = {
#     'learning_rate': 0.001,
#     'update_after': 5, 'update_target_after': 10, 'batch_size': 64, 'use_negative_rewards': None}
# plot_avg_vals(
#     'scores', 5000, 10,
#      bak_path + f'cartpole_classical/params_{790}/', 'NN, 790 params', 'purple', hps, plot_to=plot_to)
#
#
# hps = {
#     'learning_rate': 0.001,
#     'update_after': 5, 'update_target_after': 10, 'batch_size': 64, 'use_negative_rewards': None}
# plot_avg_vals(
#     'scores', 5000, 10,
#      bak_path + f'cartpole_classical/params_{405}/', 'NN, 405 params', 'magenta', hps, plot_to=plot_to)
#
#
# hps = {
#     'learning_rate': 0.001,
#     'update_after': 5, 'update_target_after': 10, 'batch_size': 64, 'use_negative_rewards': None}
# plot_avg_vals(
#     'scores', 5000, 10,
#      bak_path + f'cartpole_classical/params_{1702}/', 'NN, 1702 params', 'chartreuse', hps, plot_to=plot_to)


# 3000
plot_avg_vals(
    'scores', 5000, 10,
    bak_path + 'cartpole_classical/hp_search/', '(9, 10), 167', 'orchid',
    {'l1_units': 9, 'l2_units': 10, 'update_target_after': 10, 'update_after': 5, 'batch_size': 16,
     'learning_rate': 0.01})


# 850
hps = {
    'learning_rate': 0.01,
    'update_after': 5, 'update_target_after': 10, 'batch_size': 64, 'use_negative_rewards': False}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{365}/', '(15, 16), 365', 'black', hps, plot_to=plot_to)


# 590
hps = {
    'learning_rate': 0.001,
    'update_after': 1, 'update_target_after': 1, 'batch_size': 64, 'use_negative_rewards': False}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{770}/', '(24, 24), 770', 'g', hps, plot_to=plot_to)


# 400
hps = {
    'learning_rate': 0.001,
    'update_after': 1, 'update_target_after': 1, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{886}/', '(26, 26), 886', 'purple', hps, plot_to=plot_to)


plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Neural networks, (hidden layer units), # parameters")
# plt.ylim(ymax=200)
plt.legend()  # loc='lower right')
plt.show()