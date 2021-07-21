
import matplotlib.pyplot as plt
from config import BASE_PATH
from src.utils.plots import plot_avg_vals


bak_path = '/home/andrea/BAK/vql/data/'
path = '../../../../' + BASE_PATH
plot_to = 2500


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


# 2000
hps = {
    'learning_rate': 0.01,
    'update_after': 5, 'update_target_after': 10, 'batch_size': 16, 'use_negative_rewards': False}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{562}/', '(20, 20), 562', 'red', hps, plot_to=plot_to)


# 590
hps = {
    'learning_rate': 0.001,
    'update_after': 1, 'update_target_after': 1, 'batch_size': 64, 'use_negative_rewards': False}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{770}/', '(24, 24), 770', 'g', hps, plot_to=plot_to)


plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Comparison of best (so far) configurations")
# plt.ylim(ymax=200)
plt.legend()  # loc='lower right')
plt.show()