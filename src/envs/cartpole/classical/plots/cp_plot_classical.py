
import matplotlib.pyplot as plt

from config import BASE_PATH
from src.utils.plots import plot_avg_vals

bak_path = '/home/andrea/BAK/vql/data/'
path = '../../../../../' + BASE_PATH


hps = {
    'learning_rate': 0.01,
    'update_after': 5, 'update_target_after': 10, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + 'cartpole_classical/params_790/', '', 'g', hps)


hps = {
    'learning_rate': 0.01,
    'update_after': 5, 'update_target_after': 10, 'batch_size': 32}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + 'cartpole_classical/params_790/', '', 'b', hps)


hps = {
    'learning_rate': 0.0001,
    'update_after': 10, 'update_target_after': 20, 'batch_size': 32}
plot_avg_vals(
    'scores', 5000, 10,
     path + 'cartpole_classical/params_405/', '', 'orange', hps)


depth = 10
plot_avg_vals(
    'scores', 5000, 10,
    bak_path + f'cartpole/depth_{depth}_mse/', 'PQC', 'red',
    {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
     'batch_size': 64})

# hps = {
#     'learning_rate': 0.0001,
#     'update_after': 5, 'update_target_after': 10, 'batch_size': 16}
# plot_avg_vals(
#     'scores', 5000, 10,
#      bak_path + 'cartpole_classical/params_790/', 'update 1, 1', 'g', hps)
#
# hps = {
#     'learning_rate': 0.001,
#     'update_after': 5, 'update_target_after': 10, 'batch_size': 64}
# plot_avg_vals(
#     'scores', 5000, 10,
#      path + 'cartpole_classical/params_790/', 'update 5, 10', 'r', hps)

# hps = {
#     'learning_rate': 0.001,
#     'update_after': 10, 'update_target_after': 20}
# plot_avg_vals(
#     'scores', 5000, 10,
#      bak_path + 'cartpole_classical/params_385/', '385 params', 'b', hps)
#
# hps = {
#     'learning_rate': 0.001,
#     'update_after': 20, 'update_target_after': 30}
# plot_avg_vals(
#     'scores', 5000, 10,
#      bak_path + 'cartpole_classical/params_385/', '385 params', 'r', hps)



# hps['learning_rate'] = 0.0001
# plot_avg_vals(
#     'scores', 5000, 10,
#     bak_path + 'cartpole_classical/params_385/', '385 params', 'g', hps)


# hps = {
#     'learning_rate': 0.001,
#     'update_after': 5, 'update_target_after': 10}
# plot_avg_vals(
#     'scores', 5000, 10,
#      bak_path + 'cartpole_classical/params_582/', '582 params', 'b', hps)
#
# hps = {
#     'learning_rate': 0.0001,
#     'update_after': 5, 'update_target_after': 10}
# plot_avg_vals(
#     'scores', 5000, 10,
#     bak_path + 'cartpole_classical/params_582/', '582 params', 'g', hps)


# hps['learning_rate'] = 0.001
# plot_avg_vals(
#     'scores', 5000, 10,
#     bak_path + 'cartpole_classical/params_790/', '790 params', 'r', hps)

# hps['learning_rate'] = 0.0001
# plot_avg_vals(
#     'scores', 5000, 10,
#     path + 'cartpole_classical/params_790/', '790 params', 'r', hps)


plt.xlabel("Episode")
plt.ylabel("Score")
# plt.title("CartPole v0, averaged over 10 agents each")
# plt.ylim(ymax=200)
plt.legend()  # loc='lower right')
plt.show()