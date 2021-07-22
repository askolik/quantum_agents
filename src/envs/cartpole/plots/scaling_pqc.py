
import matplotlib.pyplot as plt
from config import BASE_PATH
from src.utils.plots import plot_avg_vals


bak_path = '/home/andrea/BAK/vql/data/'
path = '../../../../' + BASE_PATH
plot_to = 800


# plot_avg_vals(
#     'scores', 5000, 10,
#     '/home/andrea/BAK/vql/data/' + 'cartpole/cirq/good_hps/', '5 (54, old)', 'g',
#     {'data_reuploading': True, 'n_layers': 5, 'train_weights': True, 'train_data_scaling': True}, plot_to=plot_to)


# depth = 5
# plot_avg_vals(
#     'scores', 5000, 10,
#     bak_path + f'cartpole/depth_{depth}_mse/', '5 (62)', 'darkgreen',
#     {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
#      'batch_size': 32}, plot_to=plot_to)


# depth = 10
# plot_avg_vals(
#     'scores', 5000, 10,
#     bak_path + f'cartpole/depth_{depth}_mse/', '10 (122)', 'g',
#     {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
#      'batch_size': 64}, plot_to=plot_to)
#
#
# depth = 15
# plot_avg_vals(
#     'scores', 5000, 10,
#     bak_path + f'cartpole/depth_{depth}_mse/', '15 (182)', 'b',
#     {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
#      'batch_size': 32}, plot_to=plot_to)


# depth = 20
# plot_avg_vals(
#     'scores', 5000, 10,
#     bak_path + f'cartpole/depth_{depth}_mse/', '20 (242)', 'orange',
#     {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
#      'batch_size': 64}, plot_to=plot_to)


# depth = 25
# plot_avg_vals(
#     'scores', 5000, 10,
#     bak_path + f'cartpole/depth_{depth}_mse/', '25 (302)', 'red',
#     {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
#      'batch_size': 64}, plot_to=plot_to)


colors = ['g', 'b', 'orange', 'red', 'dimgrey', 'orchid', 'gold'][::-1]
for depth in [5, 9]:  # range(5, 10, 1):
    plot_avg_vals(
        'scores', 5000, 10,
        path + f'cartpole/depth_scaling/', f'{depth}, ({depth * 4 * 3 + 2})', colors.pop(),
        {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
         'batch_size': 16, 'update_after': 1, 'update_target_after': 1}, plot_to=plot_to)


plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("PQCs, # layers, (# params)")
# plt.ylim(ymax=200)
plt.legend()  # loc='lower right')
plt.show()