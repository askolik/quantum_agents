
import matplotlib.pyplot as plt
import seaborn as sb
from config import BASE_PATH
from src.utils.plots import plot_avg_vals


bak_path = '/home/andrea/BAK/vql/data/'
path = '../../../../' + BASE_PATH
plot_to = 5000


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


colors = ['g', 'b', 'orange', 'red', 'dimgrey', 'orchid', 'gold', 'purple'][::-1]
# for depth in [5, 10, 15, 20, 25, 30][::-1]:  # range(5, 10, 1):
#     plot_avg_vals(
#         'scores', 5000, 10,
#         path + f'cartpole/depth_scaling/', f'{depth}, ({depth * 4 * 3 + 2})', colors.pop(),
#         {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
#          'batch_size': 16, 'update_after': 1, 'update_target_after': 1}, plot_to=plot_to)

sb.set_style("whitegrid")
subplts = [[i, j] for i in range(4) for j in range(2)][::-1]
fig, axs = plt.subplots(4, 2)

hp_dict = {
    3: {'circuit_depth': 3, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
         'batch_size': 16, 'update_after': 1, 'update_target_after': 1},
    5: {'circuit_depth': 5, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
         'batch_size': 16, 'update_after': 1, 'update_target_after': 1},
    10: {'circuit_depth': 10, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
         'batch_size': 64, 'update_after': 10, 'update_target_after': 30},
    15: {'circuit_depth': 15, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
         'batch_size': 32, 'update_after': 10, 'update_target_after': 30},
    20: {'circuit_depth': 20, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
         'batch_size': 16, 'update_after': 10, 'update_target_after': 30},
    25: {'circuit_depth': 25, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
         'batch_size': 64, 'update_after': 10, 'update_target_after': 30},
    30: {'circuit_depth': 30, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
         'batch_size': 64, 'update_after': 10, 'update_target_after': 30}
}

path_dict = {
    3: bak_path + f'cartpole/depth_scaling/',
    5: bak_path + f'cartpole/depth_scaling/',
    10: bak_path + f'cartpole/depth_10_mse/',
    15: bak_path + f'cartpole/depth_15_mse/',
    20: bak_path + f'cartpole/depth_20_mse/',
    25: bak_path + f'cartpole/depth_25_mse/',
    30: bak_path + f'cartpole/depth_scaling/'
}

for depth in [3, 5, 10, 15, 20, 25, 30]:
    ax_coords = subplts.pop()
    curr_ax = axs[ax_coords[0], ax_coords[1]]
    plot_avg_vals(
        'scores', 5000, 10,
        path_dict[depth], f'{depth} ({depth * 4 * 3 + 2})', colors.pop(),
        hp_dict[depth], plot_to=plot_to, plt_obj=curr_ax)

    curr_ax.legend()

    # axs[ax_coords[0], ax_coords[1]].scatter([x for x in range(len(encoded_energies))], encoded_energies,
    #                                         label="Encoded")
    # axs[ax_coords[0], ax_coords[1]].scatter([x for x in range(len(rand_energies))], rand_energies, label="Random")
    # axs[ax_coords[0], ax_coords[1]].set_title("order {}".format(order))


# plt.xlabel("Episode")
# plt.ylabel("Score")
# plt.title("PQCs, # layers, (# params)")
# # plt.ylim(ymax=200)
# plt.legend()  # loc='lower right')
fig.suptitle("PQCs, legend shows: # layers (# params)")
fig.tight_layout()
plt.show()