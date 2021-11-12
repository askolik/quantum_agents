
import matplotlib.pyplot as plt
import seaborn as sb
from config import BASE_PATH
from src.utils.plots import plot_avg_vals


bak_path = '/home/andrea/BAK/vql/data/'
path = '../../../../' + BASE_PATH
plot_to = 1000


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


# # 3000
# plot_avg_vals(
#     'scores', 5000, 10,
#     bak_path + 'cartpole_classical/hp_search/', '(9, 10), 167', 'orchid',
#     {'l1_units': 9, 'l2_units': 10, 'update_target_after': 10, 'update_after': 5, 'batch_size': 16,
#      'learning_rate': 0.01})
#
#
# # 850
# hps = {
#     'learning_rate': 0.01,
#     'update_after': 5, 'update_target_after': 10, 'batch_size': 64, 'use_negative_rewards': False}
# plot_avg_vals(
#     'scores', 5000, 10,
#      bak_path + f'cartpole_classical/params_{365}/', '(15, 16), 365', 'black', hps, plot_to=plot_to)
#
#
# # 280
# hps = {
#     'learning_rate': 0.001,
#     'update_after': 1, 'update_target_after': 1, 'batch_size': 64, 'use_negative_rewards': False}
# plot_avg_vals(
#     'scores', 5000, 10,
#      bak_path + f'cartpole_classical/params_{562}/', '(20, 20), 562', 'b', hps)
#
#
# # 590
# hps = {
#     'learning_rate': 0.001,
#     'update_after': 1, 'update_target_after': 1, 'batch_size': 64, 'use_negative_rewards': False}
# plot_avg_vals(
#     'scores', 5000, 10,
#      bak_path + f'cartpole_classical/params_{770}/', '(24, 24), 770', 'g', hps, plot_to=plot_to)
#
#
# # # 400
# # hps = {
# #     'learning_rate': 0.001,
# #     'update_after': 1, 'update_target_after': 1, 'batch_size': 64}
# # plot_avg_vals(
# #     'scores', 5000, 10,
# #      bak_path + f'cartpole_classical/params_{886}/', '(26, 26), 886', 'purple', hps, plot_to=plot_to)
#
#
# hps = {
#     'learning_rate': 0.001,
#     'update_after': 1, 'update_target_after': 1, 'batch_size': 64}
# plot_avg_vals(
#     'scores', 5000, 10,
#      bak_path + f'cartpole_classical/params_{1142}/', '(30, 30), 1142', 'orange', hps, plot_to=plot_to)
#
#
# hps = {
#     'learning_rate': 0.001,
#     'update_after': 1, 'update_target_after': 1, 'batch_size': 16, 'use_negative_rewards': False}
# plot_avg_vals(
#     'scores', 5000, 10,
#      bak_path + f'cartpole_classical/params_{4610}/', '(64, 64), 4610', 'red', hps)
#
#
#
# plt.xlabel("Episode")
# plt.ylabel("Score")
# plt.title("Neural networks, (hidden layer units), # parameters")
# # plt.ylim(ymax=200)
# plt.legend()  # loc='lower right')
# plt.show()


colors = ['g', 'b', 'orange', 'red', 'dimgrey', 'orchid', 'gold'][::-1]
sb.set_style("whitegrid")
subplts = [[i, j] for i in range(3) for j in range(2)][::-1]
fig, axs = plt.subplots(3, 2)

hp_dict = {
    182: {
        'learning_rate': 0.001, 'update_after': 1, 'update_target_after': 1,
        'batch_size': 64, 'use_negative_rewards': False},
    347: {
        'learning_rate': 0.001, 'update_after': 1, 'update_target_after': 1,
        'batch_size': 16, 'use_negative_rewards': False},
    562: {
        'learning_rate': 0.001, 'update_after': 1, 'update_target_after': 1,
        'batch_size': 64, 'use_negative_rewards': False},
    770: {
        'learning_rate': 0.001, 'update_after': 1, 'update_target_after': 1,
        'batch_size': 64, 'use_negative_rewards': False},
    1142: {
        'learning_rate': 0.001, 'update_after': 1, 'update_target_after': 1,
        'batch_size': 64},
    4610: {
        'learning_rate': 0.001, 'update_after': 1, 'update_target_after': 1,
        'batch_size': 16, 'use_negative_rewards': False}
}

label_dict = {
    182: '(10, 10), 182',
    347: '(15, 15), 347',
    562: '(20, 20), 562',
    770: '(24, 24), 770',
    1142: '(30, 30), 1142',
    4610: '(64, 64), 4610'
}

# ax_coords = subplts.pop()
# curr_ax = axs[ax_coords[0], ax_coords[1]]
# plot_avg_vals(
#     'scores', 5000, 10,
#     bak_path + 'cartpole_classical/hp_search/', '(9, 10), 167', colors.pop(),
#     {'l1_units': 9, 'l2_units': 10, 'update_target_after': 10, 'update_after': 5, 'batch_size': 16,
#      'learning_rate': 0.01}, plot_to=4000, plt_obj=curr_ax)
# curr_ax.legend()

for params in [182, 347, 562, 770, 1142, 4610]:
    ax_coords = subplts.pop()
    curr_ax = axs[ax_coords[0], ax_coords[1]]
    x_axis = plot_to
    if params == 182:
        x_axis = 1110

    plot_avg_vals(
        'scores', 5000, 10,
        bak_path + f'cartpole_classical/params_{params}/', label_dict[params], colors.pop(), hp_dict[params],
        plot_to=x_axis, plt_obj=curr_ax, avg_solved=True)

    curr_ax.legend()

# fig.suptitle("NNs, legend shows: (hidden layer units), # params")
fig.supxlabel('Episode')
fig.supylabel('Score')
fig.tight_layout()
plt.show()