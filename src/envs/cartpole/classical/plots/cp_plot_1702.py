
import matplotlib.pyplot as plt

from config import BASE_PATH
from src.utils.plots import plot_avg_vals

bak_path = '/home/andrea/BAK/vql/data/'
path = '../../../../../' + BASE_PATH
params = 1702


hps = {
    'learning_rate': 0.001,
    'update_after': 5, 'update_target_after': 5, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{params}/', '', 'g', hps)


hps = {
    'learning_rate': 0.0001,
    'update_after': 5, 'update_target_after': 5, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{params}/', '', 'b', hps)


hps = {
    'learning_rate': 0.0001,
    'update_after': 5, 'update_target_after': 10, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{params}/', '', 'purple', hps)


hps = {
    'learning_rate': 0.001,
    'update_after': 10, 'update_target_after': 20, 'batch_size': 32}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{params}/', '', 'yellow', hps)


hps = {
    'learning_rate': 0.0001,
    'update_after': 10, 'update_target_after': 20, 'batch_size': 32}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{params}/', '', 'chartreuse', hps)


hps = {
    'learning_rate': 0.001,
    'update_after': 10, 'update_target_after': 30, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{params}/', '', 'darkblue', hps)


hps = {
    'learning_rate': 0.001,
    'update_after': 10, 'update_target_after': 40, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{params}/', '', 'darkgreen', hps)

hps = {
    'learning_rate': 0.001,
    'update_after': 20, 'update_target_after': 40, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{params}/', '', 'pink', hps)

hps = {
    'learning_rate': 0.001,
    'update_after': 20, 'update_target_after': 30, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{params}/', '', 'orchid', hps)

hps = {
    'learning_rate': 0.001,
    'update_after': 10, 'update_target_after': 20, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{params}/', '', 'grey', hps)

hps = {
    'learning_rate': 0.001,
    'update_after': 5, 'update_target_after': 5, 'batch_size': 32}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{params}/', '', 'orange', hps)


### BEST ###
hps = {
    'learning_rate': 0.001,
    'update_after': 5, 'update_target_after': 10, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{params}/', '', 'red', hps)








# hps = {
#     'learning_rate': 0.001,
#     'update_after': 10, 'update_target_after': 30, 'batch_size': 64}
# plot_avg_vals(
#     'scores', 5000, 10,
#      bak_path + f'cartpole_classical/params_{params}/', '', 'magenta', hps)

# hps = {
#     'learning_rate': 0.0001,
#     'update_after': 5, 'update_target_after': 5, 'batch_size': 64}
# plot_avg_vals(
#     'scores', 5000, 10,
#      bak_path + f'cartpole_classical/params_{params}/', '', 'black', hps)


plt.xlabel("Episode")
plt.ylabel("Score")
plt.title(f"NNs with varying hyperparameters, {params} parameters (preliminary)")
# plt.ylim(ymax=200)
# plt.legend()  # loc='lower right')
plt.show()