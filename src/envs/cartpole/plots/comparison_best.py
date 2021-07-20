
import matplotlib.pyplot as plt
from config import BASE_PATH
from src.utils.plots import plot_avg_vals


bak_path = '/home/andrea/BAK/vql/data/'
path = '../../../../' + BASE_PATH
plot_to = 2500


hps = {
    'learning_rate': 0.001,
    'update_after': 10, 'update_target_after': 20, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{1729}/', 'NN, 1729 params', 'orange', hps, plot_to=plot_to)


hps = {
    'learning_rate': 0.001,
    'update_after': 5, 'update_target_after': 10, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{790}/', 'NN, 790 params', 'purple', hps, plot_to=plot_to)


hps = {
    'learning_rate': 0.001,
    'update_after': 5, 'update_target_after': 10, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{405}/', 'NN, 405 params', 'magenta', hps, plot_to=plot_to)


hps = {
    'learning_rate': 0.001,
    'update_after': 5, 'update_target_after': 10, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{1702}/', 'NN, 1702 params', 'red', hps, plot_to=plot_to)


hps = {
    'learning_rate': 0.001,
    'update_after': 5, 'update_target_after': 10, 'batch_size': 64}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{792}/', 'NN, 792 params', 'g', hps, plot_to=plot_to)


depth = 10
plot_avg_vals(
    'scores', 5000, 10,
    bak_path + f'cartpole/depth_{depth}_mse/', 'PQC, 10 layers', 'b',
    {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
     'batch_size': 64}, plot_to=plot_to)


plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Comparison of best (so far) configurations")
# plt.ylim(ymax=200)
plt.legend()  # loc='lower right')
plt.show()