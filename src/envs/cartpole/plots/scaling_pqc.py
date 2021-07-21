
import matplotlib.pyplot as plt
from config import BASE_PATH
from src.utils.plots import plot_avg_vals


bak_path = '/home/andrea/BAK/vql/data/'
path = '../../../../' + BASE_PATH
plot_to = 5000


plot_avg_vals(
    'scores', 5000, 10,
    '/home/andrea/BAK/vql/data/' + 'cartpole/cirq/good_hps/', '5 layers', 'g',
    {'data_reuploading': True, 'n_layers': 5, 'train_weights': True, 'train_data_scaling': True}, plot_to=plot_to)


depth = 10
plot_avg_vals(
    'scores', 5000, 10,
    bak_path + f'cartpole/depth_{depth}_mse/', '10 layers', 'g',
    {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
     'batch_size': 64}, plot_to=plot_to)


depth = 15
plot_avg_vals(
    'scores', 5000, 10,
    bak_path + f'cartpole/depth_{depth}_mse/', '15 layers', 'b',
    {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
     'batch_size': 32}, plot_to=plot_to)


# depth = 20
# plot_avg_vals(
#     'scores', 5000, 10,
#     bak_path + f'cartpole/depth_{depth}_mse/', '20 layers', 'orange',
#     {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
#      'batch_size': 64}, plot_to=plot_to)


depth = 25
plot_avg_vals(
    'scores', 5000, 10,
    bak_path + f'cartpole/depth_{depth}_mse/', '25 layers', 'red',
    {'circuit_depth': depth, 'learning_rate': 0.001, 'learning_rate_in': 0.001, 'learning_rate_out': 0.1,
     'batch_size': 64}, plot_to=plot_to)


plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Comparison of best (so far) configurations")
# plt.ylim(ymax=200)
plt.legend()  # loc='lower right')
plt.show()