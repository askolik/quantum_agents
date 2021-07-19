
import matplotlib.pyplot as plt

from config import BASE_PATH
from src.utils.plots import plot_avg_vals

bak_path = '/home/andrea/BAK/vql/data/'
path = '../../../../../' + BASE_PATH
params = 790


hps = {
    'learning_rate': 0.001,
    'update_after': 5, 'update_target_after': 10, 'batch_size': 16}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{params}/', '', 'g', hps)


hps = {
    'learning_rate': 0.0001,
    'update_after': 5, 'update_target_after': 10, 'batch_size': 16}
plot_avg_vals(
    'scores', 5000, 10,
     bak_path + f'cartpole_classical/params_{params}/', '', 'b', hps)





plt.xlabel("Episode")
plt.ylabel("Score")
plt.title(f"NNs with varying hyperparameters, {params} parameters (preliminary)")
# plt.ylim(ymax=200)
# plt.legend()  # loc='lower right')
plt.show()