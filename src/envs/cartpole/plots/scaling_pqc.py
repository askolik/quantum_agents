
import matplotlib.pyplot as plt
from config import BASE_PATH
from src.utils.plots import plot_avg_vals


bak_path = '/home/andrea/BAK/vql/data/'
path = '../../../../' + BASE_PATH
plot_to = 2500


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