
import matplotlib.pyplot as plt
import seaborn as sb
from config import BASE_PATH
from src.utils.plots import plot_avg_vals, plot_by_model_name

bak_path = '/home/andrea/BAK/vql/data/'
sb.set_style("whitegrid")

plot_by_model_name(
    '2021-07-22_11-16-33_799_3', 'scores', bak_path + 'cartpole_classical/params_1142/', 'NN, (30, 30), 1142', 'b')
plot_by_model_name('2021-07-21_14-47-56_967_9', 'scores', bak_path + 'cartpole/depth_scaling/', 'PQC, 5 (62)', 'orange')

plt.xlabel("Episode")
plt.ylabel("Score")
# plt.title("Comparison of best configurations")
# plt.ylim(ymax=200)
plt.legend()  # loc='lower right')
plt.show()