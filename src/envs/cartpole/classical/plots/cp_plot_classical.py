
import matplotlib.pyplot as plt

from config import BASE_PATH
from src.utils.plots import plot_avg_vals

plot_avg_vals(
    'scores', 5000, 10,
    '/home/andrea/BAK/vql/data/' + 'cartpole_classical/params_385/', '385 params', 'g',
    {'learning_rate': 0.001})

plot_avg_vals(
    'scores', 5000, 10,
    '/home/andrea/BAK/vql/data/' + 'cartpole_classical/params_385/', '385 params', 'b',
    {'learning_rate': 0.0001})


plot_avg_vals(
    'scores', 5000, 10,
    '/home/andrea/BAK/vql/data/' + 'cartpole_classical/params_582/', '582 params', 'red',
    {'learning_rate': 0.001})

plot_avg_vals(
    'scores', 5000, 10,
    '/home/andrea/BAK/vql/data/' + 'cartpole_classical/params_582/', '582 params', 'magenta',
    {'learning_rate': 0.0001})



plt.xlabel("Episode")
plt.ylabel("Score")
# plt.title("CartPole v0, averaged over 10 agents each")
# plt.ylim(ymax=200)
plt.legend()  # loc='lower right')
plt.show()