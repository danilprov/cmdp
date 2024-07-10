import numpy as np
import glob
import os
import ntpath
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def moving_average(array3d, n=5):
    """
    moving averaging along rounds (axis=1)
    """
    ret = np.cumsum(array3d, axis=1)
    ret[:, n:, :] = ret[:, n:, :] - ret[:, :-n, :]
    return ret[:, n - 1:, :] / n

def parse_result(results, n_average):
    """
    results - 3d numpy
    """
    rolling_results = moving_average(results, n=n_average)
    mean_results = np.mean(rolling_results, axis=0)
    std_results = np.std(rolling_results, axis=0)

    mean_reward = mean_results[:,0]
    mean_constraint = mean_results[:,1]
    std_reward = std_results[:,0]
    std_constraint = std_results[:,1]

    return mean_reward, std_reward, mean_constraint, std_constraint

path = './log/to_plot2/'
list_of_folders = glob.glob(path + '*')
latest_folder = max(list_of_folders, key=os.path.getctime)

list_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

all_files = glob.glob(latest_folder + '/' + "*.pickle")
fig, axs = plt.subplots(2, sharex=True)#, figsize=(12,6))
#fig.suptitle('Average reward and cost (' + title + ')')

if 'box' in latest_folder:
    opt_reward = -0.8479
    safe_reward = -0.875
    fast_reward = -0.833333
    opt_cost = 0.6

    XMAX = 2000000
    n_average = 50000
    text_pos_x = 0
    eps_shift_up = 0.009
    eps_shift_down = 0.013
    legend_pos = [0.33, 0.53]
elif 'marsrover' in latest_folder:
    opt_reward = -0.93809524
    safe_reward = -0.95238095
    fast_reward = -0.92307692
    opt_cost = 0.1

    XMAX = 200000
    n_average = 5000
    text_pos_x = 50000
    eps_shift_up = 0.004
    eps_shift_down = 0.008
    legend_pos = [0.73, 0.53]
else:
    opt_reward = -0.775
    safe_reward = -0.875
    fast_reward = -0.75
    opt_cost = 0.2

    XMAX = 7000
    n_average = 500
    text_pos_x = 6500
    eps_shift_up = 0.005
    eps_shift_down = 0.023
    legend_pos = [0.48, 0.58]

i=0
for filename in all_files:
    with open(filename, 'rb') as f:
        results = pickle.load(f)

    alg_name = path_leaf(filename)[:-7]

    mean_reward, std_reward, mean_constraint, std_constraint = parse_result(results, n_average)
    #mean_reward = mean_reward[: XMAX + 1]
    #mean_constraint = mean_constraint[: XMAX + 1]
    cum_regret_main = np.cumsum(np.maximum(opt_reward - mean_reward, 0))
    cum_regret_cost = np.cumsum(np.maximum(mean_constraint - opt_cost,0))

    std_reward = std_reward[: XMAX + 1]
    std_constraint = std_constraint[: XMAX + 1]

    axs[0].plot(cum_regret_main, label=alg_name, linestyle='-.', color=list_colors[i], lw=1)
    #axs[0].fill_between(range(len(mean_reward)), mean_reward - std_reward, mean_reward + std_reward, alpha=0.2)

    axs[1].plot(cum_regret_cost, linestyle='-.', color=list_colors[i])
    # axs[1].fill_between(range(len(mean_reward)), mean_constraint - std_constraint, mean_constraint + std_constraint,
    #                     alpha=0.2)
    i+=1

axs[1].set(xlabel='rounds', ylabel='auxiliary regret')
axs[0].set(ylabel='main regret')
for ax in axs:
    ax.grid(visible=True, which='major', linestyle='--', alpha=0.5)
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', linestyle=':', alpha=0.2)

lines_labels = [ax.get_legend_handles_labels() for ax in axs]
lines, labels = [sum(_, []) for _ in zip(*lines_labels)]
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True, shadow=True, ncol=5)
#fig.legend(lines, labels, bbox_to_anchor=legend_pos, loc='center')

plt.savefig(latest_folder + '/' + 'av_regret' + '.png', bbox_inches='tight')
plt.show()
