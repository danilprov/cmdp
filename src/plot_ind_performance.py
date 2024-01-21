import numpy as np
import glob
import os
import ntpath
import pickle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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

def parse_result(rolling_results):
    """
    results - 3d numpy
    """

    mean_results = np.mean(rolling_results, axis=0)
    std_results = np.std(rolling_results, axis=0)
    mean_reward = mean_results[:,0]
    mean_constraint = mean_results[:,1]
    std_reward = std_results[:,0]
    std_constraint = std_results[:,1]

    return mean_reward, std_reward, mean_constraint, std_constraint

def compute_regret(rolling_results, opt_reward, opt_cost):
    rewards = rolling_results[:, :, 0]
    costs = rolling_results[:, :, 1]
    main_regret = np.cumsum(np.maximum(opt_reward - rewards, 0), axis=1)
    aux_regret = np.cumsum(np.maximum(opt_cost - costs, 0), axis=1)

    mean_main_regret = np.mean(main_regret, axis=0)
    std_main_regret = np.std(main_regret, axis=0)
    mean_aux_regret = np.mean(aux_regret, axis=0)
    std_aux_regret = np.std(aux_regret, axis=0)

    return mean_main_regret, std_main_regret, mean_aux_regret, std_aux_regret


path = './log/'#'../log/to_plot/'
list_of_folders = glob.glob(path + '*')
list_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
fig, axs = plt.subplots(3, sharex=True)

folder = max(list_of_folders, key=os.path.getctime)

all_files = glob.glob(folder + '/' + "*.pickle")
if 'box' in folder:
    title = 'Box'
    opt_reward = -0.8479
    safe_reward = -0.875
    fast_reward = -0.833333
    opt_cost = 0.6

    XMAX = 500000
    n_average = 1000
    text_pos_x = 0
    eps_shift_up = 0.009
    eps_shift_down = 0.013

elif 'marsrover' in folder:
    titile = 'Marsrover 8x8'
    opt_reward = -0.93809524
    safe_reward = -0.95238095
    fast_reward = -0.92307692
    opt_cost = 0.1

    XMAX = 70000
    n_average = 5000
    text_pos_x = 40000
    eps_shift_up = 0.004
    eps_shift_down = 0.008

else:
    titile = 'Marsrover 4x4'
    opt_reward = -0.775
    safe_reward = -0.875
    fast_reward = -0.75
    opt_cost = 0.2

    XMAX = 8000
    n_average = 500
    text_pos_x = 7500
    eps_shift_up = 0.005
    eps_shift_down = 0.023
    legend_pos = [0.48, 0.58]

i=0
for filename in all_files:
    with open(filename, 'rb') as f:
        results = pickle.load(f)

    alg_name = path_leaf(filename)[:-7]
    rolling_results = moving_average(results, n=n_average)
    mean_reward, std_reward, mean_constraint, std_constraint = parse_result(rolling_results)
    mean_regret, std_regret, mean_aux_regret, std_aux_regret = compute_regret(rolling_results, opt_reward, opt_cost)

    mean_reward = mean_reward[: XMAX + 1]
    std_reward = std_reward[: XMAX + 1]
    mean_constraint = mean_constraint[: XMAX + 1]
    std_constraint = std_constraint[: XMAX + 1]
    mean_regret = mean_regret[: XMAX + 1]
    std_regret = std_regret[: XMAX + 1]
    mean_aux_regret = mean_aux_regret[: XMAX + 1]
    std_aux_regret = std_aux_regret[: XMAX + 1]

    # plot main regret
    axs[0].plot(mean_regret, label=alg_name, linestyle='-.', color=list_colors[i], lw=1)
    axs[0].fill_between(range(len(mean_regret)),
                        mean_regret - std_regret,
                        mean_regret + std_regret, alpha=0.2)

    # plot actual cost
    axs[1].plot(mean_constraint, linestyle='-.', color=list_colors[i])
    axs[1].fill_between(range(len(mean_reward)),
                        mean_constraint - std_constraint,
                        mean_constraint + std_constraint, alpha=0.2)

    # plot reward
    axs[2].plot(mean_reward, linestyle='-.', color=list_colors[i], lw=1)
    axs[2].fill_between(range(len(mean_reward)),
                        mean_reward - std_reward,
                        mean_reward + std_reward, alpha=0.2)
    i+=1

axs[1].hlines(y=opt_cost, xmin=0, xmax=XMAX, colors='#7f7f7f', linestyles='--', lw=1)
axs[0].set(ylabel='main regret')
axs[1].set(ylabel='consumption')
axs[2].hlines(y=opt_reward, xmin=0, xmax=XMAX, colors='#7f7f7f', linestyles='--', lw=1)
axs[2].hlines(y=safe_reward, xmin=0, xmax=XMAX, colors='#7f7f7f', linestyles=':', lw=1)
axs[2].hlines(y=fast_reward, xmin=0, xmax=XMAX, colors='#7f7f7f', linestyles=':', lw=1)
axs[2].text(text_pos_x, safe_reward - eps_shift_down, 'safe', color='#7f7f7f')
axs[2].text(text_pos_x, fast_reward + eps_shift_up, 'fast', color='#7f7f7f')
axs[2].set(xlabel='rounds', ylabel='reward')

for ax in axs:
    ax.grid(visible=True, which='major', linestyle='--', alpha=0.5)
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', linestyle=':', alpha=0.2)

lines_labels = [ax.get_legend_handles_labels() for ax in axs]
lines, labels = [sum(_, []) for _ in zip(*lines_labels)]
fig.legend(lines, labels,
           loc='upper center',
           bbox_to_anchor=(0.5, -0.01),
           fancybox=True, shadow=True, ncol=5)


plt.savefig(folder + '/' + 'av_reward_cost' + '.png', bbox_inches='tight')
plt.show()
