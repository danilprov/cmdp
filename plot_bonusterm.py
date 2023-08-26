import numpy as np
import glob
import os
import ntpath
import pickle

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
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
    aux_regret = np.cumsum(np.maximum(costs - opt_cost, 0), axis=1)

    mean_main_regret = np.mean(main_regret, axis=0)
    std_main_regret = np.std(main_regret, axis=0)
    mean_aux_regret = np.mean(aux_regret, axis=0)
    std_aux_regret = np.std(aux_regret, axis=0)

    return mean_main_regret, std_main_regret, mean_aux_regret, std_aux_regret


path = './log/to_plot/'
list_of_folders = glob.glob(path + '*')
list_colors = ['#1f77b4', '#d62728', '#9467bd',
               '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#7f7f7f', '#bcbd22', '#17becf']

list_markers = ["o", "^", "s", "x", "|"]
#list_markers = [".", "2", "+", "x", "|"]
bonus_terms = ['.01', '.05', '0.1', '0.2', '0.5']
marker_bonus_dict = dict(zip(bonus_terms, list_markers))

plot_reward = False

fig, axs = plt.subplots(2, sharex='col', figsize=(15, 7))

# folder = max(list_of_folders, key=os.path.getctime)
folder = list_of_folders[-1]

offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
trans = plt.gca().transData


# j = 0
# for folder in list_of_folders:
all_files = glob.glob(folder + '/' + "*.pickle")
if 'box' in folder:
    title = 'Box'
    opt_reward = -0.8479
    safe_reward = -0.875
    fast_reward = -0.833333
    opt_cost = 0.6

    XMAX = 500000
    n_average = 9000
    text_pos_x = 0
    eps_shift_up = 0.009
    eps_shift_down = 0.013

elif 'marsrover' in folder:
    titile = 'Marsrover 8x8'
    opt_reward = -0.93809524
    safe_reward = -0.95238095
    fast_reward = -0.92307692
    opt_cost = 0.1

    XMAX = 20000
    n_average = 2000
    text_pos_x = 14000
    eps_shift_up = 0.004
    eps_shift_down = 0.008

    y0min, y0max = -50, 600
    y1min, y1max = -50, 1000

else:
    titile = 'Marsrover 4x4'
    opt_reward = -0.775
    safe_reward = -0.875
    fast_reward = -0.75
    opt_cost = 0.2

    XMAX = 5000
    n_average = 900
    text_pos_x = 4000
    eps_shift_up = 0.005
    eps_shift_down = 0.023
    legend_pos = [0.48, 0.58]

    y0min, y0max = None, 300
    y1min, y1max = None, 70

i=0
for filename in all_files:
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    if 'PSRLTransitions' in filename:
        alg_name = path_leaf(filename)[:-7]
        bonus_term = ' '
        marker = None
        marker_shift = 0
    else:
        alg_name = path_leaf(filename)[:-11]
        bonus_term = path_leaf(filename)[-10:-7]
        marker = marker_bonus_dict[bonus_term]
        marker_shift = int(bonus_terms.index(bonus_term) * XMAX / 10 / len(bonus_terms))
    print(alg_name, bonus_term, marker, XMAX / 10)
    ls = '-.'
    lw = '1'
    if 'CUCRLConservative' in filename:
        alg_name = 'C-UCRL'
        alg_name_title = 'C-UCRL'
        #color = list_colors[i]
        color = 'tab:blue'
    elif 'CUCRLOptimistic' in filename:
        alg_name = 'ConRL'
        alg_name_title = 'ConRL'
        #color = '#ff7f0e'
        #color = list_colors[i]
        color = 'tab:orange'
    elif 'CUCRLTransitions' in filename:
        alg_name = 'UCRL-CMDP'
        alg_name_title = 'UCRL-CMDP'
        color = list_colors[i]
    else:
        alg_name = 'PSConRL'
        #color = '#2ca02c'
        color = 'tab:green'
        ls = '-'
        lw = '3'
    rolling_results = moving_average(results, n=n_average)

    plot_name = 'regret'
    mean_regret, std_regret, mean_aux_regret, std_aux_regret = compute_regret(rolling_results, opt_reward,
                                                                              opt_cost)
    marker_indices = np.arange(0, XMAX, XMAX / 10, dtype=int) + marker_shift
    mean_regret = mean_regret[: XMAX + 1]
    std_regret = std_regret[: XMAX + 1]
    mean_aux_regret = mean_aux_regret[: XMAX + 1]
    std_aux_regret = std_aux_regret[: XMAX + 1]

    # plot main regret
    axs[0].plot(mean_regret, label=alg_name+bonus_term, linestyle=ls, lw=lw, color=color, marker=marker, markevery=marker_indices)
    axs[0].fill_between(range(len(mean_regret)),
                           mean_regret - std_regret,
                           mean_regret + std_regret, alpha=0.1, color=color)

    # plot actual cost
    axs[1].plot(mean_aux_regret, linestyle=ls, lw=lw, color=color, marker=marker, markevery=marker_indices)
    axs[1].fill_between(range(len(mean_aux_regret)),
                           mean_aux_regret - std_aux_regret,
                           mean_aux_regret + std_aux_regret, alpha=0.1, color=color)
    i+=1

axs[0].set(ylabel='main regret')
axs[1].set(ylabel='consumption')
if plot_reward:
    axs[0].hlines(y=opt_reward, xmin=0, xmax=XMAX, colors='#7f7f7f', linestyles='--', lw=1)
    axs[0].hlines(y=safe_reward, xmin=0, xmax=XMAX, colors='#7f7f7f', linestyles=':', lw=1)
    axs[0].hlines(y=fast_reward, xmin=0, xmax=XMAX, colors='#7f7f7f', linestyles=':', lw=1)
    axs[0].text(text_pos_x, safe_reward - eps_shift_down, 'safe', color='#7f7f7f')
    axs[0].text(text_pos_x, fast_reward + eps_shift_up, 'fast', color='#7f7f7f')
    #axs[2,j].set(xlabel='rounds', ylabel='reward')
    axs[0].set(xlabel='rounds')
    axs[0].xaxis.label.set_size(12)
else:
    axs[1].set(xlabel='rounds')
    axs[1].xaxis.label.set_size(12)
    axs[0].set_ylim(y0min, y0max)
    axs[1].set_ylim(y1min, y1max)

import matplotlib.markers as mmark
import matplotlib.lines as mlines
list_mak = [mmark.MarkerStyle(mark) for mark in list_markers]

handels = []
for key, value in marker_bonus_dict.items():
    legend_mark = mlines.Line2D([], [], color='tab:brown', marker=value, linestyle='None',
                          markersize=10, label=key)
    handels.append(legend_mark)

lines = axs[0].get_lines()
legend1 = plt.legend(handles=handels, loc='center right', bbox_to_anchor=(1.1, 1.09),
                     shadow=True, prop={'size': 12}, title="Bonus value")
if titile == 'Marsrover 4x4':
    legend2 = plt.legend([lines[i] for i in [3,0,9]], ['C-UCRL', 'ConRL', 'PSConRL'],
                         loc='upper center', bbox_to_anchor=(0.5, -0.19),
                         shadow=True, ncol=5, prop={'size': 15})
elif titile == 'Marsrover 8x8':
    legend2 = plt.legend([lines[i] for i in [2,8,9]], ['C-UCRL', 'ConRL', 'PSConRL'],
                         loc='upper center', bbox_to_anchor=(0.5, -0.19),
                         shadow=True, ncol=5, prop={'size': 15})
# legend1 = plt.legend(handles=handels, loc='upper center', bbox_to_anchor=(0.76, -0.19),
#                      shadow=True, prop={'size': 15}, ncol=5)
fig.add_artist(legend1)
fig.add_artist(legend2)

# fig.legend(loc='upper center',
#            bbox_to_anchor=(0.5, 0.04),
#            fancybox=True, shadow=True, ncol=5, prop={'size': 15})

fig.suptitle(titile, fontsize=20)



# for l in range(2):
for ax in axs:
    ax.grid(visible=True, which='major', linestyle='--', alpha=0.5)
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', linestyle=':', alpha=0.2)
#
# lines_labels = [ax.get_legend_handles_labels() for ax in axs[:,0]]
# lines, labels = [sum(_, []) for _ in zip(*lines_labels)]

#
# cols = ['{}'.format(col) for col in ['Marsrover 4x4', 'Marsrover 8x8', 'Box']]
# if plot_reward:
#     rows = ['{}'.format(row) for row in ['reward', 'consumption']]
# else:
#     rows = ['{}'.format(row) for row in ['main regret', 'constraint violation']]
#
# for ax, col in zip(axs[0], cols):
#     ax.set_title(col)
#
# for ax, row in zip(axs[:,0], rows):
#     ax.set_ylabel(row, rotation=90, fontsize=12)
#     ax.yaxis.label.set_size(12)
#

plt.savefig(folder + '/' + 'av_reward_cost' + '.png', bbox_inches='tight')
plt.show()


# Marsrover 4x4
# %  python -u src/run_bonusterm.py --alg cucrl_conservative --env gridworld --rounds 9000 --num_runs 20
# % python -u src/run_bonusterm.py --alg cucrl_optimistic --env gridworld --rounds 9000 --num_runs 20
# %  python -u src/run.py --alg posterior_transitions --env gridworld --rounds 9000 --num_runs 20


# Marsrover 8x8
# %  python -u src/run.py --alg posterior_transitions --env marsrover_gridworld --rounds 25000 --num_runs 20
# %  python -u src/run_bonusterm.py --alg cucrl_optimistic2 --env marsrover_gridworld --rounds 25000 --num_runs 20
# %  python -u src/run_bonusterm.py --alg cucrl_conservative --env marsrover_gridworld --rounds 25000 --num_runs 20
