import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import ntpath
import pickle

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_mean_std(array_2d):
    mean = np.mean(array_2d, axis=0)
    std = np.std(array_2d, axis=0)

    return mean, std

def parse_result(results):
    """
    results - dictionary
    """
    num_rounds = len(results['run0']['current_reward'])
    reward = np.zeros((len(results), num_rounds))
    constraint = np.zeros((len(results), num_rounds))

    for i in range(len(results)):
        for j in range(num_rounds):
            run = 'run' + str(i)
            reward[i, j] = results[run]['mixture_reward'][j]
            constraint[i, j] = results[run]['mixture_constraint'][j]

    mean_reward, std_reward = get_mean_std(reward)
    mean_constraint, std_constraint = get_mean_std(constraint)

    return mean_reward, std_reward, mean_constraint, std_constraint

path = r'C:/Users/provo501/Documents/GitHub/multi-objective-rl/src/log/EWRL/'
list_of_folders = glob.glob(path + '*')
latest_folder = max(list_of_folders, key=os.path.getctime)
# latest_folder = path + '20220512141436_gridworld_4x4_gwsc_20'

all_files = glob.glob(latest_folder + '/' + "*.pickle")
title = latest_folder[82:]
fig, axs = plt.subplots(2, sharex=True)
fig.suptitle('Average reward and cost (' + title + ')')

for filename in all_files:
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    alg_name = path_leaf(filename)[:-7]

    mean_reward, std_reward, mean_constraint, std_constraint = parse_result(results)

    axs[0].plot(mean_reward, label=alg_name)
    axs[0].fill_between(range(len(mean_reward)), mean_reward - std_reward, mean_reward + std_reward, alpha=0.4)
    axs[1].plot(mean_constraint)
    axs[1].fill_between(range(len(mean_reward)), mean_constraint - std_constraint, mean_constraint + std_constraint,
                        alpha=0.4)


axs[1].set(xlabel='epoch', ylabel='cost')
axs[0].set(ylabel='reward')
for ax in axs:
    ax.grid(b=True, which='major', linestyle='--', alpha=0.5)
    ax.minorticks_on()
    ax.grid(b=True, which='minor', linestyle=':', alpha=0.2)

lines_labels = [ax.get_legend_handles_labels() for ax in axs]
lines, labels = [sum(_, []) for _ in zip(*lines_labels)]
fig.legend(lines, labels, loc='center right')
plt.savefig(latest_folder + '/' + 'av_reward_cost' + '.png', bbox_inches='tight')
plt.show()
