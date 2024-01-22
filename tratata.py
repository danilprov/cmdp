import matplotlib.pyplot as plt
import numpy as np
from pulp import *

from src.codebase.rl_solver.lin_prog import LinProgSolver
from src.codebase.mdp import FiniteHorizonCMDP


def simulate(policy, last_state, env):
    P = env.P
    R = env.R
    C = env.C
    a = np.random.choice(actions, p=policy[last_state])
    reward = R[last_state, a]
    cost = C[0, last_state, a]
    next_state = np.random.choice(states, p=P[last_state, a, :])

    return a, reward, cost, next_state


def running_mean(arr, n=10):
    cumsum = np.cumsum(arr)
    return (cumsum[n:] - cumsum[:-n]) / float(n)


# define environment
theta = 0.9
budget = [0.5275]
states = range(2)
actions = range(2)
true_P = np.array([[[1, 0], [1 - theta, theta]], [[1, 0], [1, 0]]])
R = np.array([[1, 1], [0, 0]])
C = np.array([[[1, 1], [0, 0]]])
s0 = 0
d = 1
true_env = FiniteHorizonCMDP(s0, true_P, R, C, d, budget, None, None, None, None)

# find optimal solution for true env
lin_prog_solver = LinProgSolver(true_env)
lin_prog_solution = lin_prog_solver(return_raw_solution=True)
optimal_om = lin_prog_solver.__get_pi_list__(lin_prog_solution, true_P, return_policy=False)
optimal_reward = (R * optimal_om).sum().sum()
optimal_cost = (C[0, :, :] * optimal_om).sum().sum()

# do RL
T = 1500
num_runs = 5
REGRET = [[], []]
COST = [[], []]
CNTS = [[], []]
THETAS = [[], []]

# define two policies: shortest (exploratative) and greedy (exploitative)
policy_shortest = [[0, 1], [1, 0]]
policy_greedy = [[1, 0], [1, 0]]
policies = [policy_shortest, policy_greedy]

# run experiment
for j, infeas_policy in enumerate(policies):
    for run in range(num_runs):
        np.random.seed(run + 1)
        main_regret = [0]
        cost_regret = [0]
        thetas = []

        s0a1s0_visits = 1
        s0a1s1_visits = 1
        cnt = 0
        last_state = s0
        for i in range(T):
            # due to beta dist properties arguments below should be reversed
            # i.e., s0a0s1 first and then s0a1s0
            theta_hat = np.random.beta(s0a1s1_visits, s0a1s0_visits)
            thetas.append(theta_hat)
            # define plausible transitions
            P = np.array([[[1, 0], [1 - theta_hat, theta_hat]],
                          [[1, 0], [1, 0]]])
            env = FiniteHorizonCMDP(s0, P, R, C, d, budget, None, None, None, None)

            # solve LP
            lin_prog_solver = LinProgSolver(env)
            lin_prog_solution = lin_prog_solver()

            if LpStatus[lin_prog_solution.status] == 'Infeasible':
                policy = infeas_policy
            else:
                policy = lin_prog_solver.__get_pi_list__(lin_prog_solution, P)
                cnt += 1

            # make a step in the environment
            a, reward, cost, next_state = simulate(policy, last_state, true_env)

            # update counters and metrics
            if a == 1:
                if next_state == 0:
                    s0a1s0_visits += 1
                else:
                    s0a1s1_visits += 1
            #main_regret.append(main_regret[-1] + max(optimal_reward - reward, 0))
            main_regret.append(reward)
            cost_regret.append(cost)
            last_state = next_state

        REGRET[j].append(np.cumsum(np.maximum(optimal_reward - running_mean(main_regret, n=12), 0)))
        COST[j].append(running_mean(cost_regret, n=50))
        CNTS[j].append(cnt)
        THETAS[j].append(running_mean(thetas))

    print(CNTS[j])

fig, axs = plt.subplots(1, 3,figsize=(10,3))
axs[0].plot(np.mean(REGRET[1], axis=0), label='CMDP-PSRL',linestyle='-.')
axs[0].plot(np.mean(REGRET[0], axis=0), label='PSConRL',linestyle='-.')
axs[0].set(xlabel='rounds', ylabel='main regret')
axs[0].legend()

axs[1].hlines(y=optimal_cost, xmin=0, xmax=T, colors='#7f7f7f', linestyles='--', lw=1)
axs[1].plot(np.mean(COST[1], axis=0), label='CMDP-PSRL',linestyle='-.')
axs[1].plot(np.mean(COST[0], axis=0), label='PSConRL',linestyle='-.')
axs[1].set(xlabel='rounds', ylabel='average cost')

axs[2].hlines(y=theta, xmin=0, xmax=T, colors='#7f7f7f', linestyles='--', lw=1)
axs[2].plot(np.mean(THETAS[1], axis=0), label='CMDP-PSRL',linestyle='-.')
axs[2].plot(np.mean(THETAS[0], axis=0), label='PSConRL',linestyle='-.')
axs[2].set(xlabel='rounds', ylabel='sampled θ')
axs[2].set_ylim(0.6, None)

for ax in axs:
    ax.grid(visible=True, which='major', linestyle='--', alpha=0.5)
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', linestyle=':', alpha=0.2)

lines_labels = ax.get_legend_handles_labels()
#lines, labels = [sum(_, []) for _ in zip(*lines_labels)]
lines, labels = lines_labels[0], lines_labels[1]
# fig.legend(lines, labels,
#            loc='upper center',
#            bbox_to_anchor=(0.5, -0.01),
#            fancybox=True, shadow=True, ncol=5)
fig.tight_layout()
plt.savefig('counterexample' + '.png', bbox_inches='tight')
plt.show()
print('a')