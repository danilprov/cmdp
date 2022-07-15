import numpy as np

from codebase.algs.base import BaseAlgorithm


class PosteriorSampling(BaseAlgorithm):
    def __init__(self, G=None, M=None, args=None, planner=None):
        super(PosteriorSampling, self).__init__(G=G, M=M, args=args, planner=planner)

        # class specific parameters
        self.k = 0 # episode number
        self.π_explore = np.ones((self.num_states, self.num_actions)) * 1 / self.num_actions
        try:
            self.posterior_type = args.posterior_type
            self.bonus_coef = args.bonus_coef
        except:
            self.posterior_type = 'rewards'
            self.bonus_coef = 0.01

    def __call__(self):
        self.k += 1

        p_hat = np.zeros((self.num_states, self.num_actions, self.num_states))
        r_hat = np.zeros((self.num_states, self.num_actions))
        c_hat = np.zeros((self.M.d, self.num_states, self.num_actions))
        bonus = np.zeros((self.num_states, self.num_actions))

        if self.posterior_type == 'full':
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    visitation = self.p_sum[s, a, :].sum()
                    # sample transitions
                    p_hat[s, a, :] = np.random.dirichlet(np.maximum(self.p_sum[s, a, :], 1))
                    # sample reward and costs
                    if visitation > 0:
                        bonus[s, a] = np.sqrt(1.0 / visitation) * self.bonus_coef
                        r_hat[s, a] = np.random.normal(self.r_sum[s, a] / visitation, bonus[s, a])
                        c_hat[:, s, a] = np.random.multivariate_normal(self.c_sum[:, s, a] / visitation,
                                                                       np.eye(self.M.d) * bonus[s, a])

        elif self.posterior_type == 'transitions':
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    visitation = self.p_sum[s, a, :].sum()
                    # sample transitions
                    p_hat[s, a, :] = np.random.dirichlet(np.maximum(self.p_sum[s, a, :], 1))
                    # average reward and costs
                    if visitation > 0:
                        r_hat[s, a] = self.r_sum[s, a] / visitation
                        c_hat[:, s, a] = self.c_sum[:, s, a] / visitation

        elif self.posterior_type == 'rewards':
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    visitation = self.p_sum[s, a, :].sum()
                    # average transitions
                    if visitation == 0:
                        p_hat[s, a, s] = 1
                        bonus[s, a] = 1
                    else:
                        p_hat[s, a, :] = self.p_sum[s, a, :] / visitation
                        bonus[s, a] = np.sqrt(1.0 / visitation) * self.bonus_coef
                        # sample reward and costs
                        r_hat[s, a] = np.random.normal(self.r_sum[s, a] / visitation, bonus[s, a])
                        c_hat[:, s, a] = np.random.multivariate_normal(self.c_sum[:, s, a] / visitation,
                                                                       np.eye(self.M.d) * bonus[s, a])

        π_list = self.planner(p_hat, r_hat, c_hat)

        (p, r, c, v, last_state, _) = self.planner.run(π_list, self.k * self.H)
        # Update Counts
        self.p_sum += p
        self.r_sum += r
        self.c_sum += c
        self.visitation_sum += v

        # values = rl_solver.value_evaluation_dp(π_list, max_value_iter=1000)
        values = self.planner.monte_carlo_evaluation(π_list)
        self.policy.add_response([values['reward'], values['constraint']])
        status = f' current_reward: {values["reward"]}\n'
        status += f' current_constraint: {values["constraint"]}\n'
        status += f' mixture_reward: {self.policy.reward}\n'
        status += f' mixture_constraint: {self.policy.constraint}\n'
        status += f' last_state: {last_state}'

        alg_name = 'posterior_' + self.posterior_type

        metrics = {'mixture_reward': self.policy.reward,
                   'mixture_constraint': self.policy.constraint,
                   'current_reward': values['reward'],
                   'current_constraint': values['constraint'],
                   'alg': alg_name,
                   'policy': π_list}
                   # 'num_trajs': self.planner.stats['num_trajs'],
                   # 'expected_consumption': self.planner.stats['expected_consumption'],
                   # 'training_consumpution': self.planner.stats['training_consumpution']}
        return (metrics, status)


if __name__ == '__main__':
    import pandas as pd

    from codebase.mdp import FiniteHorizonCMDP
    from codebase.environments.gridworld import GridWorld
    from codebase.rl_solver.planner import ValueIteration
    from codebase.rl_solver.lin_prog import LinProgSolver

    d = 1
    args = {'map': "4x4_gwsc", 'horizon': 50, 'randomness': 0, 'd': d, 'infinite': True}
    gridworld = GridWorld(args)
    [mdp_values, Si, Ai] = gridworld.encode()

    s0, P, R, C = mdp_values
    budget = [0.2]

    M = FiniteHorizonCMDP(*mdp_values, d, budget, gridworld.H, Si, gridworld.terminals)
    solver = LinProgSolver(M=M)

    alg = PosteriorSampling(G=gridworld, M=M, planner=solver)

    results = []
    for round in range(100):
        [metrics, status_str] = alg()

        metrics['round'] = round
        results.append(metrics)
        print(f'Round: {round}/{100}')
        print(f'{status_str}')
        print(f'----------------------------')

    df = pd.DataFrame(results, \
                      columns=np.hstack(['round', 'mixture_reward', 'mixture_constraint', \
                                         'current_reward', 'current_constraint', 'alg', 'num_trajs', \
                                         'expected_consumption', 'training_consumpution']))


    print('a')