# https://arxiv.org/pdf/2001.09377.pdf

import numpy as np

from codebase.algs.base import BaseAlgorithm


class CUCRLAlgorithm(BaseAlgorithm):
    def __init__(self, G=None, M=None, args=None, planner=None):
        super(CUCRLAlgorithm, self).__init__(G=G, M=M, args=args, planner=planner)

        # class specific parameters
        self.k = 0 # episode number
        self.π_explore = np.ones((self.num_states, self.num_actions)) * 1 / self.num_actions

        try:
            self.known_transitions = args.known_transitions
            self.bonus_coef = args.bonus_coef
            self.power = args.power
            self.ucb_type = args.ucb_type
        except:
            self.known_transitions = False
            self.bonus_coef = 0.01
            self.power = 2
            self.ucb_type = 'cucrl_pessimistic'

    def __call__(self):
        self.k += 1
        # exploration phase
        (p, r, c, v, last_state, _) = self.planner.run(self.π_explore, self.H)
        # Update Counts
        self.p_sum += p
        self.r_sum += r
        self.c_sum += c
        self.visitation_sum += v

        # exploitation phase
        p_hat = np.zeros((self.num_states, self.num_actions, self.num_states))
        r_hat = np.zeros((self.num_states, self.num_actions))
        c_hat = np.zeros((self.M.d, self.num_states, self.num_actions))
        bonus = np.zeros((self.num_states, self.num_actions))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                visitation = self.p_sum[s, a, :].sum()
                if visitation == 0:
                    p_hat[s, a, s] = 1
                    bonus[s, a] = 1
                else:
                    p_hat[s, a, :] = self.p_sum[s, a, :] / visitation
                    r_hat[s, a] = self.r_sum[s, a] / visitation
                    c_hat[:, s, a] = self.c_sum[:, s, a] / visitation
                    bonus[s, a] = np.sqrt(1.0 / visitation) * self.bonus_coef

        if self.ucb_type == 'cucrl_pessimistic':
            π_list = self.planner(p_hat, r_hat + bonus, c_hat + bonus)
        elif self.ucb_type == 'cucrl_optimistic':
            π_list = self.planner(p_hat, r_hat + bonus, c_hat - bonus)
        elif self.ucb_type == 'cucrl_transitions':
            bonus3D = np.repeat(bonus[..., None], self.num_states, axis=2)
            π_list = self.planner(p_hat, r_hat, c_hat, bonus3D)

        (p, r, c, v, last_state, _) = self.planner.run(π_list, (self.k - 1) * self.H)
        # Update Counts
        self.p_sum += p
        self.r_sum += r
        self.c_sum += c
        self.visitation_sum += v

        # values = rl_solver.value_evaluation_dp(π_list, max_value_iter=1000)
        values1 = self.planner.monte_carlo_evaluation(self.π_explore)
        values = self.planner.monte_carlo_evaluation(π_list)
        values['reward'] =  values['reward'] * (self.k - 1) / self.k + values1['reward'] * 1 / self.k
        values['constraint'] = values['constraint'] * (self.k - 1) / self.k + values1['constraint'] * 1 / self.k
        self.policy.add_response([values['reward'], values['constraint']])
        status = f' current_reward: {values["reward"]}\n'
        status += f' current_constraint: {values["constraint"]}\n'
        status += f' mixture_reward: {self.policy.reward}\n'
        status += f' mixture_constraint: {self.policy.constraint}\n'
        status += f' last_state: {last_state}'

        metrics = {'mixture_reward': self.policy.reward,
                   'mixture_constraint': self.policy.constraint,
                   'current_reward': values['reward'],
                   'current_constraint': values['constraint'],
                   'alg': self.ucb_type,
                   'policy': π_list}
                   # 'num_trajs': self.planner.stats['num_trajs'],
                   # 'expected_consumption': self.planner.stats['expected_consumption'],
                   # 'training_consumpution': self.planner.stats['training_consumpution']}
        return (metrics, status)


if __name__ == '__main__':
    import pandas as pd

    from codebase.mdp import FiniteHorizonCMDP
    from codebase.environments.gridworld import GridWorld
    from codebase.rl_solver.lin_prog import LinProgSolver

    d = 1
    args = {'map': "4x4_gwsc", 'horizon': 50, 'randomness': 0, 'd': d, 'infinite': True}
    gridworld = GridWorld(args)
    [mdp_values, Si, Ai] = gridworld.encode()

    s0, P, R, C = mdp_values
    budget = [0.2]

    M = FiniteHorizonCMDP(*mdp_values, d, budget, gridworld.H, Si, gridworld.terminals)
    solver = LinProgSolver(M=M)

    alg = CUCRLAlgorithm(G=gridworld, M=M, planner=solver)

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