import numpy as np

from codebase.algs.base import BaseAlgorithm


class LagrangianPosteriorSampling(BaseAlgorithm):
    def __init__(self, G=None, M=None, args=None, planner=None):
        super(LagrangianPosteriorSampling, self).__init__(G=G, M=M, args=args, planner=planner)

        # class specific parameters
        self.k = 0 # episode number
        self.π_explore = np.ones((self.num_states, self.num_actions)) * 1 / self.num_actions
        self._lambda = np.zeros(self.M.d)
        try:
            self.known_rewards = args.known_rewards
            self.conplanner_iter = args.conplanner_iter
            self._lambda_lr = args.optimistic_lambda_lr
        except:
            self.known_rewards = False
            self.conplanner_iter = 10
            self._lambda_lr = [0.2]

    def conplanner(self, p_hat, r_hat, c_hat):
        _lambda_lr = self._lambda_lr
        _lambda_avg = np.zeros(self.M.d)

        for _ in range(self.conplanner_iter):
            pseudo_reward = r_hat + np.einsum("i,isa->sa", self._lambda, c_hat)
            π_list = self.planner(P=p_hat, R=pseudo_reward)['pi_list'][0]
            # c = self.planner.value_evaluation(π_list, P=p_hat, R=pseudo_reward, C=c_hat)['constraint']
            c = self.planner.monte_carlo_evaluation(π_list, P=p_hat, R=pseudo_reward, C=c_hat)['constraint']

            self._lambda = np.minimum(0, self._lambda - _lambda_lr * (c - self.budget))
            _lambda_avg += self._lambda

        _lambda_avg /= self.conplanner_iter
        pseudo_reward = r_hat + np.einsum("i,isa->sa", _lambda_avg, c_hat)
        π_list = self.planner(P=p_hat, R=pseudo_reward)['pi_list'][0]

        return π_list

    def __call__(self):
        self.k += 1
        # exploration phase
        (p, r, c, v, last_state, _) = self.planner.run(self.π_explore)
        # Update Counts
        self.p_sum += p
        self.r_sum += r
        self.c_sum += c
        self.visitation_sum += v

        # exploitation phase
        p_hat = np.zeros((self.num_states, self.num_actions, self.num_states))
        r_hat = np.zeros((self.num_states, self.num_actions))
        c_hat = np.zeros((self.M.d, self.num_states, self.num_actions))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                p_hat[s, a, :] = np.random.dirichlet(np.maximum(self.p_sum[s, a, :], 1))

                if not self.known_rewards:
                    visitation = self.p_sum[s, a, :].sum()
                    if visitation > 0:
                        r_hat[s, a] = self.r_sum[s, a] / visitation
                        c_hat[:, s, a] = self.c_sum[:, s, a] / visitation

        π_list = self.conplanner(p_hat, r_hat, c_hat)

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

        metrics = {'mixture_reward': self.policy.reward,
                   'mixture_constraint': self.policy.constraint,
                   'current_reward': values['reward'],
                   'current_constraint': values['constraint'],
                   'alg': 'lagr_posterior',
                   'policy': π_list}

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

    alg = LagrangianPosteriorSampling(G=gridworld, M=M, planner=solver)

    results = []
    for round in range(100):
        [metrics, status_str] = alg(solver)

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