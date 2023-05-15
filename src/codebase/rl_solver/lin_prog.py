import numpy as np
from pulp import *
import itertools

from codebase.rl_solver.rl_solver import RLSolver


class LinProgSolver(RLSolver):
    """
    Linear programming for infinite-horizon CMDPs.

    Altman 1999 (https://www-sop.inria.fr/members/Eitan.Altman/TEMP/h.pdf) describes
    how CMDPs can be reformulated in terms of linear programs through occupancy measure.
    """

    def __init__(self, M=None, args=None):
        super(LinProgSolver, self).__init__(M=M, args=args)
        self.States = range(self.S)
        self.Actions = range(self.A)

    def __call__(self, P=None, R=None, C=None, use_constraints=True):
        R = self.R if R is None else R
        P = self.P if P is None else P
        C = self.C if C is None else C

        cons_opt_prob = LpProblem("CMDP", LpMaximize)

        # define decision variable, mu(s,a) -- occupancy measure
        mu = LpVariable.dicts("occupancy measure", itertools.product(self.States, self.Actions), lowBound=0, upBound=1,
                              cat='Continuous')

        # add objective
        cons_opt_prob += lpSum([R[s, a] * mu[s, a] for s, a in itertools.product(self.States, self.Actions)])

        if use_constraints:
            # add constraints
            for i in range(self.d):
                cons_opt_prob += lpSum(
                    [C[i, s, a] * mu[s, a] for s, a in itertools.product(self.States, self.Actions)]) <= self.budget[i]

        cons_opt_prob += lpSum([mu[s, a] for s, a in itertools.product(self.States, self.Actions)]) == 1

        for s in self.States:
            cons_opt_prob += lpSum([mu[s, a] for a in self.Actions]) <= lpSum(
                [mu[ss, aa] * P[ss, aa, s] for ss, aa in itertools.product(self.States, self.Actions)])

        cons_opt_prob.solve(PULP_CBC_CMD(msg=0))
        #cons_opt_prob.solve()
        # The status of the solution is printed to the screen
        #print("Status:", LpStatus[cons_opt_prob.status])
        pi_list = self.__get_pi_list__(cons_opt_prob)
        return pi_list


    def __get_pi_list__(self, solution):
        varsdict = {}
        for v in solution.variables():
            varsdict[v.name] = v.varValue

        pi_list = np.zeros_like(self.R)
        for s in self.States:
            mu_s = sum([varsdict[f'occupancy_measure_({s},_{a})'] for a in self.Actions])
            mu_sa = np.array([varsdict[f'occupancy_measure_({s},_{a})'] for a in self.Actions])
            pi_list[s, :] = mu_sa / mu_s

        # replace nan with 1/|A|, otherwise sample(polisy(s,:)) always returns 0 action
        pi_list[np.isnan(pi_list)] = 1 / self.A

        return pi_list


if __name__ == '__main__':
    from codebase.mdp import FiniteHorizonCMDP
    from codebase.environments.gridworld import GridWorld
    from codebase.environments.box_gridworld import BoxGridWorld

    path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    """
    marsrover example
    """

    d = 1
    args = {'map': "8x8_marsrover", 'randomness': 0.1, 'd': d, 'infinite': True, 'horizon': 20}
    gridworld = GridWorld(args)
    [mdp_values, Si, Ai] = gridworld.encode()

    s0, P, R, C = mdp_values
    budget = [0.1]
    M = FiniteHorizonCMDP(*mdp_values, d, budget, gridworld.H, Si, gridworld.terminals)
    lin_opt = LinProgSolver(M)
    pi_list = lin_opt()

    value = lin_opt.monte_carlo_evaluation(pi_list)

    grid_index = map(Si.lookup, range(len(Si)))
    grid_policy = dict(zip(grid_index, pi_list))
    gridworld.showLearning(grid_policy, path = path + '/log')

    """
    box example
    """

    d = 1
    args = {'map': "6x6_box", 'randomness': 0.1, 'd': d, 'infinite': True, 'horizon': 100}
    gridworld = BoxGridWorld(args)
    [mdp_values, Si, Ai] = gridworld.encode()

    s0, P, R, C = mdp_values
    budget = [20]
    M = FiniteHorizonCMDP(*mdp_values, d, budget, gridworld.H, Si, gridworld.terminals)
    lin_opt = LinProgSolver(M)
    pi_list = lin_opt()

    # plot Box
    states_ = np.array(list(map(Si.lookup, range(121))))
    box_states = [np.array([2, 2]), np.array([2, 3]), np.array([3, 2])]
    for box_state in box_states:
    #box_state = np.array([2, 2])
    #start_state = None# np.array([2, 2])
        grid_policy = {}
        for s_num in np.where(np.all(states_[:, -2:] == box_state, axis=1))[0]:
            s = Si.lookup(s_num)
            grid_policy[(s[0], s[1])] = pi_list[s_num]
        gridworld.showLearning(grid_policy, box_position=box_state, path = path + '/log')

    # plot Marsrover 8x8
    # states_ = range(64)
    # grid_policy = {}
    # for s_num in states_:
    #     s = Si.lookup(s_num)
    #     grid_policy[(s[0], s[1])] = pi_list[s_num]
    # gridworld.showLearning(grid_policy)

    value = lin_opt.monte_carlo_evaluation(pi_list)
    print('a')