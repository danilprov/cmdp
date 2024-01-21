import numpy as np
from pulp import *
import itertools

#from src.codebase.rl_solver.rl_solver import RLSolver
from codebase.rl_solver.rl_solver import RLSolver

class LinProgSolverExtnded(RLSolver):
    """

    """

    def __init__(self, M=None, args=None):
        super(LinProgSolverExtnded, self).__init__(M=M, args=args)
        self.States = range(self.S)
        self.Actions = range(self.A)
        self.Steps = range(self.H)

    def __call__(self, P=None, R=None, C=None, eps=0.001, use_constraints=True, last_state=None):
        R = self.R if R is None else R
        P = self.P if P is None else P
        C = self.C if C is None else C
        s0 = self.M.s0.argmax() if last_state is None else last_state

        if not hasattr(eps, "__len__"):
            eps = np.ones((self.S, self.A)) * eps

        cons_opt_prob = LpProblem("CMDP", LpMaximize)

        # define decision variable, mu(s,a,h,s') -- extended occupancy measure
        mu = LpVariable.dicts("occupancy measure",
                              itertools.product(self.Steps, self.States, self.Actions, self.States), lowBound=0,
                              upBound=1, cat='Continuous')

        # add objective
        cons_opt_prob += lpSum([lpSum([R[s, a] * lpSum([mu[h, s, a, ss] for ss in self.States]) for s, a in
                                itertools.product(self.States, self.Actions)]) for h in self.Steps])

        # initial state
        cons_opt_prob += lpSum([mu[0, s0, a, ss] for ss, a in itertools.product(self.States, self.Actions)]) == 1

        # constraints
        if use_constraints:
            # add constraints
            for i in range(self.d):
                cons_opt_prob += lpSum([lpSum([C[i, s, a] * lpSum([mu[h, s, a, ss] for ss in self.States]) for s, a in
                                               itertools.product(self.States, self.Actions)]) for h in
                                        self.Steps]) <= self.H * self.budget[i] #* 0.475

        # total mass
        for h in self.Steps:
            cons_opt_prob += lpSum(
                [mu[h, s, a, ss] for s, a, ss in itertools.product(self.States, self.Actions, self.States)]) == 1

        # flow conservation
        for h in self.Steps[:-1]:
            for s_next in self.States:
                cons_opt_prob += lpSum(
                    [mu[h, s, a, s_next] for s, a in itertools.product(self.States, self.Actions)]) == lpSum(
                    [mu[h+1, s_next, a, s] for s, a in itertools.product(self.States, self.Actions)])

        # OFU requirement
        for h in self.Steps:
            for s in self.States:
                for a in self.Actions:
                    for s_next in self.States:
                        cons_opt_prob += mu[h, s, a, s_next] <= lpSum([mu[h, s, a, ss] for ss in self.States]) * P[
                            s, a, s_next] + eps[s, a]
                        cons_opt_prob += mu[h, s, a, s_next] >= lpSum([mu[h, s, a, ss] for ss in self.States]) * P[
                            s, a, s_next] - eps[s, a]

        cons_opt_prob.solve(PULP_CBC_CMD(msg=0))
        #cons_opt_prob.solve()
        # The status of the solution is printed to the screen
        print("Status:", LpStatus[cons_opt_prob.status])
        # if LpStatus[cons_opt_prob.status] == 'Infeasible':
        #     pi_list = np.ones([self.S, self.A]) * 0.25
        # else:
        #     pi_list = self.__get_pi_list__(cons_opt_prob, P)
        # return pi_list

        if LpStatus[cons_opt_prob.status] == 'Infeasible':
            return np.zeros((self.H, self.S, self.A, self.S))

        varsdict = {}
        for v in cons_opt_prob.variables():
            varsdict[v.name] = v.varValue

        mu_hsas = np.zeros((self.H, self.S, self.A, self.S))
        for h in self.Steps:
            for s in self.States:
                for a in self.Actions:
                    for ss in self.States:
                        mu_hsas[h, s, a, ss] = varsdict[f'occupancy_measure_({h},_{s},_{a},_{ss})']

        return mu_hsas.sum(axis=3)

if __name__ == '__main__':
    from src.codebase.mdp import FiniteHorizonCMDP
    from src.codebase.environments.gridworld import GridWorld
    from src.codebase.environments.box_gridworld import BoxGridWorld

    path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    """
    marsrover example
    """

    d = 1
    horizon = 50
    args = {'map': "4x4_gwsc", 'randomness': 0, 'd': d, 'infinite': True, 'horizon': horizon}
    gridworld = GridWorld(args)
    [mdp_values, Si, Ai] = gridworld.encode()

    s0, P, R, C = mdp_values
    budget = [0.2]
    M = FiniteHorizonCMDP(*mdp_values, d, budget, gridworld.H, Si, Ai, gridworld.terminals)
    lin_opt = LinProgSolverExtnded(M)
    p_hat = np.zeros((16, 4, 16))
    c_hat = np.zeros((1, 16, 4))
    r_hat = np.zeros((16, 4))
    mu_hsa = lin_opt()#R=r_hat, C=c_hat, P=p_hat)
    #mu_hsa = mu_hsas.sum(axis=3)

    for h in range(horizon):
        policy = mu_hsa[h, :, :] / mu_hsa[h, :, :].sum(axis=1, keepdims=True)
        grid_index = map(Si.lookup, range(len(Si)))
        grid_policy = dict(zip(grid_index, policy))
        for r in range(1, 5):
            action_list = []
            for c in range(1, 5):
                s = (r, c)
                if len(np.unique(grid_policy[s])) == 1:
                    action_list.append('*')
                    continue
                bestA = np.argmax(grid_policy[s])
                action_list.append(Ai.lookup(bestA))
            print([i for i in action_list])
        print('\n')

    list_of_policies = np.zeros((M.H, M.S, M.A))
    for h in range(M.H):
        list_of_policies[h, :, :] = mu_hsa[h, :, :] / mu_hsa[h, :, :].sum(axis=1, keepdims=True)
        # replace nan with 0.25 (a bit weird way of doing this, but this is becuase 3d array)
        list_of_policies[h, np.isnan(list_of_policies[h, :, :])] = 0.25

    (p, r, c, v, last_state, sub_rewards) = lin_opt.run(list_of_policies, horizon, (4,3))
    print('a')
