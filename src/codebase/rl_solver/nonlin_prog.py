import numpy as np
from scipy import optimize

from src.codebase.rl_solver.rl_solver import RLSolver


class NonlinProgSolver(RLSolver):
    """
    According to Algorithm 1 form (https://arxiv.org/pdf/2002.12435.pdf) define decision variables:
            mu(s,a) -- occupancy measure
            p'(s,a,s') -- "candidate" optimistic transitions
    """

    def __init__(self, M=None, args=None):
        super(NonlinProgSolver, self).__init__(M=M, args=args)
        self.States = range(self.S)
        self.Actions = range(self.A)

    def objective(self, x, R, sign=-1.):
        _x = x[:self.S * self.A]
        R = R.flatten()
        return sign * np.dot(R, _x)

    def constraint1(self, x):
        _x = x[:self.S * self.A]
        # sum mu[s,a] == 1
        A = np.ones(self.S * self.A)
        b = 1

        return np.dot(A, _x) - b

    def constraint2(self, x):
        _x = x[:self.S * self.A]
        # mu[s,a] >= 0 -> - mu[s,a] <= 0
        A = np.eye(self.S * self.A)
        b = np.zeros(self.S * self.A)

        return np.dot(A, _x)

    def constraint3(self, x, C):
        _x = x[:self.S * self.A]
        C = C.reshape(self.d, -1)
        return self.budget - np.dot(C, _x)

    def constraint4(self, x):
        mu, P = self.toMuP(x)
        return np.einsum("sap,sa->p", P, mu) - np.sum(mu, axis=1)
        # _x = x.reshape(self.S, self.A)
        # return np.einsum("sap,sa->p", self.P, _x) - np.sum(_x, axis=1)

    """
    Optimization over the second parameter is being run over 
    the confidence set Ct = {p' : |p'(s,a,s') - p(s,a,s)| < eps_t for each (s,a)}.
    To plug it into the constrained problem I break the absolute value into two parts
    
    Part 1:  p' <=   (p + eps) - constraint 5
    Part 2: -p' <= - (p - eps) - constraint 6
    """

    def constraint5(self, x, P, eps):
        # p' <= p + eps
        _x = x[self.S * self.A:]
        if isinstance(eps, float):
            return P.flatten() + eps - _x
        return P.flatten() + eps.flatten() - _x

    def constraint6(self, x, P, eps):
        _x = x[self.S * self.A:]
        if isinstance(eps, float):
            return P.flatten() + eps - _x
        return _x - (P.flatten() - eps.flatten())

    def __call__(self, P=None, R=None, C=None, eps=None, mu0=None):
        R = self.R if R is None else R
        P = self.P if P is None else P
        C = self.C if C is None else C
        eps = 0.001 if eps is None else eps

        # bnds = tuple([(0.0, 1.0) for _ in range(self.S * self.A)])
        # x0 = np.random.rand(self.S * self.A)
        # result = optimize.minimize(self.objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
        # mu = result.x.reshape(self.S, self.A)

        cons = [{'type': 'eq', 'fun': self.constraint1},
                {'type': 'ineq', 'fun': self.constraint2},
                {'type': 'ineq', 'fun': self.constraint3, 'args': (C, )},
                {'type': 'eq', 'fun': self.constraint4},
                {'type': 'ineq', 'fun': self.constraint5, 'args': (P, eps)},
                {'type': 'ineq', 'fun': self.constraint6, 'args': (P, eps)}]

        if mu0 is None:
            mu0 = np.random.rand(self.S, self.A)
        P0 = P  # np.zeros((self.S, self.A, self.S))
        x0 = self.toVector(mu0, P0)

        bnds = tuple([(0.0, 1.0) for _ in range(self.S * self.A + self.S ** 2 * self.A)])
        arguments = (R,)

        result = optimize.minimize(self.objective, x0, method='SLSQP', args = arguments,
                                   bounds=bnds, constraints=cons, options={'maxiter':30})
        # print(result)
        if result.success:
            mu, P = self.toMuP(result.x)
        else:
            mu = np.ones((self.S, self.A)) * 0.25

        return mu / mu.sum(axis=1, keepdims=True)


    def toVector(self, mu, p):
        assert mu.shape == (self.S, self.A)
        assert p.shape == (self.S, self.A, self.S)
        return np.hstack([mu.flatten(), p.flatten()])

    def toMuP(self, x):
        assert x.shape == (self.S * self.A + self.S**2 * self.A,)
        return x[:self.S * self.A].reshape(self.S, self.A), x[self.S * self.A:].reshape(self.S, self.A, self.S)


if __name__ == '__main__':
    from src.codebase.mdp import FiniteHorizonCMDP
    from src.codebase.environments.gridworld import GridWorld
    import os

    path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    d = 1
    args = {'map': "4x4_gwsc", 'randomness': 0, 'd': d, 'infinite': True, 'horizon': 100}
    gridworld = GridWorld(args)
    [mdp_values, Si, Ai] = gridworld.encode()

    s0, P, R, C = mdp_values
    budget = [0.2]
    M = FiniteHorizonCMDP(*mdp_values, d, budget, gridworld.H, Si, Ai, gridworld.terminals)
    nonlin_opt = NonlinProgSolver(M)

    pi_list = nonlin_opt()

    value = nonlin_opt.monte_carlo_evaluation(pi_list)
    print(value)
    random_policy = np.ones_like(pi_list) * 0.25
    value_random = nonlin_opt.monte_carlo_evaluation(random_policy)

    grid_index = map(Si.lookup, range(len(Si)))
    grid_policy = dict(zip(grid_index, pi_list))
    gridworld.showLearning(grid_policy, path = path + '/log')
