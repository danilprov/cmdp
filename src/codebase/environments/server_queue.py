import numpy as np
from itertools import product
import random

from arsenal import Alphabet


class SingleServerQueueEnv():
    """A discrete-time single-server queue with a buffer of finite size L.

    This environment corresponds to the version of the control problem
    described in Chapter 5 in
    Constrained Markov Decision Processes by Eitan Altman (2004).
    https://www-sop.inria.fr/members/Eitan.Altman/TEMP/h.pdf
    """

    def __init__(self, L, num_actions, a_min=0, a_max=1, b_min=0, b_max=1, d=2, test=False):
        self.states = np.arange(L + 1)
        self.S = len(self.states)

        self.a_min, self.a_max = a_min, a_max
        self.b_min, self.b_max = b_min, b_max
        self.action_set_A = np.linspace(self.a_min, self.a_max, num_actions)
        self.action_set_B = np.linspace(self.b_min, self.b_max, num_actions)
        if 0 not in self.action_set_B:
            self.action_set_B = [0] + self.action_set_B
        self.actions = list(product(self.action_set_A, self.action_set_B))
        self.A = len(self.actions)

        self.d = d

        self.test = test
        # Set seed
        self._seed()

    def encode(self):
        Si = Alphabet()
        Ai = Alphabet()
        P = np.zeros((self.S, self.A, self.S))
        R = np.zeros((self.S, self.A, self.S))
        C = np.zeros((self.d, self.S, self.A, self.S))

        for s in self.states:
            si = Si[s]
            for (a,b) in self.actions:
                ai = Ai[(a,b)]
                #print((a,b), ai)
                p, r, c = self.simulate(s, (a,b))
                P[si, ai, :] += p
                R[si, ai, :] += r
                C[:, si, ai, :] += c

        #s0 = random.randint(0, self.S)
        s0 = self.np_random.randint(0, self.S)

        Si.freeze(); Ai.freeze()
        return (s0, P, R, C), Si, Ai

    def simulate(self, s, a):
        a1, a2 = a
        p = np.zeros(self.S)
        c = np.zeros((self.d, self.S))
        if s == 0:
            p[s] += 1 - a2 * (1 - a1)
            p[s + 1] += a2 * (1 - a1)
        elif s == self.S-1:
            a2 = 0 # no arrivals are possible if buffer is full
            p[s - 1] += (1 - a2) * a1
            p[s] += a2 * a1 + (1 - a2) * (1 - a1)
        else:
            p[s - 1] += (1 - a2) * a1
            p[s] += a2 * a1 + (1 - a2) * (1 - a1)
            p[s + 1] += a2 * (1 - a1)

        r = - self.states / (self.S - 1) * 100
        c[0,:] = a1 # quality of service
        c[1,:] = 1 / (1 + a2) # throughoput of the system
        if self.test:
            c[1, :] = -a2

        return p, r, c

    def _seed(self, seed=None):
        """Set the random seed.
        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        self.np_random = np.random.RandomState()
        self.np_random.seed(seed)
        return seed


if __name__ == '__main__':
    from codebase.mdp import FiniteHorizonCMDP
    from codebase.rl_solver.lin_prog import LinProgSolver
    from codebase.rl_solver.nonlin_prog import NonlinProgSolver

    """
    marsrover example
    """

    d = 2
    args = {'map': "8x8_marsrover", 'randomness': 0.1, 'd': d, 'infinite': True, 'horizon': 20}
    server = SingleServerQueueEnv(L=1, num_actions=3, a_min=0.1, a_max=0.9, b_min=0.1, b_max=0.9)
    [mdp_values, Si, Ai] = server.encode()

    s0, P, R, C = mdp_values
    # budget = [2.5, 3.9]
    # budget = [1000, 1000]
    budget = [0.3, 0.8185]
    M = FiniteHorizonCMDP(*mdp_values, d, budget, 100, Si, terminals=None)
    lin_opt = LinProgSolver(M)
    pi_list = lin_opt()
    print('a')