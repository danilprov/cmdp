import numpy as np
from itertools import product
import random

# from arsenal import Alphabet


class SingleServerQueueEnv():
    """A discrete-time single-server queue with a buffer of finite size L.

    This environment corresponds to the version of the control problem
    described in Chapter 5 in
    Constrained Markov Decision Processes by Eitan Altman (2004).
    https://www-sop.inria.fr/members/Eitan.Altman/TEMP/h.pdf
    """

    def __init__(self, L, num_actions):

        self.buffer_size = L
        self.num_actions = num_actions
        self.a_min, self.a_max = 0, 1
        self.b_min, self.b_max = 0, 1
        action_set_A = np.linspace(self.a_min, self.a_max, self.num_actions)
        action_set_B = np.linspace(self.b_min, self.b_max, self.num_actions)

        self.states = np.arange(self.buffer_size + 1)
        self.A = product(action_set_A, action_set_B)

        # Set seed
        #self._seed()

        # Set states and actions as Alphabet
        #self._set_states()
        #self._set_actions()

    # def _set_states(self):
    #     Si = Alphabet()
    #     for s in self.states:
    #         si = Si[s]
    #     Si.freeze()
    #
    # def _set_actions(self):
    #     Ai = Alphabet()
    #     for a in self.A:
    #         ai = Ai[a]
    #     Ai.freeze()

    def encode(self):
        # Si = Alphabet()
        # Ai = Alphabet()
        P = np.zeros((len(self.states), len(self.A), len(self.states)))
        R = np.zeros((len(self.states), len(self.A), len(self.states)))
        C = np.zeros((len(self.states), len(self.A), len(self.states)))
        if self.d > 0:
            C = np.zeros((self.d, len(self.states), len(self.A), len(self.states)))

        for s in self.states:
            si = s # si = Si[s]
            for a in self.actions:
                ai = a # ai = Ai[a]
                p, r, c = self.simulate(s, a)
                P[si, ai, :] += p
                R[si, ai, :] += r

                if self.d > 0:
                    C[:, si, ai, :] = c
                else:
                    C[si, ai, :] = c

        s0 = random.randint(0, self.buffer_size)

        # Si.freeze(); Ai.freeze()
        return (s0, P, R, C)# , Si, Ai

    # def _seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

if __name__ == '__main__':
    env = SingleServerQueueEnv(L=10, num_actions=3)
    print(env)

    P = np.zeros(11, 6, 11)

    r = - env.states / (env.buffer_size + 1)

    for s in env.states[1:]:
        for (a, b) in env.A:
            p = np.zeros(11)
            p[s - 1] += (1 - b) * a
            p[s] += b * a + (1 - b) * (1 - a)
            p[s+1] += b * (1 - a)




