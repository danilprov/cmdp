import numpy as np
from collections import deque


class RCPOPolicy:
    def __init__(self, shadow=False):
        self.shadow = shadow
        self.queue = deque(maxlen=10)

    def add_response(self, best_exp_rtn=None):
        self.queue.append(best_exp_rtn)
        rtn = np.average(self.queue, axis=0)

        self.reward = rtn[0]
        self.constraint = rtn[1]

    def add__multiple_responses(self, best_exp_rtn=None):
        self.queue.extend(best_exp_rtn)
        rtn = np.average(self.queue, axis=0)

        self.reward = rtn[0]
        self.constraint = rtn[1]


class BaseAlgorithm:
    def __init__(self, G=None, M=None, args=None, planner=None):
        self.num_states = len(G.states)
        self.num_actions = len(G.A)
        self.H = G.H
        self.budget = M.budget
        self.M = M
        self.policy = RCPOPolicy()
        self.planner = planner

        self.p_sum = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.r_sum = np.zeros((self.num_states, self.num_actions))
        self.c_sum = np.zeros((self.M.d, self.num_states, self.num_actions))
        self.visitation_sum = np.zeros((self.num_states, self.num_actions))

        self.R = np.einsum('sap,sap->sa', self.M.P, self.M.R) if len(np.shape(self.M.R)) == 3 else self.M.R
        self.C = np.einsum('sap,dsap->dsa', M.P, M.C) if len(M.C.shape) == 4 else M.C
        # self.C = np.einsum('sap,sap->sa', self.M.P, self.M.C) if len(np.shape(self.M.C)) == 3 else self.M.C

    def __call__(self, *args, **kwargs):
        pass

    def reset(self):
        # self.__init__(G, M, args, planner)

        self.p_sum = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.r_sum = np.zeros((self.num_states, self.num_actions))
        self.c_sum = np.zeros((self.M.d, self.num_states, self.num_actions))
        self.visitation_sum = np.zeros((self.num_states, self.num_actions))