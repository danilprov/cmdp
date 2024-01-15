# -*- coding: utf-8 -*-
import numpy as np
from arsenal.maths import sample, random_dist
import warnings
from arsenal import Alphabet


def random_MDP(S, A, gamma=0.95, d=2, b=None, r=None):
    """Randomly generated MDP
    Text taken from http://www.jmlr.org/papers/volume15/geist14a/geist14a.pdf
      "... we consider Garnet problems (Archibald et al., 1995), which are a
      class of randomly constructed finite MDPs. They do not correspond to any
      specific application, but are totally abstract while remaining
      representative of the kind of MDP that might be encountered in
      practice. In our experiments, a Garnet is parameterized by 3 parameters
      and is written G(S, A, b): S is the number of states, A is the number of
      actions, b is a branching factor specifying how many possible next states
      are possible for each state-action pair (b states are chosen uniformly at
      random and transition probabilities are set by sampling uniform random b −
      1 cut points between 0 and 1). The reward is state-dependent: for a given
      randomly generated Garnet problem, the reward for each state is uniformly
      sampled between 0 and 1."
      "The discount factor γ is set to 0.95 in all experiments."
    We consider two types of problems, “small” and “big”, respectively
    corresponding to instances G(30, 2, p=2, dim=8) and G(100, 4, p=3, dim=20)

    ###
    d - number of constraints
    """

    if b is None: b = S
    if r is None: r = S

    P = np.zeros((S, A, S))
    states = np.array(list(range(S)))

    # rs = np.random.choice(states, size=r, replace=False)

    for s in range(S):
        for a in range(A):
            # pick b states to be connected to.
            connected = np.random.choice(states, size=b, replace=False)
            P[s, a, connected] = random_dist(b)

    #R = np.zeros((S, A, S))
    #rstates = np.random.choice(states, size=r, replace=False)
    #R[rstates, :, :] = np.random.uniform(0, 1, r)

    ###
    # my adjustment for CMDPs
    ###
    R = np.random.rand(S, A)
    C = np.random.rand(d, S, A)
    C = np.zeros((d, S, A))
    budget = np.ones(d) * 10

    M = DiscountedMDP(
        s0=random_dist(S),
        R=R,
        P=P,
        gamma=gamma,
        C=C,
        d=d,
        budget=budget,
        terminals=None,
        H=100,
        Si=Alphabet(),
        Ai=Alphabet()
    )

    return M


class CMDP(object):
    def __init__(self, s0, P, R, C, d, budget, Si, Ai, terminals):
        # P: Probability distribution p(S' | A S) stored as an array S x A x S'
        # R: Reward function r(S, A, S) -> Reals stored as an array S x A x S'
        # s0: Distribution over the initial state.
        self.s0 = s0
        [self.S, self.A, _] = P.shape
        # has a shape of (S,A,S)
        self.P = P
        # reward, has a shape of (S,A)
        self.R = R
        # constraints, has a shape of (d,S,A)
        self.C = C
        # number of constraints
        self.d = d
        # budget of constraints in shape (d)
        self.budget = budget
        self.Si = Si
        self.Ai = Ai

        self.terminals = terminals


class FiniteHorizonCMDP(CMDP):
    """Finite-horizon MDP."""

    def __init__(self, s0, P, R, C, d, budget, H, Si, Ai, terminals):
        super(FiniteHorizonCMDP, self).__init__(s0, P, R, C, d, budget, Si, Ai, terminals)
        self.H = H


class MarkovChain():
    "γ-discounted Markov chain."

    def __init__(self, s0, P, gamma):
        self.s0 = s0
        self.S = None
        self.P = P
        self.gamma = gamma

    def successor_representation(self):
        "Dayan's successor representation."
        return np.linalg.solve(np.eye(self.S) - self.gamma * self.P,
                               np.eye(self.S))

    def stationary(self):
        "Stationary distribution."
        # The stationary distribution, much like other key quantities in MRPs,
        # is the solution to a linear recurrence,
        #            d = (1-γ) s0 + γ Pᵀ d    # transpose because P is s->s' and we want s'->s.
        #   d - γ Pᵀ d = (1-γ) s0
        # (I - γ Pᵀ) d = (1-γ) s0
        # See also: stationarity condition in the linear programming solution

        return np.linalg.solve(np.eye(self.S) - self.gamma * self.P.T, (1 - self.gamma) * self.s0)  # note the transpose

    def eigenvalue_stationary(self):
        "Stationary distribution Eigen Values"
        pi = np.random.rand(13, 1)
        for _ in range(100000): pi = pi.T.dot(self.P)
        return pi


class MRP(MarkovChain):
    "Markov reward process."

    def __init__(self, s0, P, R, gamma):
        super(MRP, self).__init__(s0, P, gamma)
        self.R = R
        self.gamma = gamma
        [self.S, _] = P.shape
        assert R.ndim == 1 and R.shape[0] == P.shape[0] == P.shape[1]

    def V(self):
        "Value function"
        return np.linalg.solve(np.eye(self.S) - self.gamma * self.P, self.R)


class DiscountedMDP(CMDP):
    "γ-discounted, infinite-horizon Markov decision process."

    # def __init__(self, s0, P, R, C, Si, gamma=None):
    def __init__(self, s0, P, R, C, d, budget, H, Si, Ai, terminals, gamma=None):
        # γ: Temporal discount factor
        super(DiscountedMDP, self).__init__(s0, P, R, C, d, budget, Si, Ai, terminals)
        self.gamma = gamma
        self.Si = Si
        self.Ai = Ai
        self.H = H

    def run(self, learner):
        s = sample(self.s0)
        while True:
            a = learner(s)
            if np.random.uniform() <= (1 - self.gamma):
                sp = sample(self.s0)
                r = 0
            else:
                sp = sample(self.P[s, a, :])
                r = self.R[s, a, sp]
            if not learner.update(s, a, r, sp):
                break
            s = sp

    def mrp(self, policy, R=None):
        "MDP becomes an `MRP` when we condition on `policy`."
        R = R if R is not None else self.R
        return MRP(self.s0,
                   np.einsum('sa,sap->sp', policy, self.P),
                   np.einsum('sa,sap,sap->s', policy, self.P, R),
                   self.gamma)

    # def J(self, policy):
    #    "Expected value of `policy`."
    #    return self.mrp(policy).J()

    def V(self, policy):
        "Value function for `policy`."
        return self.mrp(policy).V()

    def successor_representation(self, policy):
        "Dayan's successor representation."
        return self.mrp(policy).successor_representation()

    def dvisit(self, policy, R=None):
        "γ-discounted stationary distribution over states conditioned `policy`."
        return self.mrp(policy, R).stationary()

    def Q(self, policy, R=None):
        "Compute the action-value function `Q(s,a)` for a policy."
        R = R if R is not None else self.R

        v = self.V(policy)
        r = np.einsum('sap,sap->sa', self.P, R)
        Q = np.zeros((self.S, self.A))
        P = self.P

        for s in range(self.S):
            for a in range(self.A):
                Q[s, a] = r[s, a] + self.gamma * self.P[s, a, :].dot(v)
        return Q

    def Advantage(self, policy):
        "Advantage function for policy."
        return self.Q(policy) - self.V(policy)[None].T  # handles broadcast

    def B(self, V):
        "Bellman operator."
        # Act greedily according to one-step lookahead on V.
        Q = self.Q_from_V(V)
        pi = np.zeros((self.S, self.A))
        pi[range(self.S), Q.argmax(axis=1)] = 1
        v = Q.max(axis=1)
        return v, pi

    def Q_from_V(self, V):
        "Lookahead by a single action from value function estimate `V`."
        return (self.P * (self.R + self.gamma * V[None, None, :])).sum(axis=2)


if __name__ == "__main__":
    M = random_MDP(20, 5, 0.7)
    print('a')
