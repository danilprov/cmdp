import numpy as np
from arsenal.maths import sample
from collections import defaultdict


class RLSolver:
    def __init__(self, M=None, args=None):
        self.M = M
        self.P = M.P
        self.S = M.S
        self.A = M.A
        self.H = M.H
        self.args = args
        self.s0 = M.s0  # initial state

        # if R and C[i] are defined as r(s,a,s') or c_i(s,a,s'),
        # we recast it as r(s,a) or c_i(s,a) by conditioning on P
        self.R = np.einsum('sap,sap->sa', M.P, M.R) if len(M.R.shape) == 3 else M.R
        self.C = np.einsum('sap,dsap->dsa', M.P, M.C) if len(M.C.shape) == 4 else M.C
        # self.C = np.einsum('sap,sap->sa', M.P, M.C) if len(M.C.shape) == 3 else M.C

        self.stats = defaultdict(int)

        self.d = M.d
        self.Si = M.Si
        self.budget = M.budget

        self.num_episodes = 1

    def __call__(self):
        pass

    def run(self, π_list, H=None, last_state=None):  # , P=None, R=None, C=None):
        """
        H - the length of the episode
            for episodic tasks it is fixed throughout learning and equal to self.H
            for infinite tasks it is defined when the method is called
        """
        (S, A, P, R, C) = (self.S, self.A, self.P, self.R, self.C)
        H = self.H if H is None else H

        p_traj = np.zeros((S, A, S))
        r_traj = np.zeros((S, A))
        c_traj = np.zeros((self.d, S, A))
        visitation = np.zeros((S, A))
        r_vector = np.zeros((H, self.d + 1))

        if len(np.shape(R)) == 3:
            R = np.einsum("sap,sap->sa", P, R)
        if len(np.shape(C)) == 4:
            C = np.einsum("sap,dsap->dsa", P, C)
            # C = np.einsum("sap,sap->sa", P, C)

        if last_state is None:
            s = sample(self.s0)
        else:
            s = self.Si[last_state]
        for h in range(H):
            # a = sample(π_list[h,s,:])
            try:
                a = π_list[h, s, :].argmax()
            except:
                a = sample(π_list[s, :])

            s_next = sample(P[s, a, :])
            print(self.Si.lookup(s), a, self.Si.lookup(s_next))
            p_traj[s, a, s_next] += 1.0
            r_traj[s, a] += R[s, a]
            c_traj[:, s, a] += C[:, s, a]
            visitation[s, a] += 1.0

            r_vector[h, 0] = R[s, a]
            r_vector[h, 1:] = C[:, s, a]
            s = s_next
        self.stats['training_consumpution'] += np.sum(c_traj, axis=(1, 2))  # c_traj.sum()
        self.stats['num_trajs'] += 1
        return p_traj, r_traj, c_traj, visitation, self.Si.lookup(s), r_vector

    def value_evaluation_dp(self, π_list, P=None, R=None, C=None, max_value_iter = 100000, delta = 0.001):
        update_consumption = False
        if P is None and R is None and C is None:
            update_consumption = True

        (S, A, H, d) = (self.S, self.A, self.H, self.d)

        C = self.C if C is None else C
        P = self.P if P is None else P
        R = self.R if R is None else R

        Q = np.zeros((S, A))
        V = np.zeros((S))
        V_C = np.zeros((d, S))
        Q_C = np.zeros((d, S, A))

        for j in range(max_value_iter):
            old_V = V.copy()
            old_V_C = V_C.copy()
            for s in range(S):
                for a in range(A):
                    Q[s, a] = R[s, a] + P[s, a, :].dot(old_V[:])
                    Q_C[:, s, a] = C[:, s, a] + old_V_C[:, :].dot(P[s, a, :])
                V[s] = π_list[s, :].dot(Q[s, :])
                V_C[:, s] = Q_C[:, s, :].dot(π_list[s, :])

            max_diff = max(np.abs(V - old_V))
            if max_diff < delta:
                break

        if update_consumption:
            self.stats['expected_consumption'] += V_C[:, :].dot(self.s0) * self.num_episodes

        return {
            'V': V,
            'V_C': V_C,
            'Q': Q,
            'Q_C': Q_C,
            'reward': self.s0.dot(V[:]),
            'constraint': V_C[:, :].dot(self.s0)
        }

    def value_evaluation(self, π_list, P=None, R=None, C=None):
        """
        Compute `H`-step value functions for `policy`.

        Warning: It is not exactly the H-step value function
                 at each iteration different policies are being used for calculations

        to allow multi dimensional constraints I changed the dimension of all arrays:
            V_C(h,s) -> V_C(d,h,s)
            Q_C(h,s,a) -> Q_C(d,h,s,a);
        where h - episode index, s,a - state-action pair, and d - dimension index.
        In case
            d=1: we have everything as it was;
            d>1: output regarding V_C, Q_C, and constraint becomes one dimension higher

        Small reminder:
            I had to rewrite all dot products from P[s, a, :].dot(V_C[h + 1, :]) to V_C[:, h + 1, :].dot(P[s, a, :])

        """

        update_consumption = False
        if P is None and R is None and C is None:
            update_consumption = True

        (S, A, H, d) = (self.S, self.A, self.H, self.d)

        C = self.C if C is None else C
        P = self.P if P is None else P
        R = self.R if R is None else R

        Q = np.zeros((H + 1, S, A))
        V = np.zeros((H + 1, S))
        V_C = np.zeros((d, H + 1, S))
        Q_C = np.zeros((d, H + 1, S, A))
        # V_C = np.zeros((H + 1, S))
        # Q_C = np.zeros((H + 1, S, A))
        for h in reversed(range(H)):
            for s in range(S):
                for a in range(A):
                    Q[h, s, a] = R[s, a] + P[s, a, :].dot(V[h + 1, :])
                    Q_C[:, h, s, a] = C[:, s, a] + V_C[:, h + 1, :].dot(P[s, a, :])
                    # Q_C[h, s, a] = C[s, a] + P[s, a, :].dot(V_C[h + 1, :])
                V[h, s] = π_list[h, s, :].dot(Q[h, s, :])
                V_C[:, h, s] = Q_C[:, h, s, :].dot(π_list[h, s, :])
                # V_C[h, s] = π_list[h, s, :].dot(Q_C[h, s, :])

        if update_consumption:
            self.stats['expected_consumption'] += V_C[:, 0, :].dot(self.s0) * self.num_episodes
            # self.stats['expected_consumption'] += self.s0.dot(V_C[0, :]) * self.num_episodes
        return {
            'V': V,
            'V_C': V_C,
            'Q': Q,
            'Q_C': Q_C,
            'reward': self.s0.dot(V[0, :]),
            'constraint': V_C[:, 0, :].dot(self.s0),
            # 'constraint': self.s0.dot(V_C[0, :]),
        }

    def monte_carlo_evaluation(self, π_list, P=None, R=None, C=None, time_horizon = 100, num_of_traj = 100):
        C = self.C if C is None else C
        P = self.P if P is None else P
        R = self.R if R is None else R

        rewards = np.zeros((num_of_traj, time_horizon))
        costs = np.zeros((self.d, num_of_traj, time_horizon))

        for j in range(num_of_traj):
            s = sample(self.s0)
            for t in range(time_horizon):
                a = sample(π_list[s, :])
                rewards[j, t] = R[s, a]
                costs[:, j, t] = C[:, s, a]
                s_next = sample(P[s, a, :])
                s = s_next

        return {
            'V': '-',
            'V_C': '-',
            'Q': '-',
            'Q_C': '-',
            'reward': rewards.mean(axis=(0, 1)),
            'constraint': costs.mean(axis=(1, 2)),
            # 'constraint': self.s0.dot(V_C[0, :]),
        }

    def exact_value_evaluation(self, π_list):
        """
        If the environment’s dynamics are completely known, then

        v = R_π + gamma * P_π v,    where _π means conditioning over policy π, i.e.,
                                    R_π(s) = sum_a π(a|s) * R(s,a) for all s.

        is a system of |S|  simultaneous linear equations in |S| unknowns:

        v (I - gamma * P_π) = R_π.

        In principle, its solution is a straightforward but only for episodic tasks (if gamma < 1).
        If gamma=1, matrix (I - gamma * P_π) is singular, i.e., has a determinant of zero.

        # TODO:
        test this method on an episodic task
        """
        gamma = 0.99
        R_pi = (self.R * π_list).sum(axis=1)
        P_pi = np.einsum("sa,sap->sp", π_list, self.P)

        v = np.linalg.solve(np.eye(self.S, dtype=int) - gamma * P_pi, R_pi)

        return v

    def reset(self):
        pass

    def reset_oracle(self):
        pass


if __name__ == '__main__':
    from codebase.mdp import FiniteHorizonCMDP
    from codebase.environments.gridworld import GridWorld

    """
    simple gridworld 4x4 from sutton barto
    """
    d = 1
    args = {'map': "4x4", 'horizon': 10, 'randomness': 0, 'd': d, 'infinite': False}
    gridworld = GridWorld(args)
    [mdp_values, Si, Ai] = gridworld.encode()

    s0, P, R, C = mdp_values

    M = FiniteHorizonCMDP(*mdp_values, d, None, gridworld.H, Si, gridworld.terminals)
    val_iter = RLSolver(M)
    policy = np.ones((len(Si), len(Ai))) * 0.25

    res = val_iter.value_evaluation_dp(policy)

    grid_V = np.zeros((4, 4))
    for elem in list(map(Si.lookup, range(16))):
        grid_V[elem[0] - 1, elem[1] - 1] = res['V'][Si(elem)]

    true_V = np.array([[0, -14, -20, -22],
                       [-14, -18, -20, -20],
                       [-20, -20, -18, -14],
                       [-22, -20, -14, 0]])
    assert np.allclose(grid_V, true_V, rtol=1e-02)
    print(grid_V)
