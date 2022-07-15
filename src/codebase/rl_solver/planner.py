import numpy as np

from codebase.rl_solver.rl_solver import RLSolver


def argmax(q_values):
    """
    Takes in a list of q_values and returns the index of the item
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top_value = float("-inf")
    ties = []
    for i in range(len(q_values)):
        if q_values[i] > top_value:
            ties = [i]
            top_value = q_values[i]
        elif q_values[i] == top_value:
            ties.append(i)
    return np.random.choice(ties)


class ValueIteration(RLSolver):
    # Model-based Value Iteration
    # https://artint.info/html/ArtInt_227.html#:~:text=Value%20iteration%20is%20a%20method,uses%20an%20arbitrary%20end%20point.
    def __init__(self, M=None, args=None):
        super(ValueIteration, self).__init__(M=M, args=args)

    def __call__(self, P=None, R=None, theta=None, fic=None):
        """Compute optimal `H`-step Q-functions and policy"""
        S = self.S
        A = self.A
        H = self.H
        Q = np.zeros((H + 1, S, A))
        V = np.zeros((H + 1, S))

        R = self.R if R is None else R
        P = self.P if P is None else P
        R = np.einsum('sap,sap->sa', P, R) if len(R.shape) == 3 else R
        π_list = np.zeros((H + 1, S, A))
        for h in reversed(range(H)):
            Q[h, :, :] = R + P.dot(V[h + 1, :])
            # in case of equal action values across each state Q[h].argmax(axis=1) would always choose 0 action;
            # to break ties arbitrary I use argmax function from utilities and apply it along axis 1
            # i.e., apply argmax to each row of Q[s,a] matrix -- np.apply_along_axis(argmax, 1, Q[h]).
            π_list[h][range(S), np.apply_along_axis(argmax, 1, Q[h])] = 1
            # π_list[h][range(S), Q[h].argmax(axis=1)] = 1
            V[h] = Q[h].max(axis=1)

        return {
            'V': V,
            'Q': Q,
            'pi_list': π_list,
            'last_state': (0, 0)
        }


class RelativeValueIteration(ValueIteration):
    def __init__(self, M=None, args=None):
        super(RelativeValueIteration, self).__init__(M=M, args=args)
        self.epsilon = 0.001
        try:
            self.max_iter = args.max_iter
        except:
            self.max_iter = 50000

    @staticmethod
    def getSpan(W):
        """
        Return the span of W
        sp(W) = max W(s) - min W(s)
        """
        return W.max() - W.min()

    def __call__(self, P=None, R=None):
        """Compute optimal `H`-step Q-functions and policy"""
        S = self.S
        A = self.A
        Q = np.zeros((S, A))
        V = np.zeros((S))
        Vnext = np.zeros((S))
        gain = 0
        gainnext = 0
        gamma = 0.95

        R = self.R if R is None else R
        P = self.P if P is None else P
        R = np.einsum('sap,sap->sa', P, R) if len(R.shape) == 3 else R
        π_list = np.zeros((S, A))

        iter = 0
        while True:
            iter += 1
            Q = R + P.dot(V)
            Vnext = Q.max(axis=1)
            Vnext = Vnext - gain

            variation = self.getSpan(Vnext - V)

            if variation < self.epsilon:
                break
            if iter == self.max_iter:
                break

            V = Vnext
            gainnext = gain + gamma * V[0]
            gain = gainnext

        π_list[range(S), np.apply_along_axis(argmax, 1, Q)] = 1

        return {
            'V': V,
            'Q': Q,
            'pi_list': π_list,
            'last_state': (0, 0)
        }


if __name__ == "__main__":
    from codebase.mdp import FiniteHorizonCMDP
    from codebase.environments.gridworld import GridWorld


    def plot_policy(pi_list, V, Si, Ai, gridworld, print_outcome=True):
        ncols = gridworld.cols - 2
        nrows = gridworld.rows - 2
        picture = [[0] * nrows for _ in range(ncols)]
        value_function = [[0] * nrows for _ in range(ncols)]

        states = list(map(Si.lookup, np.where(pi_list == 1)[0]))
        actions = list(map(Ai.lookup, np.where(pi_list == 1)[1]))
        for (i, elem) in enumerate(states):
            picture[elem[0] - 1][elem[1] - 1] = actions[i]
            value_function[elem[0] - 1][elem[1] - 1] = V[Si[(elem[0], elem[1])]]

        if print_outcome:
            print('######################################################')
            for i in range(len(picture)):
                print(picture[i])

        return value_function

    """
    simple gridworld 4x4 from sutton barto
    """
    d = 1
    args = {'map': "4x4", 'horizon': 10, 'randomness': 0, 'd': d, 'infinite': False}
    gridworld = GridWorld(args)
    [mdp_values, Si, Ai] = gridworld.encode()

    s0, P, R, C = mdp_values
    for i in range(P.shape[0]):
        print(list(Si)[i], list(map(Ai.lookup, np.where(R[i] > 0)[0])))
        print(list(map(Si.lookup, np.where(P[i] > 0)[1])))
        print(list(map(Ai.lookup, np.where(P[i] > 0)[0])))

    M = FiniteHorizonCMDP(*mdp_values, d, None, gridworld.H, Si, gridworld.terminals)
    val_iter = ValueIteration(M)
    res = val_iter()

    actual = plot_policy(res['pi_list'][0], res['V'][0], Si, Ai, gridworld)
    expected = [[0, -1.0, -2.0, -3.0], [-1.0, -2.0, -3.0, -2.0], [-2.0, -3.0, -2.0, -1.0], [-3.0, -2.0, -1.0, 0]]

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])
    print(all([a == b for a, b in zip(actual, expected)]))
    values = val_iter.value_evaluation(res['pi_list'])


    args = {'map': "4x4_v1", 'horizon': 10, 'randomness': 0, 'd': d, 'infinite': False}
    gridworld = GridWorld(args)
    [mdp_values, Si, Ai] = gridworld.encode()

    s0, P, R, C = mdp_values

    M = FiniteHorizonCMDP(*mdp_values, d, None, gridworld.H, Si, gridworld.terminals)
    val_iter = ValueIteration(M)
    res = val_iter()

    actual = plot_policy(res['pi_list'][0], res['V'][0], Si, Ai, gridworld)
    print(actual)

    values = val_iter.value_evaluation(res['pi_list'])

    print(values['reward'], values['constraint'])

    """
    16x16 4-room gridworld
    """
    # args = {'map': "16x16.txt", 'horizon': 100, 'randomness': 0}
    # gridworld = GridWorld(args)
    # [mdp_values, Si, Ai] = gridworld.encode()
    #
    # s0, P, R, C = mdp_values
    #
    # M = FiniteHorizonCMDP(*mdp_values, 0, None, gridworld.H, Si, gridworld.terminals)
    # val_iter = ValueIteration(M)
    # res = val_iter()
    #
    # actual = plot_policy(res['pi_list'][0], res['V'][0], Si, Ai, gridworld)
    # print(actual)

    """
    4x4 from Zheung
    """
    d = 1
    args = {'map': "4x4_gwsc", 'randomness': 0.1, 'd': d, 'infinite': True, 'horizon': 10000}
    gridworld = GridWorld(args)
    [mdp_values, Si, Ai] = gridworld.encode()

    s0, P, R, C = mdp_values
    budget = [0.2]
    M = FiniteHorizonCMDP(*mdp_values, d, budget, gridworld.H, Si, gridworld.terminals)

    _lambda = [-.1]
    pseudo_reward = R + np.einsum("i,isap->sap", _lambda, C)
    val_iter = RelativeValueIteration(M)
    res = val_iter(R=pseudo_reward)
    plot_policy(res['pi_list'], res['V'], Si, Ai, gridworld)
