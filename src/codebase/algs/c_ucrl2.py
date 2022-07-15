import numpy as np

from codebase.algs.base import BaseAlgorithm


class CUCRLConservative(BaseAlgorithm):
    # C-UCRL algorithm from https://arxiv.org/pdf/2001.09377.pdf
    # supports ValueIteration, RelativeValueIteration (approximate) and LinProgSolver (exact) planners
    def __init__(self, G=None, M=None, args=None, planner=None):
        super(CUCRLConservative, self).__init__(G=G, M=M, args=args, planner=planner)

        # class specific parameters
        self.k = 1 # episode number
        self.t = 1 # round number
        self.last_state = None

        try:
            self.bonus_coef = args.bonus_coef
            self.T = args.rounds
        except:
            self.bonus_coef = 0.01
            self.T = args['T']

    def __call__(self):
        # exploration phase
        explore_rewards = self.run_exploration_phase()

        # exploitation phase
        exploit_rewards, term = self.run_exploitation_phase()

        return np.vstack((exploit_rewards, explore_rewards)), term

    def run_exploration_phase(self):
        # skip exploration if less than H rounds left
        if self.T - self.t <= self.H:
            return np.array([]).reshape(0, self.M.d + 1)

        # get random policy
        π_list = np.ones((self.num_states, self.num_actions)) * 1 / self.num_actions
        # run policy
        (p, r, c, v, last_state, rewards) = self.planner.run(π_list, self.H, self.last_state)
        # Update Counts
        self.update_counts(p, r, c, v, last_state, self.H, k=0)

        return rewards

    def run_exploitation_phase(self):
        # get policy
        π_list = self.get_policy()
        # get number of steps for exploration
        num_of_steps = (self.k - 1) * self.H
        term = False
        if num_of_steps > self.T - self.t:
            num_of_steps = self.T - self.t + 1
            term = True
        # run policy for num_of_steps number of steps
        (p, r, c, v, last_state, rewards) = self.planner.run(π_list, num_of_steps, self.last_state)
        # Update Counts
        self.update_counts(p, r, c, v, last_state, num_of_steps)

        return rewards, term

    def get_policy(self):
        p_hat, r_hat, c_hat, bonus = self.get_estimates()
        π_list = self.planner(p_hat, r_hat + bonus, c_hat + bonus)

        return π_list

    def get_estimates(self):
        p_hat = np.zeros((self.num_states, self.num_actions, self.num_states))
        r_hat = np.zeros((self.num_states, self.num_actions))
        c_hat = np.zeros((self.M.d, self.num_states, self.num_actions))
        bonus = np.zeros((self.num_states, self.num_actions))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                visitation = self.p_sum[s, a, :].sum()
                if visitation == 0:
                    p_hat[s, a, s] = 1
                    bonus[s, a] = 1
                else:
                    p_hat[s, a, :] = self.p_sum[s, a, :] / visitation
                    r_hat[s, a] = self.r_sum[s, a] / visitation
                    c_hat[:, s, a] = self.c_sum[:, s, a] / visitation
                    bonus[s, a] = np.sqrt(1.0 / visitation) * self.bonus_coef

        return p_hat, r_hat, c_hat, bonus

    def update_counts(self, p, r, c, v, last_state, num_of_steps, k=1):
        self.p_sum += p
        self.r_sum += r
        self.c_sum += c
        self.visitation_sum += v
        self.last_state = last_state
        self.t += num_of_steps
        self.k += k

    def reset(self):
        super(CUCRLConservative, self).reset()
        # class specific parameters
        self.k = 1  # episode number
        self.t = 1  # round number
        self.last_state = None


class CUCRLTransitions(CUCRLConservative):
    """
    UCRL-CMDP algorithm from https://arxiv.org/pdf/2002.12435.pdf
    supports only NonlinProgSolver (exact) planner

    Although the original algorithm is to be called in epochs of length T^a (without any exploration phases),
    I borrowed the epoch construction procedure of the C-UCRL algorithm for efficiency.

    Note that the main distinction is hidden in 'self.planner' which is, unlike C-UCRL, a nonlinear program.
    """

    def __init__(self, G=None, M=None, args=None, planner=None):
        super(CUCRLTransitions, self).__init__(G=G, M=M, args=args, planner=planner)

    def get_policy(self):
        p_hat, r_hat, c_hat, bonus = self.get_estimates()
        bonus3D = np.repeat(bonus[..., None], self.num_states, axis=2)
        π_list = self.planner(p_hat, r_hat, c_hat, bonus3D)

        return π_list


class CUCRLOptimistic2(CUCRLConservative):
    """
    The main idea is borrowed from ConRL algorithm from https://arxiv.org/pdf/2001.09377.pdf,
    however I also use the C-UCRL epoch construction procedure as ConRL is originally developed for episodic tasks.

    supports ValueIteration, RelativeValueIteration (approximate) and LinProgSolver (exact) planners
    """

    def __init__(self, G=None, M=None, args=None, planner=None):
        super(CUCRLOptimistic2, self).__init__(G=G, M=M, args=args, planner=planner)

    def get_policy(self):
        p_hat, r_hat, c_hat, bonus = self.get_estimates()
        π_list = self.planner(p_hat, r_hat + bonus, c_hat - bonus)

        return π_list
