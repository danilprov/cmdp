import numpy as np

from codebase.algs.base import BaseAlgorithm


class FHA(BaseAlgorithm):
    """
    Finite Horizon Approximation for CMDP - Algorithm 3 from https://arxiv.org/pdf/2202.00150.pdf
    """
    def __init__(self, G=None, M=None, args=None, planner=None):
        super(FHA, self).__init__(G=G, M=M, args=args, planner=planner)

        # class specific parameters
        self.k = 1  # episode number
        self.t = 1  # round number
        self.last_state = None
        self.known_rewards = False
        self.visitation_sum += 1

        try:
            self.bonus_coef = args.bonus_coef
            self.T = args.rounds
        except:
            self.bonus_coef = 0.01
            self.T = args['T']

    def __call__(self):
        visitation_episode = np.zeros((self.num_states, self.num_actions))
        list_of_policies = self.get_policy()

        term = False
        rewards = np.array([]).reshape(0, self.M.d + 1)
        num_of_steps = self.M.H
        if num_of_steps > self.T - self.t:
            num_of_steps = self.T - self.t + 1
            term = True

            (p, r, c, v, last_state, sub_rewards) = self.planner.run(list_of_policies, num_of_steps,
                                                                     self.last_state)

            # Update Counts
            self.p_sum += p
            self.r_sum += r
            self.c_sum += c
            visitation_episode += v
            rewards = np.vstack((rewards, sub_rewards))

            self.last_state = last_state
            self.t += num_of_steps

            return rewards, term

        (p, r, c, v, last_state, sub_rewards) = self.planner.run(list_of_policies, num_of_steps, self.last_state)

        # Update Counts
        self.p_sum += p
        self.r_sum += r
        self.c_sum += c
        visitation_episode += v
        rewards = np.vstack((rewards, sub_rewards))

        self.last_state = last_state
        self.t += num_of_steps

        self.k += 1
        self.visitation_sum += visitation_episode
        # print(visitation_episode.sum())

        return rewards, term

    def get_policy(self):
        p_hat = np.zeros((self.num_states, self.num_actions, self.num_states))
        if self.known_rewards:
            r_hat = np.einsum('sap,sap->sa', self.M.P, self.M.R) if len(self.M.R.shape) == 3 else self.M.R
            c_hat = np.einsum('sap,dsap->dsa', self.M.P, self.M.C) if len(self.M.C.shape) == 4 else self.M.C
        else:
            r_hat = np.zeros((self.num_states, self.num_actions))
            c_hat = np.zeros((self.M.d, self.num_states, self.num_actions))
        bonus = np.zeros((self.num_states, self.num_actions))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                visitation = self.p_sum[s, a, :].sum()
                if visitation == 0:
                    p_hat[s, a, s] = 1
                    bonus[s, a] = 1 * self.bonus_coef
                else:
                    p_hat[s, a, :] = self.p_sum[s, a, :] / visitation
                    r_hat[s, a] = self.r_sum[s, a] / visitation
                    c_hat[:, s, a] = self.c_sum[:, s, a] / visitation
                    bonus[s, a] = np.sqrt(1.0 / visitation) * self.bonus_coef

        if self.last_state is not None:
            last_state_enc = self.M.Si(self.last_state)
        else:
            last_state_enc = None
        list_of_oms = self.planner(p_hat, r_hat, eps=bonus, last_state=last_state_enc)
        list_of_policies = np.zeros((self.M.H, self.M.S, self.M.A))
        for h in range(self.M.H):
            list_of_policies[h, :, :] = list_of_oms[h, :, :] / list_of_oms[h, :, :].sum(axis=1, keepdims=True)
            # replace nan with 0.25 (a bit weird way of doing this, but this is becuase 3d array)
            list_of_policies[h, np.isnan(list_of_policies[h, :, :])] = 0.25

        return list_of_policies

    def reset(self):
        super(FHA, self).reset()
        self.k = 1  # episode number
        self.t = 1  # round number
        self.last_state = None
        self.visitation_sum += 1
