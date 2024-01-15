import numpy as np

from codebase.algs.base import BaseAlgorithm


class PSRLOptimistic(BaseAlgorithm):
    def __init__(self, G=None, M=None, args=None, planner=None):
        super(PSRLOptimistic, self).__init__(G=G, M=M, args=args, planner=planner)

        # class specific parameters
        self.k = 1  # episode number
        self.t = 1  # round number
        self.last_state = None
        self.known_rewards = True
        self.visitation_sum += 1

        try:
            self.bonus_coef = args.bonus_coef
            self.T = args.rounds
        except:
            self.bonus_coef = 0.01
            self.T = args['T']

    def __call__(self):
        visitation_episode = np.zeros((self.num_states, self.num_actions))
        π_list = self.get_policy()

        term = False
        rewards = np.array([]).reshape(0, self.M.d + 1)
        while True:
            num_of_steps = self.get_num_steps(visitation_episode)
            if num_of_steps > self.T - self.t:
                num_of_steps = self.T - self.t + 1
                term = True

                (p, r, c, v, last_state, sub_rewards) = self.planner.run(π_list, num_of_steps, self.last_state)

                # Update Counts
                self.p_sum += p
                self.r_sum += r
                self.c_sum += c
                visitation_episode += v
                rewards = np.vstack((rewards, sub_rewards))

                self.last_state = last_state
                self.t += num_of_steps

                break
            if num_of_steps <= 0:
                break

            (p, r, c, v, last_state, sub_rewards) = self.planner.run(π_list, num_of_steps, self.last_state)

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
        r_hat = np.zeros((self.num_states, self.num_actions))
        c_hat = np.zeros((self.M.d, self.num_states, self.num_actions))
        bonus = np.zeros((self.num_states, self.num_actions))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                visitation = self.p_sum[s, a, :].sum()
                # average transitions
                if visitation != 0:
                    p_hat[s, a, :] = self.p_sum[s, a, :] / visitation
                    bonus[s, a] = np.sqrt(1.0 / visitation) * self.bonus_coef
                    # sample reward and costs
                    r_hat[s, a] = np.random.normal(self.r_sum[s, a] / visitation, bonus[s, a])
                    c_hat[:, s, a] = np.random.multivariate_normal(self.c_sum[:, s, a] / visitation,
                                                                   np.eye(self.M.d) * bonus[s, a])
                else:
                    # p_hat[s, a, s] = 1
                    p_hat[s, a, :] = np.random.dirichlet(np.ones(self.num_states))
                    bonus[s, a] = 1

        π_list = self.planner(p_hat, r_hat, c_hat)

        return π_list

    def get_num_steps(self, visitation_episode, d=3):
        max_dif = np.amin(d * self.visitation_sum - visitation_episode)
        if max_dif <= 0:
            return 0
        return int(max_dif)

    def reset(self):
        super(PSRLOptimistic, self).reset()
        self.k = 1  # episode number
        self.t = 1  # round number
        self.last_state = None
        self.visitation_sum += 1


class PSRLTransitions(PSRLOptimistic):
    def __init__(self, G=None, M=None, args=None, planner=None):
        super(PSRLTransitions, self).__init__(G=G, M=M, args=args, planner=planner)

    def get_policy(self):
        p_hat = np.zeros((self.num_states, self.num_actions, self.num_states))
        if self.known_rewards:
            r_hat = np.einsum('sap,sap->sa', self.M.P, self.M.R) if len(self.M.R.shape) == 3 else self.M.R
            c_hat = np.einsum('sap,dsap->dsa', self.M.P, self.M.C) if len(self.M.C.shape) == 4 else self.M.C
        else:
            r_hat = np.zeros((self.num_states, self.num_actions))
            c_hat = np.zeros((self.M.d, self.num_states, self.num_actions))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                visitation = self.p_sum[s, a, :].sum()
                # sample transitions
                p_hat[s, a, :] = np.random.dirichlet(np.maximum(self.p_sum[s, a, :], 1))
                # average reward and costs
                if visitation > 0 and self.known_rewards is False:
                    r_hat[s, a] = self.r_sum[s, a] / visitation
                    c_hat[:, s, a] = self.c_sum[:, s, a] / visitation

        π_list = self.planner(p_hat, r_hat, c_hat)

        grid_index = map(self.M.Si.lookup, range(len(self.M.Si)))
        grid_policy = dict(zip(grid_index, π_list))
        for r in range(1, 5):
            action_list = []
            for c in range(1, 5):
                s = (r, c)
                if len(set(grid_policy[s])) == 1:
                    action_list.append('*')
                    continue
                bestA = np.argmax(grid_policy[s])
                action_list.append(self.M.Ai.lookup(bestA))
            print([i for i in action_list])

        return π_list


class PSRLLagrangian(PSRLOptimistic):
    def __init__(self, G=None, M=None, args=None, planner=None):
        super(PSRLLagrangian, self).__init__(G=G, M=M, args=args, planner=planner)
        self._lambda = np.zeros(self.M.d)
        self._lambda_lr = args._lambda_lr

    def __call__(self):
        rewards, term = super().__call__()

        # update lambda using OGD
        # 1/t sum_j c_t
        average_cost = rewards[:, 1:].sum(axis=0) / rewards.shape[0]
        self._lambda = np.minimum(0, self._lambda - self._lambda_lr * (average_cost - self.budget))

        return rewards, term

    def get_policy(self):
        p_hat = np.zeros((self.num_states, self.num_actions, self.num_states))
        r_hat = np.zeros((self.num_states, self.num_actions))
        c_hat = np.zeros((self.M.d, self.num_states, self.num_actions))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                visitation = self.p_sum[s, a, :].sum()
                # sample transitions
                p_hat[s, a, :] = np.random.dirichlet(np.maximum(self.p_sum[s, a, :], 0.01))
                # average reward and costs
                if visitation > 0:
                    r_hat[s, a] = self.r_sum[s, a] / visitation
                    c_hat[:, s, a] = self.c_sum[:, s, a] / visitation

        # define pseude-reward: r_lambda = r_hat + lambda * |c_hat - budget|
        pseudo_reward = r_hat + np.einsum("i,isa->sa", self._lambda, c_hat)

        # solve classical model-based RL problem
        # π_list = self.planner(p_hat, r_lambda)
        # where self.planner is ValueIteration or A2C   (no lin prog here!)
        π_list = self.planner(p_hat, pseudo_reward)['pi_list']

        return π_list

    # def lagrangian_planner(self, p_hat, r_hat, c_hat):
    #     _lambda_lr = self._lambda_lr
    #     _lambda_avg = np.zeros(self.M.d)
    #
    #     for _ in range(self.conplanner_iter):
    #         pseudo_reward = r_hat + np.einsum("i,isa->sa", _lambda_lr, c_hat)
    #         π_list = self.planner(P=p_hat, R=pseudo_reward, use_constraints=False)
    #         # c = self.planner.value_evaluation(π_list, P=p_hat, R=pseudo_reward, C=c_hat)['constraint']
    #         c = self.planner.monte_carlo_evaluation(π_list, P=p_hat, R=pseudo_reward, C=c_hat)['constraint']
    #
    #         self._lambda = np.minimum(0, self._lambda - _lambda_lr * (c - self.budget))
    #         _lambda_avg += self._lambda
    #
    #     _lambda_avg /= self.conplanner_iter
    #     pseudo_reward = r_hat + np.einsum("i,isa->sa", _lambda_avg, c_hat)
    #     π_list = self.planner(P=p_hat, R=pseudo_reward, use_constraints=False)
    #
    #     return π_list

    def reset(self):
        super(PSRLLagrangian, self).reset()
        self._lambda = np.zeros(self.M.d)


class CUCRLOptimistic(PSRLOptimistic):
    def __init__(self, G=None, M=None, args=None, planner=None):
        super(CUCRLOptimistic, self).__init__(G=G, M=M, args=args, planner=planner)

    def get_policy(self):
        p_hat = np.zeros((self.num_states, self.num_actions, self.num_states))
        r_hat = np.zeros((self.num_states, self.num_actions))
        c_hat = np.zeros((self.M.d, self.num_states, self.num_actions))
        bonus = np.zeros((self.num_states, self.num_actions))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                visitation = self.p_sum[s, a, :].sum()
                if visitation == 0:
                    # p_hat[s, a, s] = 1
                    p_hat[s, a, :] = np.random.dirichlet(np.ones(self.num_states))
                    bonus[s, a] = 1
                else:
                    p_hat[s, a, :] = self.p_sum[s, a, :] / visitation
                    r_hat[s, a] = self.r_sum[s, a] / visitation
                    c_hat[:, s, a] = self.c_sum[:, s, a] / visitation
                    bonus[s, a] = np.sqrt(1.0 / visitation) * self.bonus_coef

        π_list = self.planner(p_hat, r_hat + bonus, c_hat - bonus)

        return π_list
