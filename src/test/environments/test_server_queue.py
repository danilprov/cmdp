import unittest

import numpy as np

from codebase.environments.server_queue import SingleServerQueueEnv


class TestSingleServerQueueEnv(unittest.TestCase):
    def test_init(self):
        env = SingleServerQueueEnv(4, 3)
        # check states
        self.assertCountEqual(env.states, np.arange(5))
        self.assertEqual(env.S, 5)
        # check actions
        self.assertEqual(env.a_min, 0)
        self.assertEqual(env.a_max, 1)
        self.assertEqual(env.b_min, 0)
        self.assertEqual(env.b_max, 1)
        self.assertCountEqual(env.action_set_A, np.linspace(0, 1, 3))
        self.assertCountEqual(env.action_set_B, np.linspace(0, 1, 3))
        self.assertIn(0, env.action_set_B)
        self.assertCountEqual(env.actions,
                              [(0, 0), (0, 0.5), (0, 1), (0.5, 0), (0.5, 0.5), (0.5, 1), (1, 0), (1, 0.5), (1, 1)])
        self.assertEqual(env.A, 9)
        # check budget dimension
        self.assertEqual(env.d, 2)

    def test_simulate(self):
        env = SingleServerQueueEnv(4, 3, test=True)

        p, r, c = env.simulate(0, (0, 0))
        np.testing.assert_array_equal(p, np.array([1, 0, 0, 0, 0]))
        np.testing.assert_array_equal(r, np.array([0, -0.25, -0.5, -0.75, -1.]))
        np.testing.assert_array_equal(c, np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))

        p, r, c = env.simulate(0, (0.5, 0.5))
        np.testing.assert_array_equal(p, np.array([0.75, 0.25, 0, 0, 0]))
        np.testing.assert_array_equal(r, np.array([0, -0.25, -0.5, -0.75, -1.]))
        np.testing.assert_array_equal(c, np.array([[0.5, 0.5, 0.5, 0.5, 0.5], [-0.5, -0.5, -0.5, -0.5, -0.5]]))

        p, r, c = env.simulate(0, (1, 1))
        np.testing.assert_array_equal(p, np.array([1, 0, 0, 0, 0]))
        np.testing.assert_array_equal(r, np.array([0, -0.25, -0.5, -0.75, -1.]))
        np.testing.assert_array_equal(c, np.array([[1, 1, 1, 1, 1], [-1, -1, -1, -1, -1]]))

        p, r, c = env.simulate(2, (0, 0))
        np.testing.assert_array_equal(p, np.array([0, 0, 1, 0, 0]))
        np.testing.assert_array_equal(r, np.array([0, -0.25, -0.5, -0.75, -1.]))
        np.testing.assert_array_equal(c, np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))

        p, r, c = env.simulate(2, (0.5, 0.5))
        np.testing.assert_array_equal(p, np.array([0, 0.25, 0.5, 0.25, 0]))
        np.testing.assert_array_equal(r, np.array([0, -0.25, -0.5, -0.75, -1.]))
        np.testing.assert_array_equal(c, np.array([[0.5, 0.5, 0.5, 0.5, 0.5], [-0.5, -0.5, -0.5, -0.5, -0.5]]))

        p, r, c = env.simulate(2, (1, 1))
        np.testing.assert_array_equal(p, np.array([0, 0, 1, 0, 0]))
        np.testing.assert_array_equal(r, np.array([0, -0.25, -0.5, -0.75, -1.]))
        np.testing.assert_array_equal(c, np.array([[1, 1, 1, 1, 1], [-1, -1, -1, -1, -1]]))

        p, r, c = env.simulate(4, (0, 0))
        np.testing.assert_array_equal(p, np.array([0, 0, 0, 0, 1]))
        np.testing.assert_array_equal(r, np.array([0, -0.25, -0.5, -0.75, -1.]))
        np.testing.assert_array_equal(c, np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))

        p, r, c = env.simulate(4, (0.5, 0.5))
        np.testing.assert_array_equal(p, np.array([0, 0, 0, 0.5, 0.5]))
        np.testing.assert_array_equal(r, np.array([0, -0.25, -0.5, -0.75, -1.]))
        np.testing.assert_array_equal(c, np.array([[0.5, 0.5, 0.5, 0.5, 0.5], [0, 0, 0, 0, 0]]))

        p, r, c = env.simulate(4, (1, 1))
        np.testing.assert_array_equal(p, np.array([0, 0, 0, 1, 0]))
        np.testing.assert_array_equal(r, np.array([0, -0.25, -0.5, -0.75, -1.]))
        np.testing.assert_array_equal(c, np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]))

    def test_encode(self):
        env = SingleServerQueueEnv(4, 3, test=True)
        env._seed(1)
        (s0, P, R, C), Si, Ai = env.encode()

        np.testing.assert_array_equal(P, np.array([[[1, 0, 0, 0, 0],  # 0, 0
                                                    [0.5, 0.5, 0, 0, 0],  # 0, 0.5
                                                    [0, 1, 0, 0, 0],  # 0, 1
                                                    [1, 0, 0, 0, 0],  # 0.5, 0
                                                    [0.75, 0.25, 0, 0, 0],  # 0.5, 0.5
                                                    [0.5, 0.5, 0, 0, 0],  # 0.5, 0.1
                                                    [1, 0, 0, 0, 0],  # 1, 0
                                                    [1, 0, 0, 0, 0],  # 1, 0.5
                                                    [1, 0, 0, 0, 0]],
                                                   [[0, 1, 0, 0, 0],  # 0, 0
                                                    [0, 0.5, 0.5, 0, 0],  # 0, 0.5
                                                    [0, 0, 1, 0, 0],  # 0, 1
                                                    [0.5, 0.5, 0, 0, 0],  # 0.5, 0
                                                    [0.25, 0.5, 0.25, 0, 0],  # 0.5, 0.5
                                                    [0, 0.5, 0.5, 0, 0],  # 0.5, 1
                                                    [1, 0, 0, 0, 0],  # 1, 0
                                                    [0.5, 0.5, 0, 0, 0],  # 1, 0.5
                                                    [0, 1, 0, 0, 0]],
                                                   [[0, 0, 1, 0, 0],  # 0, 0
                                                    [0, 0, 0.5, 0.5, 0],  # 0, 0.5
                                                    [0, 0, 0, 1, 0],  # 0, 1
                                                    [0, 0.5, 0.5, 0, 0],  # 0.5, 0
                                                    [0, 0.25, 0.5, 0.25, 0],  # 0.5, 0.5
                                                    [0, 0, 0.5, 0.5, 0],  # 0.5, 1
                                                    [0, 1, 0, 0, 0],  # 1, 0
                                                    [0, 0.5, 0.5, 0, 0],  # 1, 0.5
                                                    [0, 0, 1, 0, 0]],
                                                   [[0, 0, 0, 1, 0],  # 0, 0
                                                    [0, 0, 0, 0.5, 0.5],  # 0, 0.5
                                                    [0, 0, 0, 0, 1],  # 0, 1
                                                    [0, 0, 0.5, 0.5, 0],  # 0.5, 0
                                                    [0, 0, 0.25, 0.5, 0.25],  # 0.5, 0.5
                                                    [0, 0, 0, 0.5, 0.5],  # 0.5, 1
                                                    [0, 0, 1, 0, 0],  # 1, 0
                                                    [0, 0, 0.5, 0.5, 0],  # 1, 0.5
                                                    [0, 0, 0, 1, 0]],
                                                   [[0, 0, 0, 0, 1],  # 0, 0
                                                    [0, 0, 0, 0, 1],  # 0, 0.5
                                                    [0, 0, 0, 0, 1],  # 0, 1
                                                    [0, 0, 0, 0.5, 0.5],  # 0.5, 0
                                                    [0, 0, 0, 0.5, 0.5],  # 0.5, 0.5
                                                    [0, 0, 0, 0.5, 0.5],  # 0.5, 1
                                                    [0, 0, 0, 1, 0],  # 1, 0
                                                    [0, 0, 0, 1, 0],  # 1, 0.5
                                                    [0, 0, 0, 1, 0]]]))  # 1, 1

        np.testing.assert_array_equal(R, np.array([[[0, -0.25, -0.5, -0.75, -1.]] * 9] * 5))

        np.testing.assert_array_equal(C[0, :, :, :], np.array([[[0, 0, 0, 0, 0],  # 0, 0
                                                               [0, 0, 0, 0, 0],  # 0, 0.5
                                                               [0, 0, 0, 0, 0],  # 0, 1
                                                               [0.5, 0.5, 0.5, 0.5, 0.5],  # 0.5, 0
                                                               [0.5, 0.5, 0.5, 0.5, 0.5],  # 0.5, 0.5
                                                               [0.5, 0.5, 0.5, 0.5, 0.5],  # 0.5, 0.1
                                                               [1, 1, 1, 1, 1],  # 1, 0
                                                               [1, 1, 1, 1, 1],  # 1, 0.5
                                                               [1, 1, 1, 1, 1]]] * 5))

        np.testing.assert_array_equal(C[1, :, :, :], -1 * np.array([[[0, 0, 0, 0, 0],  # 0, 0
                                                                    [0.5, 0.5, 0.5, 0.5, 0.5],  # 0, 0.5
                                                                    [1, 1, 1, 1, 1],  # 0, 1
                                                                    [0, 0, 0, 0, 0],  # 0.5, 0
                                                                    [0.5, 0.5, 0.5, 0.5, 0.5],  # 0.5, 0.5
                                                                    [1, 1, 1, 1, 1],  # 0.5, 0.1
                                                                    [0, 0, 0, 0, 0],  # 1, 0
                                                                    [0.5, 0.5, 0.5, 0.5, 0.5],  # 1, 0.5
                                                                    [1, 1, 1, 1, 1]],
                                                                    [[0, 0, 0, 0, 0],  # 0, 0
                                                                     [0.5, 0.5, 0.5, 0.5, 0.5],  # 0, 0.5
                                                                     [1, 1, 1, 1, 1],  # 0, 1
                                                                     [0, 0, 0, 0, 0],  # 0.5, 0
                                                                     [0.5, 0.5, 0.5, 0.5, 0.5],  # 0.5, 0.5
                                                                     [1, 1, 1, 1, 1],  # 0.5, 0.1
                                                                     [0, 0, 0, 0, 0],  # 1, 0
                                                                     [0.5, 0.5, 0.5, 0.5, 0.5],  # 1, 0.5
                                                                     [1, 1, 1, 1, 1]],
                                                                    [[0, 0, 0, 0, 0],  # 0, 0
                                                                     [0.5, 0.5, 0.5, 0.5, 0.5],  # 0, 0.5
                                                                     [1, 1, 1, 1, 1],  # 0, 1
                                                                     [0, 0, 0, 0, 0],  # 0.5, 0
                                                                     [0.5, 0.5, 0.5, 0.5, 0.5],  # 0.5, 0.5
                                                                     [1, 1, 1, 1, 1],  # 0.5, 0.1
                                                                     [0, 0, 0, 0, 0],  # 1, 0
                                                                     [0.5, 0.5, 0.5, 0.5, 0.5],  # 1, 0.5
                                                                     [1, 1, 1, 1, 1]],
                                                                    [[0, 0, 0, 0, 0],  # 0, 0
                                                                     [0.5, 0.5, 0.5, 0.5, 0.5],  # 0, 0.5
                                                                     [1, 1, 1, 1, 1],  # 0, 1
                                                                     [0, 0, 0, 0, 0],  # 0.5, 0
                                                                     [0.5, 0.5, 0.5, 0.5, 0.5],  # 0.5, 0.5
                                                                     [1, 1, 1, 1, 1],  # 0.5, 0.1
                                                                     [0, 0, 0, 0, 0],  # 1, 0
                                                                     [0.5, 0.5, 0.5, 0.5, 0.5],  # 1, 0.5
                                                                     [1, 1, 1, 1, 1]],
                                                                    [[0, 0, 0, 0, 0],  # 0, 0
                                                                     [0, 0, 0, 0, 0],  # 0, 0.5
                                                                     [0, 0, 0, 0, 0],  # 0, 1
                                                                     [0, 0, 0, 0, 0],  # 0.5, 0
                                                                     [0, 0, 0, 0, 0],  # 0.5, 0.5
                                                                     [0, 0, 0, 0, 0],  # 0.5, 0.1
                                                                     [0, 0, 0, 0, 0],  # 1, 0
                                                                     [0, 0, 0, 0, 0],  # 1, 0.5
                                                                     [0, 0, 0, 0, 0]]]))

        self.assertEqual(s0, 3)


if __name__ == '__main__':
    unittest.main()
