# -*- coding: utf-8 -*-
"""
Grid world environment
"""
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from past.utils import old_div

from arsenal import Alphabet


class Action(object):
    def __init__(self, name, dx, dy):
        self.name = name
        self.dx = dx
        self.dy = dy

    def __iter__(self):
        return iter((self.dx, self.dy))

    def __repr__(self):
        return self.name


class GridWorld(object):
    """
    A two-dimensional grid MDP.  All you have to do is specify the grid as a list
    of lists of rewards; use None for an obstacle. Also, you should specify the
    terminal states.  An action is an (x, y) unit vector; e.g. (1, 0) means move
    east.
    """
    EMPTY, WALL, START, GOAL, QM, PIT = np.array(['0', 'W', 'S', 'G', '?', 'P'], dtype='|S1')

    def __init__(self, args=None):
        try:
            self.mapname = args.map
            self.H = args.horizon
            self.step = 1.0 / self.H
            self.randomness = args.randomness
            self.infinite = args.infinite
        except:
            self.mapname = args['map']
            self.randomness = args['randomness']
            self.H = args['horizon']
            self.step = 1.0 / self.H
            self.infinite = args['infinite'] # binary flag whether an agent ever terminates or not

        self.d = 1
        self.SHIFT = .1

        # Load Map
        __location__ = os.path.dirname((os.path.abspath(__file__)))
        default_map_dir = os.path.join(__location__, "maps")

        self.ax = None
        self.A = [
            Action('⬆', 0, -1),  # up     b/c 0,0 is upper left
            Action('⬇', 0, 1),  # down
            Action('⬅', -1, 0),  # left
            Action('➡', 1, 0),  # right
        ]

        self.states = set()
        self.reward = {}
        self.constraint = {}

        self.grid = grid = np.loadtxt(os.path.join(default_map_dir, self.mapname), dtype='|S1')
        [self.rows, self.cols] = self.grid.shape
        self.num_states = grid.size

        # Parse Map
        # self.initial_state = np.argwhere(self.grid == self.START)[0]
        # self.initial_state = tuple(self.initial_state)

        # self.terminals = np.argwhere(self.grid == self.GOAL)[0]
        # self.terminals = [tuple(self.terminals)]

        self.initial_states = []
        self.terminals = []
        self.pit_grid = np.zeros(self.grid.shape)

        # Convert Numpy bytes to int
        new_grid = np.zeros(self.grid.shape)

        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r, c] != self.WALL:
                    s = (r, c)
                    self.states.add(s)

                    c_value = 0.0  # c_value = [0., 0.]
                    if grid[r, c] == self.PIT:
                        r_value = -1.0
                        c_value = 1.0  # c_value = [1., 5.]
                        # self.terminals.append((r, c))
                        self.pit_grid[r][c] = 1.0
                    elif grid[r, c] == self.GOAL:
                        r_value = -1.0
                        self.terminals.append((r, c))
                    elif grid[r, c] == self.START:
                        r_value = -1.0
                        self.initial_states.append((r, c))
                    elif grid[r, c] == self.EMPTY:
                        r_value = -1.0
                    else:
                        raise Exception("Unknown grid attribute: ", grid[r, c])

                    self.reward[s] = r_value
                    self.constraint[s] = c_value
                    new_grid[r, c] = r_value
                else:
                    new_grid[r, c] = None
        self.grid_for_plot = grid.copy()
        new_grid = np.where(np.isnan(new_grid), None, new_grid)
        self.grid = new_grid.copy()

    def encode(self):
        # Encode problem into an MDP so we can call its solver.
        Si = Alphabet()
        Ai = Alphabet()
        P = np.zeros((len(self.states), len(self.A), len(self.states)))
        R = np.zeros((len(self.states), len(self.A), len(self.states)))
        C = np.zeros((len(self.states), len(self.A), len(self.states)))
        if self.d > 0:
            C = np.zeros((self.d, len(self.states), len(self.A), len(self.states)))

        for s in self.states:
            # row, col = s
            random_move_prob = np.random.uniform(low=self.randomness, high=self.randomness + 0.2)
            si = Si[s]

            # P = (1 - randomness) + sum_|A| {randomness / |A|}
            for a in self.A:
                ai = Ai[a]
                sp, r, c = self.simulate(s, a)
                spi = Si[sp]
                P[si, ai, spi] += 1 * (1 - random_move_prob)
                R[si, ai, spi] = r
                if self.d > 0:
                    C[:, si, ai, spi] = c
                else:
                    C[si, ai, spi] = c

                for a_r in self.A:
                    ai_r = Ai[a_r]
                    sp, r, c = self.simulate(s, a_r)
                    spi = Si[sp]
                    P[si, ai, spi] += 1 * random_move_prob / (len(self.A))

        s0 = np.zeros(len(self.states))
        # s0[Si[self.initial_state]] = 1
        s0[list(map(Si, self.initial_states))] = 1
        s0 = s0 / np.sum(s0)

        Si.freeze(); Ai.freeze()
        return (s0, P, R, C), Si, Ai

    def simulate(self, s, a):
        if s in self.terminals:
            # return s, self.reward[s], self.constraint[s]
            # return s, self.reward[s] * self.step, self.constraint[s] * self.step
            # return s, 0, self.constraint[s] * self.step
            if not self.infinite:
                return s, 0, self.constraint[s]
            else:
                s = random.choice(self.initial_states)
                return s, 0, self.constraint[s]
        dx, dy = a
        sp = (s[0] + dy, s[1] + dx)
        if sp in self.states:
            sp = sp
            r = self.reward[sp]
            if sp in self.constraint:
                c = self.constraint[sp]
            else:
                c = 0. # c = [0., 0.]
        else:
            sp = s  # stay in same state if we hit a wall
            r = -1.0
            c = 0. # c = [0., 0.]
            # r = -0.05   # negative reward for crashing into a wall.
        return sp, r, c

    def showLearning(self, policy, value_fn = None, path=None):
        """
        This method is partly borrowed from
        https://github.com/rlpy/rlpy/blob/af25d2011fff1d61cb7c5cc8992549808f0c6103/rlpy/Domains/GridWorld.py
        """
        plt.figure("Value Function")
        self.grid[self.grid == None] = -2
        self.valueFunction_fig = plt.imshow(
            # self.grid.astype(int), # self.map
            (self.grid + self.pit_grid).astype(int),
            cmap = 'tab20c',#'Pastel1', # cmap = 'RdYlGn', # cmap='ValueFunction',
            interpolation='nearest',
            vmin=0, # self.MIN_RETURN
            vmax=19) # self.MAX_RETURN
        plt.xticks(np.arange(self.cols) + 0.5, color='w')
        plt.yticks(np.arange(self.rows) + 0.5, color='w')
        plt.grid(visible=True, which='major', linestyle='-', alpha=0.9)

        # Create quivers for each action. 4 in total
        X = np.arange(self.rows) - self.SHIFT
        Y = np.arange(self.cols)
        X, Y = np.meshgrid(X, Y)
        DX = DY = np.ones(X.shape)
        C = np.zeros(X.shape)
        C[0, 0] = 1  # Making sure C has both 0 and 1
        # length of arrow/width of bax. Less then 0.5 because each arrow is
        # offset, 0.4 looks nice but could be better/auto generated
        arrow_ratio = 0.4
        Max_Ratio_ArrowHead_to_ArrowLength = 0.25
        ARROW_WIDTH = 0.5 * Max_Ratio_ArrowHead_to_ArrowLength / 5.0
        self.upArrows_fig = plt.quiver(
            Y,
            X,
            DY,
            DX,
            C,
            units='y',
            cmap='Blues', # cmap='Actions',
            scale_units="height",
            scale=old_div(self.rows,
                          arrow_ratio),
            width=-
                  1 *
                  ARROW_WIDTH)
        self.upArrows_fig.set_clim(vmin=0, vmax=1)
        X = np.arange(self.rows)  + self.SHIFT
        Y = np.arange(self.cols)
        X, Y = np.meshgrid(X, Y)
        self.downArrows_fig = plt.quiver(
            Y,
            X,
            DY,
            DX,
            C,
            units='y',
            cmap='Blues', #cmap='Actions',
            scale_units="height",
            scale=old_div(self.rows,
                          arrow_ratio),
            width=-
                  1 *
                  ARROW_WIDTH)
        self.downArrows_fig.set_clim(vmin=0, vmax=1)
        X = np.arange(self.rows)
        Y = np.arange(self.cols) - self.SHIFT
        X, Y = np.meshgrid(X, Y)
        self.leftArrows_fig = plt.quiver(
            Y,
            X,
            DY,
            DX,
            C,
            units='x',
            cmap='Blues', #cmap='Actions',
            scale_units="width",
            scale=old_div(self.cols,
                          arrow_ratio),
            width=ARROW_WIDTH)
        self.leftArrows_fig.set_clim(vmin=0, vmax=1)
        X = np.arange(self.rows)
        Y = np.arange(self.cols) + self.SHIFT
        X, Y = np.meshgrid(X, Y)
        self.rightArrows_fig = plt.quiver(
            Y,
            X,
            DY,
            DX,
            C,
            units='x',
            cmap='Blues', #cmap='Actions',
            scale_units="width",
            scale=old_div(self.cols,
                          arrow_ratio),
            width=ARROW_WIDTH)
        self.rightArrows_fig.set_clim(vmin=0, vmax=1)
        plt.show()
        plt.figure("Value Function")

        V = np.ones((self.rows, self.cols)) * 18
        # Boolean 3 dimensional array. The third array highlights the action.
        # Thie mask is used to see in which cells what actions should exist
        Mask = np.ones(
            (self.cols,
             self.rows,
             len(self.A)),
            dtype='bool')
        arrowSize = np.zeros(
            (self.cols,
             self.rows,
             len(self.A)),
            dtype='float')
        # 0 = suboptimal action, 1 = optimal action
        arrowColors = np.zeros(
            (self.cols,
             self.rows,
             len(self.A)),
            dtype='uint8')
        for r in range(1, self.rows-1):
            for c in range(1, self.cols-1):
                s = (r, c)
                # As = self.possibleActions(s)
                # terminal = self.isTerminal(s)
                Qs = policy[s]
                bestA = np.where(policy[s] > 0)[0]
                Mask[c, r, :] = False
                arrowColors[c, r, bestA] = 1
                arrowSize[c, r, :] = Qs

                if value_fn is None:
                    if self.grid_for_plot[r, c] == self.WALL:
                        V[r, c] = 18
                    if self.grid_for_plot[r, c] == self.GOAL:
                        V[r, c] = 8
                    if self.grid_for_plot[r, c] == self.PIT:
                        V[r, c] = 13
                    if self.grid_for_plot[r, c] == self.EMPTY:
                        V[r, c] = 15
                    if self.grid_for_plot[r, c] == self.START:
                        V[r, c] = 10

                    # V[r, c] = int(self.grid[r, c])  # max(Qs)
                    # V[r, c] = int(self.grid[r, c] * 1.4 + 3.5 * self.pit_grid[r, c])
                    # if (r, c) in self.initial_states:
                    #     V[r, c] = 0
                    if (r, c) in self.terminals:
                        #V[r, c] = 0.3
                        arrowColors[c, r, bestA] = 0
                        arrowSize[c, r, :] = np.zeros(len(self.A))
                else:
                    V[r, c] = value_fn[r, c]
        # Show Value Function
        self.valueFunction_fig.set_data(V)
        # Show Policy Up Arrows
        DX = arrowSize[:, :, 0]
        DY = np.zeros((self.rows, self.cols))
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 0])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 0])
        C = np.ma.masked_array(arrowColors[:, :, 0], mask=Mask[:, :, 0])
        self.upArrows_fig.set_UVC(DY, DX, C)
        # Show Policy Down Arrows
        DX = -arrowSize[:, :, 1]
        DY = np.zeros((self.rows, self.cols))
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 1])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 1])
        C = np.ma.masked_array(arrowColors[:, :, 1], mask=Mask[:, :, 1])
        self.downArrows_fig.set_UVC(DY, DX, C)
        # Show Policy Left Arrows
        DX = np.zeros((self.rows, self.cols))
        DY = -arrowSize[:, :, 2]
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 2])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 2])
        C = np.ma.masked_array(arrowColors[:, :, 2], mask=Mask[:, :, 2])
        self.leftArrows_fig.set_UVC(DY, DX, C)
        # Show Policy Right Arrows
        DX = np.zeros((self.rows, self.cols))
        DY = arrowSize[:, :, 3]
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 3])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 3])
        C = np.ma.masked_array(arrowColors[:, :, 3], mask=Mask[:, :, 3])
        self.rightArrows_fig.set_UVC(DY, DX, C)
        if path is None:
            path = os.path.abspath(os.curdir)
        plt.savefig(path + '/' + self.mapname, bbox_inches='tight')
        plt.draw()
        plt.close()


if __name__ == '__main__':
    path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    args = {'map': "4x4", 'horizon': 1000, 'randomness': 0, 'infinite': False}

    gridworld = GridWorld(args)

    (s0, P, R, C), Si, Ai = gridworld.encode()

    random_policy = np.ones([len(Si), len(Ai)]) * 0.25
    grid_index = map(Si.lookup, range(16))
    grid_policy = dict(zip(grid_index, random_policy))
    gridworld.showLearning(grid_policy, path = path + '/log')


