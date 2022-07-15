# -*- coding: utf-8 -*-
"""
Grid world environment
"""
import random
import pylab as pl
import numpy as np
from matplotlib.table import Table
import os
import matplotlib.pyplot as plt
from past.utils import old_div

from arsenal import Alphabet
from arsenal.viz import update_ax

class Action(object):
    def __init__(self, name, dx, dy):
        self.name = name
        self.dx = dx
        self.dy = dy
    def __iter__(self):
        return iter((self.dx, self.dy))
    def __repr__(self):
        return self.name


class BoxGridWorld(object):
    """A two-dimensional grid MDP.  All you have to do is specify the grid as a list
    of lists of rewards; use None for an obstacle. Also, you should specify the
    terminal states.  An action is an (x, y) unit vector; e.g. (1, 0) means move
    east.

    """
    EMPTY, WALL, START, GOAL, PIT, BOX = np.array(['0','W','S','G', 'P', 'X'], dtype='|S1')

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
            self.infinite = args['infinite']  # binary flag whether an agent ever terminates or not

        self.d = 1
        self.SHIFT = .1

        # Load Map
        __location__ = os.path.dirname((os.path.abspath(__file__)))
        default_map_dir = os.path.join(__location__, "maps")

        self.ax = None
        self.A = [
            Action('⬆',0,-1), # up     b/c 0,0 is upper left
            Action('⬇',0,1),  # down
            Action('⬅',-1,0), # left
            Action('➡',1,0),  # right
        ]

        self.states = set()
        self.reward = {}
        self.constraint = {}
        self.walls = set()

        self.grid = grid = np.loadtxt(os.path.join(default_map_dir, self.mapname), dtype='|S1')
        [self.rows, self.cols] = self.grid.shape
        self.num_states = grid.size

        # Parse Map
        self.initial_states = np.argwhere(self.grid == self.START)[0]
        self.box = np.argwhere(self.grid == self.BOX)[0]
        self.initial_states = tuple(self.initial_states.tolist() + self.box.tolist())
        self.terminals = []

        # Convert Numpy bytes to int
        new_grid = np.zeros(self.grid.shape)

        params = [(ra, ca, rb, cb)
                    for ra in range(self.rows)
                    for ca in range(self.cols)
                    for rb in range(self.rows)
                    for cb in range(self.cols) ]

        for (ra, ca, rb, cb) in params:
            # Agent cant be in wall location
            # Block cant be in wall location
            if grid[ra, ca] != self.WALL and grid[rb, cb] != self.WALL:
                s = (ra, ca, rb, cb)
                self.states.add(s)

                if grid[ra,ca] == self.GOAL:
                    self.terminals.append(tuple(s))
                    r_value = -1.0
                elif grid[ra,ca] == self.START: r_value = -1.0
                elif grid[ra,ca] == self.EMPTY: r_value = -1.0
                elif grid[ra,ca] == self.BOX: r_value = -1.0
                else: raise Exception("Unknown grid attribute: ", grid[ra, ca])

                self.reward[s] = r_value
                new_grid[ra, ca] = r_value
            elif grid[ra, ca] == self.WALL:
                self.walls.add((ra,ca))
                new_grid[ra, ca] = None
                new_grid[ra, ca] = None

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
            #row, col = s
            si = Si[s]
            for a in self.A:
                ai = Ai[a]
                sp, r, c = self.simulate(s, a)
                spi = Si[sp]
                P[si,ai,spi] += 1 * (1-self.randomness)
                R[si,ai,spi] = r
                if self.d > 0:
                    C[:, si, ai, spi] = c
                else:
                    C[si, ai, spi] = c

                for a_r in self.A:
                    ai_r = Ai[a_r]
                    sp, r, c = self.simulate(s, a_r)
                    spi = Si[sp]
                    P[si, ai_r, spi] += 1 * self.randomness/(len(self.A))
                    R[si, ai_r, spi] = r
                    if self.d > 0:
                        C[:, si, ai_r, spi] = c
                    else:
                        C[si, ai_r, spi] = c

        s0 = np.zeros(len(self.states))
        s0[Si[self.initial_states]] = 1

        Si.freeze(); Ai.freeze()
        return (s0, P, R, C), Si, Ai

    def check_box_location(self, box_location):
        open_area = []
        for a in self.A:
            dx, dy = a
            sp = (box_location[0] + dy, box_location[1] + dx)
            open_area.append( int(sp in self.walls) )
        return [1] if sum(open_area) >= 2 else [0]


    def simulate(self, s, a):
        if s in self.terminals:
            # return s, self.reward[s]*self.steps, self.check_box_location(s[2:])*self.steps
            if not self.infinite:
                return s, 0, self.constraint[s]
            else:
                s = self.initial_states
                return s, 0, self.check_box_location(s[2:])

        # Transition the agent to next state
        dx, dy = a
        sp = [s[0] + dy, s[1] + dx, s[2], s[3]]

        # If the agent next position is the box location
        # move the box in the same direction
        if tuple(sp[:2]) == tuple(s[2:]):
            sp[2:] = [(s[2] + dy), (s[3] + dx)]
        sp = tuple(sp)

        if sp in self.states:
            sp = sp
            r = self.reward[sp]
        else:
            sp = s # stay in same state if we hit a wall
            r = -1.
        # c = self.check_box_location(sp[2:])*self.steps
        c = self.check_box_location(sp[2:])
        return sp, r, c

    def showLearning(self, policy, value_fn = None, box_position=None, start_position=None, path=None):
        """
        This method is partly borrowed from
        https://github.com/rlpy/rlpy/blob/af25d2011fff1d61cb7c5cc8992549808f0c6103/rlpy/Domains/GridWorld.py
        """
        plt.figure("Value Function")
        self.grid[self.grid == None] = -2
        self.valueFunction_fig = plt.imshow(
            # self.grid.astype(int), # self.map
            self.grid.astype(int),
            cmap = 'tab20c',#'Pastel1', # cmap = 'RdYlGn', # cmap='ValueFunction',
            interpolation='nearest',
            vmin=0, # self.MIN_RETURN
            vmax=19) # self.MAX_RETURN
        plt.xticks(np.arange(self.cols) + 0.5, color='w')
        plt.yticks(np.arange(self.rows) + 0.5, color='w')
        plt.grid(b=True, which='major', linestyle='-', alpha=0.9)

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
                if self.grid[s] != -2:
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
                        if start_position is None:
                            if self.grid_for_plot[r, c] == self.START:
                                V[r, c] = 10
                        else:
                            if self.grid_for_plot[r, c] == self.START:
                                V[r, c] = 15
                            if (np.array([r,c]) == start_position).all():
                                V[r, c] = 10
                        if box_position is not None:
                            if (np.array([r,c]) == box_position).all():
                                V[r, c] = 0

                        # V[r, c] = int(self.grid[r, c])  # max(Qs)
                        # V[r, c] = int(self.grid[r, c] * 1.4 + 3.5 * self.pit_grid[r, c])
                        # if (r, c) in self.initial_states:
                        #     V[r, c] = 0
                        if (r, c) == self.terminals[0][:2]:
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
