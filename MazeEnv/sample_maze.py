"""
Reinforcement learning maze
"""

import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_ROW = 10  # maze rows
MAZE_CLM = 10  # maze column


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.available_action = ['^', 'v', '<', '>']
        self.n_actions = len(self.available_action)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_CLM * UNIT, MAZE_ROW * UNIT))
        self.origin = np.array([20, 20])
        self.hells = np.array([[3,6], [3,7],[3, 8],[4, 6],[5, 6],[6, 6],[6, 7],[6, 8],[7, 8],[8, 6],[8, 7],[8,8],[9, 6]], np.float64)
        self.hellcoords=list()
        self.paradises = np.array([[5,7]], np.float64)
        self.paradisecoords=list()
        self._build_maze()

    # create a ground type, it can be either positive(paradise) or negative(hell) rewarded
    def create_ground(self,coord,ground_type):
        gr_center = self.origin + np.array([UNIT * (coord[0]), UNIT * (coord[1])])
        if ground_type == 'paradise':
            ground = self.canvas.create_oval(
                        gr_center[0] - 15, gr_center[1] - 15,
                        gr_center[0] + 15, gr_center[1] + 15,
                        fill='yellow')
            self.hellcoords.append(self.canvas.coords(ground))
        elif ground_type == 'hell':
            ground = self.canvas.create_rectangle(
                        gr_center[0] - 15, gr_center[1] - 15,
                        gr_center[0] + 15, gr_center[1] + 15,
                        fill='black')
            self.paradisecoords.append(self.canvas.coords(ground))
        return ground

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_ROW * UNIT,
                           width=MAZE_CLM * UNIT)

        # create grids
        for c in range(0, MAZE_CLM * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_ROW * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_ROW * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_ROW * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # now just an illustration on how to build different types of maze ground, we shall carefully designate it later
        for row in self.hells:
            self.create_ground(row, 'hell')
        for row in self.paradises:
            self.create_ground(row, 'paradise')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='red')
        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_ROW - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_CLM - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state
        print(s_)

        # reward function
        reward = 0
        done = False
        for row in self.hellcoords:
            if s_ == row:
                reward = 1
                done = True
        for row in self.paradisecoords:
            if s_ == row:
                reward = -1
                done = True

        return s_, reward, done

    def render(self):
        time.sleep(0.001)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Maze()
    env.after(1, update)
    env.mainloop()
