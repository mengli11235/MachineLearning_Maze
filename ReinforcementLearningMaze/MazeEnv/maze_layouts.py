import numpy as np
from ..MazeEnv.maze_env import MazeSimulator


class MazeLarge:
    def __init__(self, init_pos=[0, 0], size_maze=[20, 20]):
        # set the size of maze:     column x row
        self.size_maze = size_maze
        self.init_pos = init_pos

    def init_maze(self, is_render):
        maze = MazeSimulator(self.size_maze[1], self.size_maze[0], self.init_pos, is_render)

        # the cost of each move
        maze.set_step_penalty(-1)

        # set walls
        walls = np.array(
            [[5, 2], [6, 2], [7, 2], [8, 2], [9, 2], [11, 2], [10, 2], [12, 2], [8, 0], [8, 1],  # NO1
             [16, 0], [16, 1], [16, 2], [16, 3], [16, 4], [16, 5],  # NO2
             [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [0, 9], [1, 9], [2, 9],  # NO3
             [7, 13], [7, 14], [7, 15], [7, 16],  # NO4
             [0, 17], [1, 17], [2, 17], [3, 17], [4, 17], [5, 17], [6, 17], [7, 17],
             [10, 5], [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [11, 10], [12, 10], [13, 10],  # NO5
             [16, 13], [16, 14], [16, 15], [16, 16], [16, 17],  # NO6
             [11, 15], [12, 15], [13, 15], [14, 15], [15, 15],
             [11, 16], [11, 17], [11, 18], [11, 19]],
            np.float64)

        # set pits (extra cost when stepping on it)
        pits = np.array(
            [[7, 11], [14, 10], [14, 12]],
            np.float64)

        # set exit, with certain rewards
        exits = np.array(
            [[13, 18]],
            np.float64)

        maze.set_agent()

        for row in walls:
            maze.set_wall(row, -3, False)
        for row in pits:
            maze.set_fixed_obj(row, -15, False)
        for row in exits:
            # You might need to change set_fixed_obj() function if you change the reward for exit
            maze.set_fixed_obj(row, 500, True)

        # set chest and its key
        maze.set_key_chest([18, 18], [18, 0], '1', 0, 1500)
        maze.set_key_chest([2, 5], [8, 18], '2', 0, 1000)

        # maze.set_key_chest([14, 1], [4, 14], '3', 0, 500)

        # build the rendered maze
        maze.build_maze()
        return maze


class MazeMedium:
    def __init__(self, init_pos=[0, 0], size_maze=[12, 12]):
        # set the size of maze:     column x row
        self.size_maze = size_maze
        self.init_pos = init_pos

    def init_maze(self, is_render):
        maze = MazeSimulator(self.size_maze[1], self.size_maze[0], self.init_pos, is_render)

        # the cost of each move
        maze.set_step_penalty(-1)

        # set walls
        walls = np.array(
            [[3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [5, 0],
             [9, 0], [9, 1], [9, 2], [9, 3],
             [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [0, 6], [1, 6],
             [5, 4], [5, 5], [5, 6], [6, 5], [7, 5],
             [5, 9], [6, 9], [7, 8], [7, 9], [8, 9], [9, 9], [5, 10]],
            np.float64)

        # set pits (extra cost when stepping on it)
        pits = np.array(
            [[8, 4], [6, 7]],
            np.float64)

        # set exit, with certain rewards
        exits = np.array(
            [[8, 6]],
            np.float64)

        maze.set_agent()

        for row in walls:
            maze.set_wall(row, -3, False)
        for row in pits:
            maze.set_fixed_obj(row, -150, False)
        for row in exits:
            # You might need to change set_fixed_obj() function if you change the reward for exit
            maze.set_fixed_obj(row, 300, True)

        # set chest and its key
        maze.set_key_chest([4, 0], [7, 4], '1', 0, 700)

        # build the rendered maze
        maze.build_maze()
        return maze


class MazeSmall:
    def __init__(self, init_pos=[0, 0], size_maze=[9, 6]):
        # set the size of maze:     column x row
        self.size_maze = size_maze
        self.init_pos = init_pos

    def init_maze(self, is_render):
        maze = MazeSimulator(self.size_maze[1], self.size_maze[0], self.init_pos, is_render)

        # the cost of each move
        maze.set_step_penalty(-1)

        # set walls
        walls = np.array(
            [[2, 1], [2, 2], [2, 3],
             [5, 4],
             [7, 0], [7, 1], [7, 2]],
            np.float64)

        # set pits (extra cost when stepping on it)
        pits = np.array(
            [

            ],
            np.float64)

        # set exit, with certain rewards
        exits = np.array(
            [[8, 0]],
            np.float64)

        maze.set_agent()

        for row in walls:
            maze.set_wall(row, -3, False)
        for row in pits:
            maze.set_fixed_obj(row, -3, False)
        for row in exits:
            # You might need to change set_fixed_obj() function if you change the reward for exit
            maze.set_fixed_obj(row, 50, True)

        # set chest and its key
        # maze.set_key_chest([0, 5], [8, 5], '1', 0, 50)

        # build the rendered maze
        maze.build_maze()
        return maze
