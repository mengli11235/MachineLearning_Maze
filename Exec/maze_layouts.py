import numpy as np
from MazeEnv.moderate_maze import MazeSimulator


class MazeLarge:
    def __init__(self, init_pos=[0, 0], size_maze=[20, 20]):
        # set the size of maze:     column x row
        self.size_maze = size_maze
        self.init_pos = init_pos

    def init_maze(self, is_render):
        maze = MazeSimulator(self.size_maze[1], self.size_maze[0], self.init_pos, is_render)

        maze.set_step_penalty(-1)

        # set fixed object ([column, row], reward, isFinishedWhenReach)
        maze.set_fixed_obj([8, 8], -1000, False)
        maze.set_key_chest([7, 5], [2, 17], 'k', 600, 1000)
        maze.set_key_chest([19, 15], [0, 0], 'k2', 0, 600)
        maze.set_key_chest([10, 18], [0, 3], 'w', 0, 600)

        # maze.set_key_chest([19, 15], [0, 0], 'key', 0, 600)
        # maze.set_key_chest([3, 3], [18, 15], 'key2', 0, 800)
        # maze.set_key_chest([2, 14], [18, 4], 'key3', 0, 1000)

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
            maze.set_wall(row, 0, False)
        for row in pits:
            maze.set_fixed_obj(row, -3, False)
        for row in exits:
            # You might need to change set_fixed_obj() function if you change the reward for exit
            maze.set_fixed_obj(row, 400, True)

        maze.set_key_chest([0, 5], [8, 5], 'k', 0, 600)

        # build the rendered maze
        maze.build_maze()
        return maze
