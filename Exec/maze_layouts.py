from MazeEnv.moderate_maze import MazeSimulator


class MazeLarge:
    def __init__(self):
        # set the size of maze:     column x row
        self.size_maze = [20, 20]
        self.init_pos = [0, 0]

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
    def __init__(self):
        # set the size of maze:     column x row
        self.size_maze = [20, 20]
        self.init_pos = [0, 0]

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
