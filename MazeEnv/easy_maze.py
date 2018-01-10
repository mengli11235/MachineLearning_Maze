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


class MazeSimulator(tk.Tk, object):
    pixel = 40
    anchor = pixel/2
    grid_height = 1
    grid_width = 1
    object_list = []
    final_object_list = []
    ending_con_map = {} #dictionary
    agent_con_map = {}

    def __init__(self, grid_height, grid_width, init_position, is_render):
        if is_render:
            super(MazeSimulator, self).__init__()
        self.is_render = is_render
        self.init_position = init_position
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.available_action = ['^', 'v', '<', '>']
        self.n_actions = len(self.available_action)

        self._init_grid()
        self.set_agent()

    def _init_grid(self):
        if self.is_render:
            self.title('easy maze')
            self.geometry('{0}x{1}'.format(self.grid_width * self.pixel, self.grid_height * self.pixel))
            self.canvas = tk.Canvas(self, bg='white',
                                    height=self.grid_height * self.pixel,
                                    width=self.grid_width * self.pixel)

            # create grids
            for c in range(0, self.grid_width * self.pixel, self.pixel):
                x0, y0, x1, y1 = c, 0, c, self.grid_height * self.pixel
                self.canvas.create_line(x0, y0, x1, y1)
            for r in range(0, self.grid_height * self.pixel, self.pixel):
                x0, y0, x1, y1 = 0, r, self.grid_height * self.pixel, r
                self.canvas.create_line(x0, y0, x1, y1)

    def set_agent(self):
        self.agent = [self.init_position[0], self.init_position[1]]

        if self.is_render:
            # create origin coord
            self.origin_coord = np.array(
                [self.anchor, self.anchor])
            # create agent in Maze
            self.agent_coord = np.array(
                [self.anchor + (self.pixel * self.agent[0]), self.anchor + (self.pixel * self.agent[1])])
            # create agent
            self.agent_avatar = self.canvas.create_rectangle(
                self.agent_coord[0] - 15, self.agent_coord[1] - 15,
                self.agent_coord[0] + 15, self.agent_coord[1] + 15,
                fill='red')

    def set_fixed_obj(self, position, reward, is_done):
        self.object_list.append([position, reward, is_done])

        if self.is_render:
            # draw this object
            new_obj = self.origin_coord + np.array([self.pixel * position[0], self.pixel * position[1]])
            if reward < 0:
                self.canvas.create_rectangle(
                    new_obj[0] - 15, new_obj[1] - 15,
                    new_obj[0] + 15, new_obj[1] + 15,
                    fill='black')
            elif reward > 0:
                self.canvas.create_oval(
                    new_obj[0] - 15, new_obj[1] - 15,
                    new_obj[0] + 15, new_obj[1] + 15,
                    fill='yellow')
            else:
                self.canvas.create_rectangle(
                    new_obj[0] - 15, new_obj[1] - 15,
                    new_obj[0] + 15, new_obj[1] + 15,
                    fill='white')

    def set_collect_all_rewards(self, reward_position_list, reward, flag_name):
        for ind_pos in reward_position_list:
            self.set_fixed_obj(ind_pos, reward, flag_name)
        self.ending_con_map[flag_name] = len(reward_position_list) # add new keys to dic

    def build_maze(self):
        self.final_object_list = self.object_list[:]
        if self.is_render:
            # pack all
            self.canvas.pack()

    def reset(self):
        self.agent[0] = self.init_position[0]
        self.agent[1] = self.init_position[1]
        self.agent_con_map = {}
        self.object_list = self.final_object_list[:]

        if self.is_render:
            self.update()
            time.sleep(0.5)
            self.canvas.delete(self.agent_avatar)
            self.agent_coord = np.array(
                [self.anchor + (self.pixel * self.agent[0]), self.anchor + (self.pixel * self.agent[1])])
            self.agent_avatar = self.canvas.create_rectangle(
                self.agent_coord[0] - 15, self.agent_coord[1] - 15,
                self.agent_coord[0] + 15, self.agent_coord[1] + 15,
                fill='red')

        # return position of agent
        return self.agent

    def taking_action(self, action):
        state = self.agent
        new_position = [0, 0]
        if action == 0:   # ^
            if state[1] > 0:
                new_position[1] = -1
        elif action == 1:   # v
            if state[1] < (self.grid_height - 1):
                new_position[1] = 1
        elif action == 2:   # <
            if state[0] > 0:
                new_position[0] = -1
        elif action == 3:   # >
            if state[0] < (self.grid_width - 1):
                new_position[0] = 1

        if self.is_render:
            self.canvas.move(self.agent_avatar, new_position[0] * self.pixel, new_position[1] * self.pixel)  # move agent

        self.agent[0] = self.agent[0] + new_position[0]  # next state
        self.agent[1] = self.agent[1] + new_position[1]  # next state

        # reward function
        outcomes = [obj for obj in self.object_list if obj[0][0] == self.agent[0] and obj[0][1] == self.agent[1]]
        if len(outcomes) > 0:
            obj = outcomes[0]
            reward = obj[1]
            is_done_content = obj[2]
            if isinstance(is_done_content, str):
                idx_list = [idx for idx in range(len(self.object_list)) if self.object_list[idx][0][0] == self.agent[0]
                 and self.object_list[idx][0][1] == self.agent[1]]
                for index in idx_list:
                    del self.object_list[index]
                if is_done_content in self.agent_con_map:
                    self.agent_con_map[is_done_content] = self.agent_con_map[is_done_content] + 1

                else:
                    self.agent_con_map[is_done_content] = 1

                if self.ending_con_map[is_done_content] == self.agent_con_map[is_done_content]:
                    is_done = True
                else:
                    is_done = False
            else:
                is_done = is_done_content
        else:
            reward = 0
            is_done = False
        if is_done and self.is_render:
            print(self.agent)
        return self.agent, reward, is_done

    def render(self, time_in_ms):
        if self.is_render:
            if time_in_ms > 0:
                time.sleep(time_in_ms)
                self.update()
            else:
                self.update()


# if __name__ == '__main__':
#     print()
    # env = Maze()
    # env.after(1, update)
    # env.mainloop()
