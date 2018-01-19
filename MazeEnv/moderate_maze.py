
"""
Reinforcement learning maze

Please change the maze objects (e.g. walls, pits, chests, keys, exits in reset())
"""

import numpy as np
import time
import tkinter as tk

class MazeSimulator(tk.Tk, object):
    step_penalty = 0
    pixel = 40
    anchor = pixel/2
    grid_height = 1
    grid_width = 1
    object_list = []
    final_object_list = []
    # ending_con_map = {} #dictionary
    # agent_con_map = {}
    walls = []

    key_chest_to_set = []
    key_list = []
    chest_list = []
    agent_keys = []
    agent_chests = []

    def __init__(self, grid_height, grid_width, init_position, is_render):
        if is_render:
            super(MazeSimulator, self).__init__()
        self.is_render = is_render
        self.init_position = init_position
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.available_action = ['^', 'v', '<', '>']
        self.n_actions = len(self.available_action)

        walls = np.array(
            [[3, 9], [3, 10], [3, 11], [3, 12], [3, 13],
             [6, 4], [6, 5], [6, 6], [6, 7], [6, 8],
             [9, 9], [9, 10], [9, 11], [9, 12], [9, 13],
             [12, 4], [12, 5], [12, 6], [12, 7], [12, 8],
             [15, 9], [15, 10], [15, 11], [15, 12], [15, 13]],
            np.float64)

        pits = np.array(
            [[2, 3], [10, 9]],
            np.float64)
        exits = np.array(
            [[18, 6]],
            np.float64)

        self._init_grid()
        self.set_agent()
        for row in walls:
            self.set_wall(row, 0, False)
        for row in pits:
            self.set_fixed_obj(row, -3, False)
        for row in exits:
            # You might need to change set_fixed_obj() function if you change the reward for exit
            self.set_fixed_obj(row, 400, True)

        if self.is_render:
            self.update()

    def _init_grid(self):
        if self.is_render:
            self.title('moderate maze')
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

    def set_step_penalty(self, step_penalty):
        self.step_penalty = step_penalty

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

    def set_wall(self, position, reward, is_done):
        self.walls.append([position, reward, is_done])

        if self.is_render:
            # draw this object
            new_obj = self.origin_coord + np.array([self.pixel * position[0], self.pixel * position[1]])
            self.canvas.create_rectangle(
                new_obj[0] - 15, new_obj[1] - 15,
                new_obj[0] + 15, new_obj[1] + 15,
                fill='black')

    def set_fixed_obj(self, position, reward, is_done):
        self.object_list.append([position, reward, is_done])

        if self.is_render:
            # draw this object
            new_obj = self.origin_coord + np.array([self.pixel * position[0], self.pixel * position[1]])
            if reward < 0:
                self.canvas.create_oval(
                    new_obj[0] - 15, new_obj[1] - 15,
                    new_obj[0] + 15, new_obj[1] + 15,
                    fill='black')
            # Feel free to change this reward for exit, but remember doing it also here, otherwise there might be errors
            elif reward > 10:
                self.canvas.create_oval(
                    new_obj[0] - 15, new_obj[1] - 15,
                    new_obj[0] + 15, new_obj[1] + 15,
                    fill='green')
            elif reward > 0:
                self.canvas.create_oval(
                    new_obj[0] - 15, new_obj[1] - 15,
                    new_obj[0] + 15, new_obj[1] + 15,
                    fill='yellow')
            else:
                self.canvas.create_rectangle(
                    new_obj[0] - 15, new_obj[1] - 15,
                    new_obj[0] + 15, new_obj[1] + 15,
                    fill='black')

    def set_key_chest(self, key_position, reward_position, key, key_reward, chest_reward):
        self.key_chest_to_set.append([key_position, reward_position, key, key_reward, chest_reward])


    def set_collect_all_rewards(self, reward_position_list, reward, flag_name):
        for ind_pos in reward_position_list:
            self.set_fixed_obj(ind_pos, reward, flag_name)
        #self.ending_con_map[flag_name] = len(reward_position_list) # add new keys to dic

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
        self.agent_keys = []
        self.agent_chests = []
        self.key_list = []
        self.chest_list = []

        for [key_position, reward_position, key, key_reward, chest_reward] in self.key_chest_to_set:
            key_coordinates = -1
            chest_coordinates = -1
            if self.is_render:
                key_obj = self.origin_coord + np.array([self.pixel * key_position[0], self.pixel * key_position[1]])
                key_coordinates = self.canvas.create_text(
                    key_obj[0], key_obj[1],
                    fill='black', text="key")

                chest_obj = self.origin_coord + np.array(
                    [self.pixel * reward_position[0], self.pixel * reward_position[1]])
                chest_coordinates = self.canvas.create_text(
                    chest_obj[0], chest_obj[1],
                    fill='black', text="chest")

            self.key_list.append([key_position, key, key_reward, chest_reward, key_coordinates])
            self.chest_list.append([reward_position, key, 0, chest_coordinates])

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

        # return position of agent and has keys or not
        return np.array([self.agent[0], self.agent[1], len(self.agent_keys), len(self.agent_chests)])

    def taking_action(self, action):
        state = self.agent
        new_position = [0, 0]
        has_hit_border = True
        if action == 0:   # ^
            new_state = [state[0],  state[1]-1]
            if state[1] > 0 and self.check_no_wall(new_state):
                new_position[1] = -1
                has_hit_border = False
        elif action == 1:   # v
            new_state = [state[0],  state[1]+1]
            if state[1] < (self.grid_height - 1) and self.check_no_wall(new_state):
                new_position[1] = 1
                has_hit_border = False
        elif action == 2:   # <
            new_state = [state[0]-1,  state[1]]
            if state[0] > 0 and self.check_no_wall(new_state):
                new_position[0] = -1
                has_hit_border = False
        elif action == 3:   # >
            new_state = [state[0]+1,  state[1]]
            if state[0] < (self.grid_width - 1) and self.check_no_wall(new_state):
                new_position[0] = 1
                has_hit_border = False

        # Check if the agent collides with the wall, if so, it harshly dies
        collide_walls = [obj for obj in self.walls if (obj[0][0] == new_state[0] and obj[0][1] == new_state[1] and obj[1] == 0)]
        if len(collide_walls) > 0:
            new_state = np.array([self.agent[0], self.agent[1], len(self.agent_keys), len(self.agent_chests)])
            return new_state, -10, False

        if has_hit_border:
            new_state = np.array([self.agent[0], self.agent[1], len(self.agent_keys), len(self.agent_chests)])
            return new_state, self.step_penalty*2, False

        if self.is_render:
            self.canvas.move(self.agent_avatar, new_position[0] * self.pixel, new_position[1] * self.pixel)  # move agent

        self.agent[0] = self.agent[0] + new_position[0]  # next state
        self.agent[1] = self.agent[1] + new_position[1]  # next state

        # Check if it reaches a fixed object that is reachable
        reward = self.step_penalty
        outcomes = [obj for obj in self.object_list if obj[0][0] == self.agent[0] and obj[0][1] == self.agent[1]]
        if len(outcomes) > 0:
            obj = outcomes[0]
            reward = obj[1]
            is_done = obj[2]
            # if isinstance(is_done_content, str):
            #     idx_list = [idx for idx in range(len(self.object_list)) if self.object_list[idx][0][0] == self.agent[0]
            #      and self.object_list[idx][0][1] == self.agent[1]]
            #     for index in idx_list:
            #         del self.object_list[index]
            #     if is_done_content in self.agent_con_map:
            #         self.agent_con_map[is_done_content] = self.agent_con_map[is_done_content] + 1
            #
            #     else:
            #         self.agent_con_map[is_done_content] = 1
            #
            #     if self.ending_con_map[is_done_content] == self.agent_con_map[is_done_content]:
            #         is_done = True
            #     else:
            #         is_done = False
            # else:
            #    is_done = is_done_content
        else:
            is_done, reward = self.check_key_chest(self.agent)
            # is_done = False

            # if reward > 0:
            #     is_done = True
            # else:
            #     is_done = False

        if is_done:
            if self.is_render:
                print()
                # print("Reward: ", reward)
                for obj in self.key_list:
                    self.canvas.delete(obj[4])
                for obj in self.chest_list:
                    self.canvas.delete(obj[3])
        new_state = np.array([self.agent[0], self.agent[1], len(self.agent_keys), len(self.agent_chests)])
        return new_state, reward, is_done

    def check_no_wall(self, new_position):
        found_wall = [obj for obj in self.walls if obj[0][0] == new_position[0] and obj[0][1] == new_position[1]]
        if len(found_wall) > 0:
            return False
        else:
            return True

    def check_key_chest(self, new_position):
        is_done = False
        found_key = [obj for obj in self.key_list if obj[0][0] == new_position[0] and obj[0][1] == new_position[1]]
        checked_reward = self.step_penalty
        if len(found_key) > 0:
            for key_obj in found_key:
                if key_obj[1] not in self.agent_keys:
                    self.agent_keys.append(key_obj[1])
                    checked_reward = key_obj[2]
                    for obj in self.chest_list:
                        if obj[1] == key_obj[1]:
                            obj[2] = key_obj[3]
                    self.key_list.remove(key_obj)
                    if key_obj[4] > -1:
                        self.canvas.delete(key_obj[4])
        else:
            found_chest = [obj for obj in self.chest_list if obj[0][0] == new_position[0] and obj[0][1] == new_position[1]]
            if len(found_chest) > 0:
                for chest in found_chest:
                    chest_key = chest[1]
                    if chest_key in self.agent_keys:
                        if chest_key not in self.agent_chests:
                            self.agent_chests.append(chest_key)
                        checked_reward = chest[2]
                        self.chest_list.remove(chest)
                        if chest[3] > -1:
                            self.canvas.delete(chest[3])
                        is_done = True
        return is_done, checked_reward

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
