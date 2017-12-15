"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from MazeEnv.easy_maze import MazeSimulator
from LearningAlgos.easy_maze_RL import QLearningTable
import pandas as pd
import time


def learning(epi, time_in_ms, _is_render, QL, env):
    for episode in range(epi):
        # initiate the agent
        agent = env.reset()

        while True:
            # fresh env
            env.render(time_in_ms)

            # RL choose action based on observation
            current_state = str(agent)
            action = QL.choose_action(current_state)

            # RL take action and get next observation and reward
            new_state, reward, is_done = env.taking_action(action)

            # RL learn from this transition
            QL.learn(current_state, action, reward, str(new_state), is_done)

            # swap observation
            agent = new_state

            # break while loop when end of this episode
            if is_done:
                break

    # end of game
    print('game over')
    if _is_render:
        time.sleep(1)
        env.destroy()

    QL.q_table.to_csv("temp_q_table.csv", sep=',', encoding='utf-8')
    print(QL.q_table)


def running(epi, time_in_ms, _is_render, QL, env):
    try:
        df = pd.DataFrame.from_csv('temp_q_table.csv', sep=',', encoding='utf8')
        QL.set_prior_qtable(df)
        print("set prior q")
    except Exception:
        pass

    for episode in range(epi):
        # initiate the agent
        agent = env.reset()

        while True:
            # fresh env
            env.render(time_in_ms)

            # RL choose action based on observation
            action = QL.choose_action(str(agent))

            # RL take action and get next observation and reward
            new_state, reward, is_done = env.taking_action(action)

            # swap observation
            agent = new_state

            # break while loop when end of this episode
            if is_done:
                break

    # end of game
    print('game over')
    if _is_render:
        time.sleep(1)
        env.destroy()


if __name__ == "__main__":
    # set if render the GUI
    is_render = False
    # set number of runs
    episodes = 30000
    # animation interval
    interval = 0.005
    # set the size of maze: column x row
    size_maze = [4, 6]
    # initial position of the agent
    # all position count from 0
    init_pos = [0, 0]

    # initiate maze simulator
    maze = MazeSimulator(size_maze[1], size_maze[0], init_pos, is_render)

    # set fixed object ([column, row], reward, isFinishedWhenReach)
    maze.set_fixed_obj([3, 3], 1, True)
    maze.set_fixed_obj([1, 2], -1, True)
    maze.set_fixed_obj([2, 1], -1, True)
    maze.set_fixed_obj([2, 4], 1, True)

    # build the rendered maze
    maze.build_maze()

    # initiate QLearner
    actions = list(range(maze.n_actions))
    learning_rate = 0.01
    reward_gamma = 0.9
    greedy = 0.9
    QLearner = QLearningTable(actions, learning_rate, reward_gamma, greedy)

    # run the simulation of training
    if is_render:
        maze.after(1, learning(episodes, interval, is_render, QLearner, maze))
        maze.mainloop()
    else:
        learning(episodes, interval, is_render, QLearner, maze)

    #running(50, interval, is_render, QLearner, maze)

