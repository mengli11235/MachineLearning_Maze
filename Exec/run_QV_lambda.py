from MazeEnv.maze_layouts_qv import MazeSmallQV, MazeLargeQV, MazeMediumQV
from LearningAlgos.QV_lambda_RL import QTable, VTable
import matplotlib.pyplot as plt
import time, math


def learning(epi, max_steps, time_in_ms, _is_render, QL, VL, env):
    rewards = []
    time_array = []
    epo = []
    step_array = []
    per_5 = math.floor(epi / 20)

    for episode in range(epi):
        # initiate the agent
        agent = env.reset()
        reward_in_each_epi = 0
        init_time = time.time()

        if episode%per_5 == 0:
            print("{} %".format((episode/per_5)*5))
            print()

        for step in range(max_steps):
            # fresh env
            env.render(time_in_ms)

            # RL choose action based on observation
            current_state = str(agent)
            action = QL.choose_action(current_state)

            # RL take action and get next observation and reward
            new_state, reward, is_done = env.taking_action(action)
            reward_in_each_epi += reward

            # RL learn from this transition
            QL.learn(VL, current_state, action, reward, str(new_state), is_done)
            VL.update(current_state, reward, str(new_state), is_done)

            # swap observation
            agent = new_state

            # break while loop when end of this episode
            if is_done:
                # print(epo)
                break
        # print(episode/epi)
        rewards.append(reward_in_each_epi)
        # print(rewards)
        time_array.append(format(time.time() - init_time, '.2f'))
        # print(time_array)
        epo.append(episode + 1)
        step_array.append(step)

    # end of game
    print('game over')
    if _is_render:
        time.sleep(1)
        env.destroy()

    QL.q_table.to_csv("temp_qv_table.csv", sep=',', encoding='utf-8')
    plt.figure(1)
    plt.plot(epo, rewards)
    plt.figure(2)
    plt.plot(epo, step_array)
    plt.show()


if __name__ == "__main__":
    # set if render the GUI
    is_render = False
    is_demo = False
    # set number of runs
    episodes = 600
    # animation interval
    interval = 0.005

    # initial position of the agent
    # all position count from 0
    init_pos = [0, 0]

    # maximal number of states
    max_steps = 400

    # initiate maze simulator for learning and running
    if is_demo:
        is_render = True

    # maze = MazeSmallQV(init_pos).init_maze(is_render)
    maze = MazeMediumQV(init_pos).init_maze(is_render)
    # maze = MazeLargeQV(init_pos).init_maze(is_render)

    # initiate QLearner
    actions = list(range(maze.n_actions))
    learning_rate_v = 0.1
    learning_rate_q = 0.1
    reward_gamma = 0.95
    greedy = 0.6
    lambda_v = 0.5
    QLearner = QTable(actions, learning_rate_q, reward_gamma, greedy)
    Vlearner = VTable(learning_rate_v, reward_gamma, lambda_v)

    # run the simulation of training
    if is_render:
        maze.after(1, learning(episodes, max_steps, interval, is_render, QLearner, Vlearner, maze))
        maze.mainloop()
    else:
        learning(episodes, max_steps, interval, is_render, QLearner, Vlearner, maze)

