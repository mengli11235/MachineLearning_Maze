from MazeEnv.moderate_maze import MazeSimulator
from LearningAlgos.QLearning_RL import QLearningTable
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
import time


def learning(epi, time_in_ms, _is_render, QL, env):
    rewards = []
    time_array = []
    epo = []

    for episode in range(epi):
        # initiate the agent
        agent = env.reset()
        reward_in_each_epi = 0
        init_time = time.time()

        while True:
            # fresh env
            env.render(time_in_ms)

            # RL choose action based on observation
            action = QL.choose_action(agent)

            # RL take action and get next observation and reward
            new_state, reward, is_done = env.taking_action(action)
            reward_in_each_epi += reward

            # RL learn from this transition
            QL.learn(agent, action, reward, new_state, is_done)

            # swap observation
            agent = new_state

            # break while loop when end of this episode
            if is_done:
                rewards.append(reward_in_each_epi)
                time_array.append(format(time.time() - init_time, '.2f'))
                # print(time_array)
                epo.append(episode+1)
                if _is_render:
                    print(episode/epi)
                    print(rewards)
                    print(epo)
                break

    # end of game
    print('game over')
    if _is_render:
        time.sleep(1)
        env.destroy()

    QL.q_table.to_csv("temp_q_table.csv", sep=',', encoding='utf-8')
    print(QL.q_table)

    plt.plot(epo, rewards)
    plt.title('rewards in each epoch')
    plt.show()


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
        reward_in_each_epi = 0

        while True:
            # fresh env
            env.render(time_in_ms)

            # RL choose action based on observation
            action = QL.choose_action(str(agent))

            # RL take action and get next observation and reward
            new_state, reward, is_done = env.taking_action(action)
            reward_in_each_epi += reward

            # swap observation
            agent = new_state

            # break while loop when end of this episode
            if is_done:
                break

        # if _is_render:
            # print(reward_in_each_epi)

    # end of game)
    print('game over')
    if _is_render:
        time.sleep(1)
        env.destroy()


if __name__ == "__main__":
    # set if render the GUI
    is_render = False
    is_demo = True
    # set number of runs
    episodes = 1200

    # animation interval
    interval = 0.005
    # set the size of maze: column x row
    size_maze = [20, 20]
    # initial position of the agent
    # all position count from 0
    init_pos = [0, 0]

    # initiate maze simulator for learning and running
    if is_demo:
        is_render = True
    maze = MazeSimulator(size_maze[1], size_maze[0], init_pos, is_render)

    maze.set_step_penalty(-1)

    maze.set_key_chest([10, 16], [9, 4], 'key', 800, 1200)

    # build the rendered maze
    maze.build_maze()

    # initiate QLearner
    actions = list(range(maze.n_actions))
    learning_rate = 0.1
    reward_gamma = 0.95
    greedy = 0.7
    QLearner = QLearningTable(actions, learning_rate, reward_gamma, greedy)
    QLearner.set_greedy_rule(20, 0.9)

    # run the simulation of training
    if not is_demo:
        if is_render:
            maze.after(1, learning(episodes, interval, is_render, QLearner, maze))
            maze.mainloop()
        else:
            learning(episodes, interval, is_render, QLearner, maze)
    else:
        # Q decision with 99% greedy strategy
        demo_greedy = 0.99
        demo_interval = 0.05
        QRunner = QLearningTable(actions, learning_rate, reward_gamma, demo_greedy)
        running(30, demo_interval, True, QRunner, maze)

