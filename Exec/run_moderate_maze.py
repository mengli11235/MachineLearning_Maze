from MazeEnv.moderate_maze import MazeSimulator
from LearningAlgos.moderate_maze_RL import QLearningTable
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv


def learning(epi, time_in_ms, _is_render, QL, env):
    rewards = []
    time_array = []
    epo = []
    step_array = []

    for episode in range(epi):
        # initiate the agent
        agent, cond = env.reset()
        reward_in_each_epi = 0
        init_time = time.time()
        step = 0

        while True:
            # fresh env
            env.render(time_in_ms)

            # RL choose action based on observation
            action = QL.choose_action(agent, cond)

            # RL take action and get next observation and reward
            new_state, new_cond, reward, is_done = env.taking_action(action)
            reward_in_each_epi += reward

            # RL learn from this transition
            QL.learn(agent, cond, action, reward, new_state, new_cond, is_done)

            # swap observation
            agent = new_state
            cond = new_cond

            # count step
            step = step + 1

            # break while loop when end of this episode
            if is_done:
                rewards.append(reward_in_each_epi)
                time_array.append(format(time.time() - init_time, '.2f'))
                step_array.append(step)
                # print(time_array)
                epo.append(episode+1)
                if _is_render:
                    # print(episode/epi)
                    print(reward_in_each_epi)
                    print()
                    # print(epo)
                break

    # end of game
    print('game over')
    if _is_render:
        time.sleep(1)
        env.destroy()

    qtable_keys = QL.q_table_category.keys()
    with open('tmp_data/q_table_category.csv', 'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(qtable_keys)
    for key in qtable_keys:
        QL.q_table_category[key].to_csv("tmp_data/temp_q_table_" + key + ".csv", sep=',', encoding='utf-8')
        print(QL.q_table_category[key])
    plt.figure(1)
    plt.plot(epo, rewards)
    # plt.figure(2)
    # plt.plot(epo, step_array)
    # plt.figure(3)
    # plt.plot(epo, [r/s for r, s in zip(rewards, step_array)])
    plt.show()

    # plt.plot(epo, rewards)
    # plt.show()


def running(epi, time_in_ms, _is_render, QL, env):
    try:
        with open('tmp_data/q_table_category.csv', 'r') as f:
            reader = csv.reader(f)
            qtable_keys = list(reader)[0]
            for key in qtable_keys:
                df = pd.DataFrame.from_csv("tmp_data/temp_q_table_" + key + ".csv", sep=',', encoding='utf8')
                QL.set_prior_qtable(key, df)
            print("set prior q")
    except Exception:
        pass

    for episode in range(epi):
        # initiate the agent
        agent, cond = env.reset()
        reward_in_each_epi = 0

        while True:
            # fresh env
            env.render(time_in_ms)

            # RL choose action based on observation
            action = QL.choose_action(agent, cond)

            # RL take action and get next observation and reward
            new_state, new_cond, reward, is_done = env.taking_action(action)
            reward_in_each_epi += reward

            # swap observation
            agent = new_state
            cond = new_cond

            # break while loop when end of this episode
            if is_done:
                break
        if _is_render:
            print(reward_in_each_epi)

    # end of game
    print('game over')
    if _is_render:
        time.sleep(500)
        env.destroy()


if __name__ == "__main__":
    # set if render the GUI
    is_render = False
    is_demo = True

    # set number of runs
    episodes = 2100
    # animation interval
    interval = 0.005
    # set the size of maze: column x row
    size_maze = [20, 20]
    # initial position of the agent
    # all position count from 0
    init_pos = [0, 0]
    # init_pos = [10, 16]

    # initiate maze simulator for learning and running
    if is_demo:
        is_render = True
    maze = MazeSimulator(size_maze[1], size_maze[0], init_pos, is_render)
    # demo_maze = MazeSimulator(size_maze[1], size_maze[0], init_pos, True)
    maze.set_step_penalty(-1)

    # set fixed object ([column, row], reward, isFinishedWhenReach)
    # set rewards
    # maze.set_fixed_obj([3, 4], 1, True)
    # demo_maze.set_fixed_obj([3, 4], 1, True)
    maze.set_key_chest([19, 15], [0, 0], 'key', 0, 600)
    maze.set_key_chest([3, 3], [18, 15], 'key2', 0, 800)
    maze.set_key_chest([2, 14], [18, 4], 'key3', 0, 1000)
    # maze.set_fixed_obj([17, 15], 800, True)

    # maze.set_fixed_obj([1, 3], 1, True)
    # demo_maze.set_fixed_obj([1, 3], 1, True)
    # maze.set_collect_all_rewards([[3, 4], [1, 3]], 1, "golds")
    # demo_maze.set_collect_all_rewards([[3, 4], [1, 3]], 1, "golds")

    # set traps
    # maze.set_fixed_obj([1, 2], -1, True)
    # demo_maze.set_fixed_obj([1, 2], -1, True)
    # maze.set_fixed_obj([2, 1], -1, True)
    # demo_maze.set_fixed_obj([2, 1], -1, True)

    # build the rendered maze
    maze.build_maze()
    # demo_maze.build_maze()

    # initiate QLearner
    actions = list(range(maze.n_actions))
    learning_rate = 0.1
    reward_gamma = 0.975
    greedy = 0.4
    QLearner = QLearningTable(actions, learning_rate, reward_gamma, greedy)
    QLearner.set_greedy_rule([0.9], episodes, 0.9)

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
        demo_interval = 0.1
        QRunner = QLearningTable(actions, learning_rate, reward_gamma, demo_greedy)
        running(30, demo_interval, True, QRunner, maze)

