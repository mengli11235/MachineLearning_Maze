from MazeEnv.moderate_maze import MazeSimulator
from LearningAlgos.Sarsa_lambda_RL import SarsaLambda
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
import time
import csv


def learning(epi, time_in_ms, _is_render, SL, env):
    rewards = []
    time_array = []
    epo = []

    for episode in range(epi):
        # initial observation
        # observation = env.reset()
        reward_in_epoch = 0
        init_time = time.time()

        # initial observation
        agent, cond = env.reset()

        # SL choose action based on observation
        action = SL.choose_action(agent, cond)

        # initial all zero eligibility trace
        SL.reset_trace()

        while True:
            # fresh env
            env.render(time_in_ms)

            # SL take action and get next observation and reward
            new_state, new_cond, reward, is_done = env.taking_action(action)
            reward_in_epoch += reward

            # SL choose action based on next observation
            action_ = SL.choose_action(new_state, new_cond)

            # SL learn from this transition (s, a, r, s, a) ==> Sarsa
            SL.learn(agent, action, reward, new_state, action_, cond, new_cond, is_done)

            # swap observation and action
            agent = new_state
            cond = new_cond
            action = action_

            # break while loop when end of this episode
            if is_done:
                rewards.append(reward_in_epoch)
                time_array.append(format(time.time()-init_time, '.2f'))
                epo.append(episode + 1)
                break

    # end of game
    print('game over, total rewards gained for each epoch:')
    print(rewards)
    print('time (in sec) spent over epochs:')
    print(time_array)

    sarsa_keys = SL.q_table_category.keys()
    with open('tmp_data/sarsa_category.csv', 'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(sarsa_keys)
    for key in sarsa_keys:
        SL.q_table_category[key].to_csv("tmp_data/temp_sarsa_table_" + key + ".csv", sep=',', encoding='utf-8')
        print(SL.q_table_category[key])

    if is_render:
        env.destroy()

    plt.plot(epo, rewards)
    plt.show()


def running(epi, time_in_ms, _is_render, SL, env):
    try:
        with open('tmp_data/sarsa_category.csv', 'r') as f:
            reader = csv.reader(f)
            qtable_keys = list(reader)[0]
            for key in qtable_keys:
                df = pd.DataFrame.from_csv("tmp_data/temp_sarsa_table_" + key + ".csv", sep=',', encoding='utf8')
                SL.set_prior_qtable(key, df)
            print("set prior q")
    except Exception:
        pass

    rewards = []
    time_array = []
    epo = []
    for episode in range(epi):
        reward_in_each_epi = 0
        init_time = time.time()

        # initial observation
        agent, cond = env.reset()

        # SL choose action based on observation
        action = SL.choose_action(agent, cond)

        # initial all zero eligibility trace
        SL.reset_trace()

        while True:
            # fresh env
            env.render(time_in_ms)

            # SL take action and get next observation and reward
            new_state, new_cond, reward, is_done = env.taking_action(action)
            reward_in_each_epi += reward

            # SL choose action based on next observation
            action_ = SL.choose_action(new_state, new_cond)

            # swap observation and action
            # agent = new_state
            # cond = new_cond
            action = action_

            # break while loop when end of this episode
            if is_done:
                rewards.append(reward_in_each_epi)
                time_array.append(format(time.time()-init_time, '.2f'))
                epo.append(episode+1)
                break

        if _is_render:
            print(reward_in_each_epi)

    # end of game
    print('game over, total rewards gained for each epoch:')
    print(rewards)
    print('time (in sec) spent over epochs:')
    print(time_array)

    if is_render:
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
    
    # initiate maze simulator for learning and running
    if is_demo:
        is_render = True
    maze = MazeSimulator(size_maze[1], size_maze[0], init_pos, is_render)

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

    # initiate SarsaLearner
    actions = list(range(maze.n_actions))
    learning_rate = 0.1
    reward_gamma = 0.95
    greedy = 0.4
    lambda_val = 0.6
    max_reward_coefficient = 0.75
    SLearner = SarsaLambda(actions, learning_rate, reward_gamma, greedy, lambda_val, max_reward_coefficient)
    SLearner.set_greedy_rule([0.9], episodes*0.9, 0.95)

    # run the training
    if not is_demo:
        if is_render:
            maze.after(1, learning(episodes, interval, is_render, SLearner, maze))
            maze.mainloop()
        else:
            learning(episodes, interval, is_render, SLearner, maze)
    # run the simulation of result
    else:
        # Q decision with 99% greedy strategy
        demo_greedy = 0.99
        demo_interval = 0.05
        SRunner = SarsaLambda(actions, learning_rate, reward_gamma, demo_greedy, max_reward_coefficient)
        running(30, demo_interval, True, SRunner, maze)
