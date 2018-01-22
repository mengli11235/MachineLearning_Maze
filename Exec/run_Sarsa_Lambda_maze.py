from MazeEnv.maze_layouts import MazeSmall, MazeLarge, MazeMedium
from LearningAlgos.Sarsa_lambda_RL import SarsaLambda
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
import time
import csv
import math


def learning(epi, time_in_ms, _is_render, SL, env):
    rewards = []
    time_array = []
    epo = []
    step_array = []
    training_time = time.time()
    per_5 = math.floor(epi/20)

    for episode in range(epi):
        # initial observation
        # observation = env.reset()
        reward_in_epoch = 0
        init_time = time.time()
        step = 0

        if episode%per_5 == 0:
            print("{} %".format((episode/per_5)*5))
            print()

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

            # count step
            step = step + 1

            # break while loop when end of this episode
            if is_done:
                rewards.append(reward_in_epoch)
                time_array.append(format(time.time()-init_time, '.2f'))
                step_array.append(step)
                epo.append(episode + 1)
                break

    # end of game
    print('game over')
    # print training time
    training_time = time.time() - training_time
    m, s = divmod(training_time, 60)
    h, m = divmod(m, 60)
    print("Total training time: %d hr %02d min %02d sec" % (h, m, s))

    sarsa_keys = SL.q_table_category.keys()
    with open('tmp_data/sarsa_category.csv', 'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(sarsa_keys)
    for key in sarsa_keys:
        SL.q_table_category[key].to_csv("tmp_data/temp_sarsa_table_" + key + ".csv", sep=',', encoding='utf-8')
        print(SL.q_table_category[key])

    plt.figure(1)
    plt.plot(epo, rewards)
    plt.figure(2)
    plt.plot(epo, step_array)
    plt.show()

    if _is_render:
        time.sleep(1)
        env.destroy()


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
    is_demo = False
    # set number of runs
    episodes = 1200
    # animation interval
    interval = 0.005

    # initial position of the agent
    # all position count from 0
    init_pos = [0, 0]
    
    # initiate maze simulator for learning and running
    if is_demo:
        is_render = True

    # maze = MazeSmall(init_pos).init_maze(is_render)
    # maze = MazeMedium(init_pos).init_maze(is_render)
    maze = MazeLarge(init_pos).init_maze(is_render)

    # initiate SarsaLearner
    actions = list(range(maze.n_actions))
    learning_rate = 0.1
    reward_gamma = 0.95

    greedy = 0.4
    lambda_val = 0
    # lambda_val = 0.5
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
