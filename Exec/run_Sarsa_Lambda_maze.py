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


def learning(total_steps, time_in_ms, _is_render, SL, env, max_steps):
    episode = 0
    step_counter = 0
    rewards = []
    time_array = []
    epo = []
    step_array = []
    training_time = time.time()
    per_5 = math.floor(total_steps/20)
    totalStep = 0

    while True:
        force_exit = False
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

        for step in range(max_steps):
            step_counter = step_counter + 1
            if step_counter % per_5 == 0:
                print("{} %".format((step_counter / per_5) * 5))
                print()

            # fresh env
            env.render(time_in_ms)

            # SL take action and get next observation and reward
            new_state, new_cond, reward, is_done = env.taking_action(action)
            reward_in_epoch += reward

            # SL choose action based on next observation
            action_ = SL.choose_action(new_state, new_cond)

            if step == max_steps - 1:
                force_exit = True
            # SL learn from this transition (s, a, r, s, a) ==> Sarsa
            SL.learn(agent, action, reward, new_state, action_, cond, new_cond, is_done, force_exit)

            # swap observation and action
            agent = new_state
            cond = new_cond
            action = action_

            # count step
            totalStep = totalStep + 1

            # break while loop when end of this episode
            if is_done or step_counter >= total_steps:
                break

        episode = episode + 1
        rewards.append(reward_in_epoch)
        time_array.append(format(time.time() - init_time, '.2f'))
        step_array.append(step)
        epo.append(episode + 1)

        if step_counter >= total_steps:
            # print(epo)
            break

    # end of game
    print('game over', totalStep)
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
        # print(SL.q_table_category[key])

    axes = plt.gca()
    axes.set_ylim([-1000, 1000])
    plt.figure(1)
    plt.plot(epo[:-1], rewards[:-1])
    plt.ylabel("rewards")
    plt.xlabel("epoches")
    plt.title("small maze rewards: Sarsa_lambda(lambda=0.5)")

    plt.figure(2)
    plt.plot(epo[:-1], step_array[:-1])
    plt.ylabel("steps")
    plt.xlabel("epoches")
    plt.title("small maze steps: Sarsa_lambda(lambda=0.5)")
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
    # episodes = 1200
    # set number of total steps
    total_steps = 5000  # 60000 for medium, 5000 for simple
    # animation interval
    interval = 0.005
    # maximal number of states
    max_steps = 150  # 1000 for medium, 150 for simple

    # initial position of the agent
    # all position count from 0
    init_pos = [0, 0]

    # initiate maze simulator for learning and running
    if is_demo:
        is_render = True

    maze = MazeSmall(init_pos).init_maze(is_render)
    # maze = MazeMedium(init_pos).init_maze(is_render)
    # maze = MazeLarge(init_pos).init_maze(is_render)

    # initiate SarsaLearner
    actions = list(range(maze.n_actions))
    learning_rate = 0.1
    reward_gamma = 0.95
    greedy = 0.9

    # lambda_val = 0
    lambda_val = 0.5
    max_reward_coefficient = 0.75
    SLearner = SarsaLambda(actions, learning_rate, reward_gamma, greedy, lambda_val, max_reward_coefficient)
    SLearner.set_greedy_rule([0.9], 50, greedy)

    # run the training
    if not is_demo:
        if is_render:
            maze.after(1, learning(total_steps, interval, is_render, SLearner, maze, max_steps))
            maze.mainloop()
        else:
            learning(total_steps, interval, is_render, SLearner, maze, max_steps)
    # run the simulation of result
    else:
        # Q decision with 99% greedy strategy
        demo_greedy = 0.99
        demo_interval = 0.05
        SRunner = SarsaLambda(actions, learning_rate, reward_gamma, demo_greedy, max_reward_coefficient)
        running(30, demo_interval, True, SRunner, maze)
