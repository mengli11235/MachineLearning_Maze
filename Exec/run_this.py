from MazeEnv.moderate_maze import MazeSimulator
from LearningAlgos.RL_brain import SarsaLambdaTable
import pandas as pd
import matplotlib.pyplot as plt
import time

def learning(epi, time_in_ms, _is_render, RL, env):
    step = 0
    rewards = []
    time_array = []
    epo = []

    for episode in range(epi):
        # initial observation
        observation = env.reset()
        reward_in_epoch = 0
        init_time = time.time()

        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        # initial all zero eligibility trace
        RL.eligibility_trace *= 0

        while True:
            # fresh env
            env.render(time_in_ms)

            # RL take action and get next observation and reward
            observation_, reward, done = env.taking_action(action)
            reward_in_epoch += reward

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                rewards.append(reward_in_epoch)
                time_array.append(format(time.time()-init_time, '.2f'))
                epo.append(episode+1)
                break
            step += 1

    # end of game
    print('game over, total rewards gained for each epoch:')
    print(rewards)
    print('time (in sec) spent over epochs:')
    print(time_array)
    RL.q_table.to_csv("temp_sarsa_table.csv", sep=',', encoding='utf-8')
    if is_render:
        env.destroy()

    plt.plot(epo, rewards)
    plt.show()


def running(epi, time_in_ms, _is_render, RL, env):
    try:
        df = pd.DataFrame.from_csv('temp_q_table.csv', sep=',', encoding='utf8')
        RL.set_prior_qtable(df)
        print("set prior q")
    except Exception:
        pass

    step = 0
    rewards = []
    time_array = []
    epo = []
    for episode in range(epi):
        # initial observation
        observation = env.reset()
        reward_in_epoch = 0
        init_time = time.time()

        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        # initial all zero eligibility trace
        RL.eligibility_trace *= 0

        while True:
            # fresh env
            env.render(time_in_ms)

            # RL take action and get next observation and reward
            observation_, reward, done = env.taking_action(action)
            reward_in_epoch += reward

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            # RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                rewards.append(reward_in_epoch)
                time_array.append(format(time.time()-init_time, '.2f'))
                epo.append(episode+1)
                break
            step += 1

    # end of game
    print('game over, total rewards gained for each epoch:')
    print(rewards)
    print('time (in sec) spent over epochs:')
    print(time_array)
    RL.q_table.to_csv("temp_sarsa_table.csv", sep=',', encoding='utf-8')
    if is_render:
        env.destroy()

    plt.plot(epo, time_array)
    plt.show()

if __name__ == "__main__":
    #env = MazeSimulator()
    # set if render the GUI
    is_render = False
    is_demo = False
    # set number of runs
    episodes = 300
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
    maze.set_key_chest([1, 0], [11, 15], 'key', 1500)
    maze.set_step_penalty(-1)

    # build the rendered maze
    maze.build_maze()

    # initiate SarsaLearner
    actions = list(range(maze.n_actions))
    learning_rate = 0.1
    reward_gamma = 0.95
    greedy = 0.7
    SLearner = SarsaLambdaTable(actions, 
                                learning_rate, 
                                reward_decay=reward_gamma, e_greedy=greedy, 
                                trace_decay=0.9
                                )

    if not is_demo:
        # run the simulation of training
        if is_render:
            maze.after(1, learning(episodes, interval, is_render, SLearner, maze))
            maze.mainloop()
        else:
            learning(episodes, interval, is_render, SLearner, maze)
    else:
        running(episodes, interval, is_render, SLearner, maze)
