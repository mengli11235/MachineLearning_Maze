from MazeEnv.moderate_maze import MazeSimulator
from LearningAlgos.DQN_maze_RL import DeepQNetwork
import time


def learning(epi, time_in_ms, _is_render, QL, env):
    step = 0
    rewards = []
    time_array = []
    for episode in range(epi):
        # initial observation
        observation = env.reset()
        reward_in_epoch = 0;
        init_time = time.time()
        while True:
            # fresh env
            env.render(time_in_ms)
            # QL choose action based on observation
            action = QL.choose_action(observation)

            # QL take action and get next observation and reward
            observation_, reward, done = env.taking_action(action)
            reward_in_epoch += reward

            QL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                QL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                rewards.append(reward_in_epoch)
                time_array.append(format(time.time()-init_time, '.2f'))
                break
            step += 1

    # end of game
    print('game over, total rewards gained for each epoch:')
    print(rewards)
    print('time (in sec) spent over epochs:')
    print(time_array)
    env.destroy()
    # QL.


if __name__ == "__main__":
    # # maze game
    # env = MazeSimulator()
    # RL = DeepQNetwork(env.n_actions, env.n_features,
    #                   learning_rate=0.01,
    #                   reward_decay=0.9,
    #                   e_greedy=0.9,
    #                   replace_target_iter=200,
    #                   memory_size=2000,
    #                   # output_graph=True
    #                   )
    # env.after(100, run_maze)
    # env.mainloop()
    # RL.plot_cost()

    # set if render the GUI
    is_render = True
    # set number of runs
    episodes = 10
    # animation interval
    interval = 0.005
    # set the size of maze: column x row
    size_maze = [20, 20]
    # initial position of the agent
    # all position count from 0
    init_pos = [0, 0]

    # initiate maze simulator for learning and running
    maze = MazeSimulator(size_maze[1], size_maze[0], init_pos, is_render)
    # demo_maze = MazeSimulator(size_maze[1], size_maze[0], init_pos, True)

    # set fixed object ([column, row], reward, isFinishedWhenReach)
    # set rewards
    # maze.set_fixed_obj([3, 4], 1, True)
    # demo_maze.set_fixed_obj([3, 4], 1, True)
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
    reward_gamma = 0.8
    greedy = 0.9
    QLearner = DeepQNetwork(4, 2,
                                        learning_rate,
                                        reward_decay=reward_gamma,
                                        e_greedy=greedy,
                                        replace_target_iter=200,
                                        memory_size=2000,
                                        # output_graph=True
                                        )

    # run the simulation of training
    if is_render:
        maze.after(1, learning(episodes, interval, is_render, QLearner, maze))
        maze.mainloop()
    else:
        learning(episodes, interval, is_render, QLearner, maze)

    # Q decision with 99% greedy strategy
    # demo_greedy = 1
    # demo_interval = 0.01
    # QRunner = QLearningTable(actions, learning_rate, reward_gamma, demo_greedy)
    # running(50, demo_interval, True, QRunner, demo_maze)