from MazeEnv.maze_env import MazeSimulator
from LearningAlgos.QV_lambda_RL import QTable, VTable
import matplotlib.pyplot as plt
import time


def learning(epi, max_steps, time_in_ms, _is_render, QL, VL, env):
    rewards = []
    time_array = []
    epo = []

    for episode in range(epi):
        # initiate the agent
        agent = env.reset()
        reward_in_each_epi = 0
        init_time = time.time()

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
            VL.update(current_state, reward, str(new_state), step, is_done)

            # swap observation
            agent = new_state

            # break while loop when end of this episode
            if is_done:
                # print(episode/epi)
                rewards.append(reward_in_each_epi)
                # print(rewards)
                time_array.append(format(time.time() - init_time, '.2f'))
                # print(time_array)
                epo.append(episode+1)
                # print(epo)
                break


    # end of game
    print('game over')
    if _is_render:
        time.sleep(1)
        env.destroy()

    QL.q_table.to_csv("temp_q_table.csv", sep=',', encoding='utf-8')
    print(QL.q_table)
    plt.plot(epo, rewards)
    plt.show()


if __name__ == "__main__":
    # set if render the GUI
    is_render = False
    is_demo = False
    # set number of runs
    episodes = 100
    # animation interval
    interval = 0.005
    # set the size of maze: column x row
    size_maze = [20, 20]
    # initial position of the agent
    # all position count from 0
    init_pos = [0, 0]

    # maximal number of states
    max_steps = 1000

    # initiate maze simulator for learning and running
    if is_demo:
        is_render = True
    maze = MazeSimulator(size_maze[1], size_maze[0], init_pos, is_render)
    # demo_maze = MazeSimulator(size_maze[1], size_maze[0], init_pos, True)

    # set fixed object ([column, row], reward, isFinishedWhenReach)
    # set rewards
    # maze.set_fixed_obj([3, 4], 1, True)
    # demo_maze.set_fixed_obj([3, 4], 1, True)
    maze.set_key_chest([1, 0], [11, 15], 'key', 3)

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
    learning_rate_v = 0.1
    learning_rate_q = 0.1
    reward_gamma = 0.95
    greedy = 0.85
    lambda_v = 0.6
    QLearner = QTable(actions, learning_rate_q, reward_gamma, greedy)
    Vlearner = VTable(max_steps, learning_rate_v, reward_gamma, lambda_v)

    # run the simulation of training
    if is_render:
        maze.after(1, learning(episodes, max_steps, interval, is_render, QLearner, Vlearner, maze))
        maze.mainloop()
    else:
        learning(episodes, max_steps, interval, is_render, QLearner, Vlearner, maze)

