import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
import csv, time

from MazeEnv.maze_layouts import MazeSmall, MazeLarge, MazeMedium


def plot_csv(csv_name):
    fig, ax = plt.subplots()
    try:
        for name in csv_names:
            for i in [0, 1, 2]:
            # for i in [0]:
                with open(name + str(i) + ".csv", 'r') as f:
                    reader = csv.reader(f)
                    the_list = list(reader)[0]
                    the_list = [float(num) for num in the_list]
                    ax.plot(range(len(the_list)), the_list, label=name + "-" + str(i))
    except Exception:
        pass
    legend = ax.legend(loc='upper left', shadow=True)
    plt.show()


def plot_maze(maze_index):
    init_pos = [0, 0]
    is_render = True
    if maze_index == 0:
        maze = MazeSmall(init_pos).init_maze(is_render)
    elif maze_index == 1:
        maze = MazeMedium(init_pos).init_maze(is_render)
    elif maze_index == 2:
        maze = MazeLarge(init_pos).init_maze(is_render)
    maze.reset()
    while True:
        maze.render(0.05)
        time.sleep(0.05)


if __name__ == "__main__":
    # csv_names = ["q_lambda_avg_reward", "sarsa_lambda_avg_reward", "qv_lambda_avg_reward"]
    # csv_names = ["q_lambda_avg_reward", "sarsa_lambda_avg_reward", "qv_lambda_avg_reward"]
    # csv_names = ["medium_maze/q_lambda_avg_reward", "medium_maze/sarsa_lambda_avg_reward"]
    csv_names = ["small_maze/q_lambda_avg_reward", "small_maze/sarsa_lambda_avg_reward"]
    # plot_csv(csv_names)

    # plot_maze(0)
    plot_maze(1)
    # plot_maze(2)


