from Exec.for_output_run.QLambda import QLambda
from Exec.for_output_run.SarsaLambda import _SarsaLambda
from Exec.for_output_run.QVLambda import QVLambda
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
import time
import csv


def single_run(func, lmb, maze_index, total_epi, episodes, algo_type, simulation, max_steps, reward_gamma, greedy_rule, max_reward_coefficient):
    print("start", algo_type)
    min_steps = 999999
    f_avg_rw_arr = []
    f_avg_rw = []
    for ix in range(simulation):
        p = func(lmb, maze_index, total_epi, episodes, max_steps, reward_gamma, greedy_rule, max_reward_coefficient)
        f_avg_rw_arr.append(p)
        if min_steps > len(p):
            min_steps = len(p)
    for idx in range(min_steps):
        sum_val = 0
        length = len(f_avg_rw_arr)
        for ix in range(length):
            sum_val += f_avg_rw_arr[ix][idx]
        f_avg_rw.append(sum_val / length)

    with open('tmp_data/'+ algo_type + str(index) + '.csv',
              'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(f_avg_rw)


if __name__ == "__main__":
    algo_names = ["q_lambda_avg_reward", "sarsa_lambda_avg_reward", "qv_lambda_avg_reward"]
    lambda_arr = [0]
    # lambda_arr = [0, 0.5, 0.8]
    f1 = QLambda().run
    f2 = _SarsaLambda().run
    f3 = QVLambda().run

    simulation = 1
    maze_index = 0

    # base parameter values
    total_epi = 15000
    episodes = 250
    max_steps = 500
    reward_gamma = 0.95
    greedy_rule = [0.4, [0.9], 0.9]
    max_reward_coefficient = 0.8

    para_names = ["total_epi", "episodes", "max_steps", "reward_gamma", "greedy_rule", "max_reward_coefficient"]
    parameter_dict = {
        algo_names[0]: [{
            para_names[0]: total_epi,
            para_names[1]: 300,
            para_names[2]: 500,
            para_names[3]: 0.95,
            para_names[4]: greedy_rule,
            para_names[5]: 0.8
        }],
        algo_names[1]: [{
            para_names[0]: total_epi,
            para_names[1]: 300,
            para_names[2]: 500,
            para_names[3]: 0.95,
            para_names[4]: greedy_rule,
            para_names[5]: 0.8
        }],
        algo_names[2]: [{
            para_names[0]: total_epi,
            para_names[1]: 180,
            para_names[2]: 500,
            para_names[3]: 0.95,
            para_names[4]: [0.6, [0.9], 0.9],
            para_names[5]: 0.8
        }]
    }

    for index in range(len(lambda_arr)):
        lmb = lambda_arr[index]

        _parameters = parameter_dict[algo_names[0]][index]
        single_run(f1, lmb, maze_index, _parameters[para_names[0]], _parameters[para_names[1]], algo_names[0],
                   simulation, _parameters[para_names[2]], _parameters[para_names[3]], _parameters[para_names[4]],
                   _parameters[para_names[5]])

        _parameters = parameter_dict[algo_names[1]][index]
        single_run(f2, lmb, maze_index, _parameters[para_names[0]], _parameters[para_names[1]], algo_names[1],
                   simulation, _parameters[para_names[2]], _parameters[para_names[3]], _parameters[para_names[4]],
                   _parameters[para_names[5]])

        _parameters = parameter_dict[algo_names[2]][index]
        single_run(f3, lmb, maze_index, _parameters[para_names[0]], _parameters[para_names[1]], algo_names[2],
                   simulation, _parameters[para_names[2]], _parameters[para_names[3]], _parameters[para_names[4]],
                   _parameters[para_names[5]])

        # print("start Q lambda")
        # min_steps = 999999
        # f1_avg_rw_arr = []
        # f1_avg_rw = []
        # for ix in range(simulation):
        #     p = f1(lmb, maze_index, total_epi, steps_epoch)
        #     f1_avg_rw_arr.append(p)
        #     if min_steps > len(p):
        #         min_steps = len(p)
        # for idx in range(min_steps):
        #     sum_val = 0
        #     length = len(f1_avg_rw_arr)
        #     for ix in range(length):
        #         sum_val += f1_avg_rw_arr[ix][idx]
        #     f1_avg_rw.append(sum_val/length)
        #
        # with open('tmp_data/q_lambda_avg_reward' + str(index) + '.csv', 'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        #     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        #     wr.writerow(f1_avg_rw)
        #
        # print("start SARSA lambda")
        # min_steps = 999999
        # f2_avg_rw_arr = []
        # f2_avg_rw = []
        # for ix in range(simulation):
        #     p = f2(lmb, maze_index, total_epi, steps_epoch)
        #     f2_avg_rw_arr.append(p)
        #     if min_steps > len(p):
        #         min_steps = len(p)
        # for idx in range(min_steps):
        #     sum_val = 0
        #     length = len(f2_avg_rw_arr)
        #     for ix in range(length):
        #         sum_val += f2_avg_rw_arr[ix][idx]
        #     f2_avg_rw.append(sum_val / length)
        #
        # with open('tmp_data/sarsa_lambda_avg_reward' + str(index) + '.csv',
        #           'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        #     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        #     wr.writerow(f2_avg_rw)

        # print("start QV lambda")
        # min_steps = 999999
        # f3_avg_rw_arr = []
        # f3_avg_rw = []
        # for ix in range(simulation):
        #     p = f3(lmb, maze_index, total_epi, steps_epoch)
        #     f3_avg_rw_arr.append(p)
        #     if min_steps > len(p):
        #         min_steps = len(p)
        # for idx in range(min_steps):
        #     sum_val = 0
        #     length = len(f3_avg_rw_arr)
        #     for ix in range(length):
        #         sum_val += f3_avg_rw_arr[ix][idx]
        #     f3_avg_rw.append(sum_val / length)
        #
        # with open('tmp_data/qv_lambda_avg_reward' + str(index) + '.csv',
        #           'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        #     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        #     wr.writerow(f3_avg_rw)

        print("finish lambda", str(lmb))
        # plt.plot(range(len(f1_avg_rw)), f1_avg_rw)
        # plt.plot(range(len(f2_avg_rw)), f2_avg_rw)
        # plt.plot(range(len(f3_avg_rw)), f3_avg_rw)
        # plt.show()
        #
        # for ix in range(simulation):
        #     f2(lmb)
        #
        # for ix in range(simulation):
        #     f3(lmb)

