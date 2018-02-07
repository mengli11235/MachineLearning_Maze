from Exec.for_output_run.QLambda import QLambda
from Exec.for_output_run.SarsaLambda import _SarsaLambda
from Exec.for_output_run.QVLambda import QVLambda
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
import numpy as np
import time
import csv


def single_run(func, lmb, maze_index, total_epi, episodes, algo_type, simulation, max_steps, reward_gamma, greedy_rule, max_reward_coefficient):
    print("start", algo_type)
    min_steps = 999999
    max_step = 0
    f_avg_rw_arr = []
    f_avg_rw = []
    for ix in range(simulation):
        p = func(lmb, maze_index, total_epi, episodes, max_steps, reward_gamma, greedy_rule, max_reward_coefficient)
        f_avg_rw_arr.append(p)
        if min_steps > len(p):
            min_steps = len(p)
        if max_step < len(p):
            max_step = len(p)
    for idx in range(max_step):
        sum_val = 0
        sum_count = 0
        length = len(f_avg_rw_arr)
        for ix in range(length):
            if len(f_avg_rw_arr[ix]) >= idx + 1:
                sum_val += f_avg_rw_arr[ix][idx]
                sum_count = sum_count + 1
        f_avg_rw.append(sum_val / sum_count)

    with open('tmp_data/'+ algo_type + str(index) + '.csv',
              'w') as f:  # Just use 'w' mode in 3.x, otherwise 'wb'
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(f_avg_rw)


if __name__ == "__main__":
    algo_names = ["q_lambda_avg_reward", "sarsa_lambda_avg_reward", "qv_lambda_avg_reward"]
    # lambda_arr = [0.8]
    lambda_arr = [0, 0.5, 0.8]
    f1 = QLambda().run
    f2 = _SarsaLambda().run
    f3 = QVLambda().run

    # base parameter values
    total_epi = 30000
    episodes = 300
    max_steps = 300

    reward_gamma = 0.95
    greedy_rule = [0.6, [0.9], 0.7]
    max_reward_coefficient = 0.8

    para_names = ["total_epi", "episodes", "max_steps", "reward_gamma", "greedy_rule", "max_reward_coefficient"]
    simulation = 30
    maze_index = 1

    # "total_epi"
    # "episodes"
    # "max_steps"
    # "reward_gamma"
    # "greedy_rule"
    # "max_reward_coefficient"
    parameter_dict = {

        # algo_names[0]: [{
        #     para_names[0]: total_epi,
        #     para_names[1]: 600,
        #     para_names[2]: 400,
        #     para_names[3]: 0.95,
        #     para_names[4]: [0.4, [0.9], 0.99],
        #     para_names[5]: 0.8
        # }],
        # algo_names[1]: [{
        #     para_names[0]: total_epi,
        #     para_names[1]: 400,
        #     para_names[2]: 300,
        #     para_names[3]: 0.95,
        #     para_names[4]: [0.4, [0.95], 0.85],
        #     para_names[5]: 0.9
        # }],
        # algo_names[2]: [{
        #     para_names[0]: total_epi,
        #     para_names[1]: 400,
        #     para_names[2]: 1000,
        #     para_names[3]: 0.95,
        #     para_names[4]: [0.6, [0.9], 0.9],
        #     para_names[5]: 0.8
        # }]

        algo_names[0]: [
            {
                para_names[0]: total_epi,
                para_names[1]: episodes,
                para_names[2]: max_steps,
                para_names[3]: 0.95,
                para_names[4]: greedy_rule,
                para_names[5]: max_reward_coefficient
            },
            {
                para_names[0]: total_epi,
                para_names[1]: episodes,
                para_names[2]: max_steps,
                para_names[3]: 0.85,
                para_names[4]: greedy_rule,
                para_names[5]: max_reward_coefficient
            },
            {
                para_names[0]: total_epi,
                para_names[1]: episodes,
                para_names[2]: max_steps,
                para_names[3]: 0.8,
                para_names[4]: greedy_rule,
                para_names[5]: max_reward_coefficient
            }
        ],
        algo_names[1]: [
            {
                para_names[0]: 40000,
                para_names[1]: episodes,
                para_names[2]: 400,
                para_names[3]: 0.95,
                para_names[4]: [0.6, [0.9], 0.7],
                para_names[5]: max_reward_coefficient
            },
            {
                para_names[0]: 40000,
                para_names[1]: episodes,
                para_names[2]: 400,
                para_names[3]: 0.9,
                para_names[4]: [0.6, [0.9], 0.7],
                para_names[5]: max_reward_coefficient
            },
            {
                para_names[0]: 40000,
                para_names[1]: episodes,
                para_names[2]: 400,
                para_names[3]: 0.9,
                para_names[4]: [0.6, [0.9], 0.7],
                para_names[5]: max_reward_coefficient
            }
        ],
        algo_names[2]: [
            {
                para_names[0]: total_epi,
                para_names[1]: episodes,
                para_names[2]: 500,
                para_names[3]: 0.95,
                para_names[4]: [0.6],
                para_names[5]: max_reward_coefficient
            },
            {
                para_names[0]: total_epi,
                para_names[1]: episodes,
                para_names[2]: 500,
                para_names[3]: 0.95,
                para_names[4]: [0.7],
                para_names[5]: max_reward_coefficient
            },
            {
                para_names[0]: total_epi,
                para_names[1]: 350,
                para_names[2]: 500,
                para_names[3]: 0.8,
                para_names[4]: [0.7],
                para_names[5]: max_reward_coefficient
            }
        ]
    }

    #====== for small maze======#
    # parameter_dict = {
    #     algo_names[0]: [
    #         {
    #             para_names[0]: total_epi,
    #             para_names[1]: 300,
    #             para_names[2]: 500,
    #             para_names[3]: 0.95,
    #             para_names[4]: greedy_rule,
    #             para_names[5]: max_reward_coefficient
    #         },
    #         {
    #             para_names[0]: total_epi,
    #             para_names[1]: 350,
    #             para_names[2]: 500,
    #             para_names[3]: 0.95,
    #             para_names[4]: greedy_rule,
    #             para_names[5]: max_reward_coefficient
    #         },
    #         {
    #             para_names[0]: total_epi,
    #             para_names[1]: 300,
    #             para_names[2]: 500,
    #             para_names[3]: 0.8,
    #             para_names[4]: greedy_rule,
    #             para_names[5]: max_reward_coefficient
    #         }
    #     ],
    #     algo_names[1]: [
    #         {
    #             para_names[0]: total_epi,
    #             para_names[1]: 200,
    #             para_names[2]: 500,
    #             para_names[3]: 0.95,
    #             para_names[4]: greedy_rule,
    #             para_names[5]: max_reward_coefficient
    #         },
    #         {
    #             para_names[0]: total_epi,
    #             para_names[1]: 320,
    #             para_names[2]: 500,
    #             para_names[3]: 0.95,
    #             para_names[4]: greedy_rule,
    #             para_names[5]: max_reward_coefficient
    #         },
    #         {
    #             para_names[0]: total_epi,
    #             para_names[1]: 320,
    #             para_names[2]: 500,
    #             para_names[3]: 0.85,
    #             para_names[4]: greedy_rule,
    #             para_names[5]: max_reward_coefficient
    #         }
    #     ],
    #     algo_names[2]: [
    #         {
    #             para_names[0]: total_epi,
    #             para_names[1]: 180,
    #             para_names[2]: 500,
    #             para_names[3]: 0.95,
    #             para_names[4]: [0.6],
    #             para_names[5]: max_reward_coefficient
    #         },
    #         {
    #             para_names[0]: total_epi,
    #             para_names[1]: 350,
    #             para_names[2]: 500,
    #             para_names[3]: 0.95,
    #             para_names[4]: [0.7],
    #             para_names[5]: max_reward_coefficient
    #         },
    #         {
    #             para_names[0]: total_epi,
    #             para_names[1]: 350,
    #             para_names[2]: 500,
    #             para_names[3]: 0.8,
    #             para_names[4]: [0.7],
    #             para_names[5]: max_reward_coefficient
    #         }
    #     ]
    # }

    for index in range(len(lambda_arr)):
        lmb = lambda_arr[index]

        index = 0
        # index = 2  # just for tuning REMOVE LATER!!!

        # _parameters = parameter_dict[algo_names[0]][index]
        # single_run(f1, lmb, maze_index, _parameters[para_names[0]], _parameters[para_names[1]], algo_names[0],
        #            simulation, _parameters[para_names[2]], _parameters[para_names[3]], _parameters[para_names[4]],
        #            _parameters[para_names[5]])

        _parameters = parameter_dict[algo_names[1]][index]
        print(lmb)
        print(_parameters)
        single_run(f2, lmb, maze_index, _parameters[para_names[0]], _parameters[para_names[1]], algo_names[1],
                   simulation, _parameters[para_names[2]], _parameters[para_names[3]], _parameters[para_names[4]],
                   _parameters[para_names[5]])

        # _parameters = parameter_dict[algo_names[2]][index]
        # single_run(f3, lmb, maze_index, _parameters[para_names[0]], _parameters[para_names[1]], algo_names[2],
        #            simulation, _parameters[para_names[2]], _parameters[para_names[3]], _parameters[para_names[4]],
        #            _parameters[para_names[5]])

        print("finish lambda", str(lmb))

