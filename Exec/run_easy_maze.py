"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from MazeEnv.easy_maze import MazeSimulator
from LearningAlgos.easy_maze_RL import QLearningTable
import time

def running(epi, time_in_ms, is_render):
    for episode in range(epi):
        # initiate the agent
        agent = env.reset()

        while True:
            # fresh env
            env.render(time_in_ms)

            # RL choose action based on observation
            action = RL.choose_action(str(agent))

            # RL take action and get next observation and reward
            new_state, reward, is_done = env.taking_action(action)

            # RL learn from this transition
            RL.learn(str(agent), action, reward, str(new_state))

            # swap observation
            agent = new_state

            # break while loop when end of this episode
            if is_done:
                break

    # end of game
    print('game over')
    if is_render:
        time.sleep(1)
        env.destroy()


if __name__ == "__main__":
    env = MazeSimulator(5, 4, [0, 0], True)
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(1, running(10, 0.001, True))
    env.mainloop()
    print(RL.q_table)