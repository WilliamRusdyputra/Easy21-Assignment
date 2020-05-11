import numpy as np
from agents.monte_carlo import MonteCarloAgent
from agents.temporal_difference import SarsaAgent

mc_agent = MonteCarloAgent()
ss_agent = SarsaAgent()
lambdas = np.arange(0, 1.1, 0.1)


def monte_carlo_control():
    for i in range(1000000):
        print('EPISODE: ', i + 1)
        mc_agent.train()

    mc_agent.plot_value_surface()


def td_learning():
    mses = []
    for i in range(len(lambdas)):
        mse = []
        print('LAMBDA: ', lambdas[i])

        for j in range(10000):
            episode_mse = ss_agent.train(lambdas[i], mc_agent.action_value_function)
            mse.append(episode_mse)

        print('REPORT MSE: ', mse[len(mse) - 1])
        mses.append(mse)
        ss_agent.reset()

    episodes = np.arange(1, 10001)
    ss_agent.plot_mse(episodes, [mses[0], mses[10]])


monte_carlo_control()
td_learning()
