import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from env import Environment


class MonteCarloAgent:
    def __init__(self):
        self.actions = [0, 1]
        self.value_function = np.zeros([11, 22])
        self.value_steps = np.zeros([11, 22])
        self.action_value_function = np.zeros([11, 22, len(self.actions)])
        self.action_value_steps = np.zeros([11, 22, len(self.actions)])
        self.n_zero = 100

    def train(self):
        environment = Environment()
        terminated = False

        # initial state where both player and dealer draw black card
        card = round(random.uniform(1, 10))
        dealer_card = round(random.uniform(1, 10))

        states = []
        actions = []
        rewards = []

        # first state
        states.append([dealer_card, card])

        while not terminated:
            self.value_steps[dealer_card][card] += 1
            action = self.epsilon_action([dealer_card, card])
            action_index = 1 if action == 'hit' else 0

            state_prime, reward = environment.step([dealer_card, card], action)
            card = state_prime[1]

            if reward == -1 or reward == 1 or reward == 0:
                terminated = True
            else:
                # reward is 2 when state is not terminated because 0 overlaps with termination state draw reward
                states.append(state_prime)
                reward = 0

            actions.append(action_index)
            rewards.append(reward)

        average_reward = np.sum(rewards)

        visited_states = []

        for i in range(len(states)):
            card, dealer_card = states[i][1], states[i][0]

            if self.check_visited(visited_states, card, dealer_card):
                self.action_value_steps[dealer_card][card][actions[i]] += 1

                current_action_value = self.action_value_function[dealer_card][card][actions[i]]
                current_action_value_step = self.action_value_steps[dealer_card][card][actions[i]]

                # Q(St, At) = Q(St, At) + 1 / N(St, At) * (Gt - Q(St, At))
                new_action_value = (current_action_value + (1 / current_action_value_step) *
                                    (average_reward - current_action_value))

                self.action_value_function[dealer_card][card][actions[i]] = new_action_value

                visited_states.append([card, dealer_card])

    def check_visited(self, states, card, dealer_card):
        for state in states:
            if state[0] == card and state[1] == dealer_card:
                return False
        return True

    def epsilon_action(self, state):
        epsilon = self.n_zero / (self.n_zero + self.value_steps[state[0]][state[1]])

        greedy_action = np.argmax(self.action_value_function[state[0]][state[1]])
        greedy_action = 'hit' if greedy_action == 1 else 'stick'

        explore_action = np.random.choice(self.actions)
        explore_action = 'hit' if explore_action == 1 else 'stick'

        prob = np.random.random()
        if prob <= epsilon:
            return explore_action
        else:
            return greedy_action

    def plot_value_surface(self):
        z = []

        # compute V*
        for i in range(1, 11):
            for j in range(1, 22):
                self.value_function[i][j] = max(self.action_value_function[i][j][0],
                                                self.action_value_function[i][j][1])

        for i in range(1, 22):
            for j in range(1, 11):
                z.append(self.value_function[j][i])

        x = np.arange(1, 11, 1)
        y = np.arange(1, 22, 1)
        x, y = np.meshgrid(x, y)
        z = np.array(z)
        figure = plt.figure(figsize=(8, 6))

        ax = figure.gca(projection=Axes3D.name)
        ax.axes.set_zlim3d(bottom=-0.6, top=1.0)
        ax.plot_trisurf(x.flatten(), y.flatten(), z, cmap='plasma')
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_title('Value Surface')
        plt.show()
