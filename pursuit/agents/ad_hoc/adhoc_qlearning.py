import random
from collections import defaultdict

import numpy as np

from pursuit.agents.base_agent import Agent


class AdhocQLearning(Agent):
    actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def __init__(self, id):
        super().__init__(id)
        self.eps = 0.05
        self.qtable = defaultdict(lambda: np.zeros(4))
        self.lrate = 0.8
        self.discount = 0.95
        self.dataset = {}
        self.i = 0

    def act(self, state):
        # eps-greedy
        if random.random() < self.eps:
            return random.choice(self.actions)
        else:
            best_actions = np.argwhere(self.qtable[state] == np.max(self.qtable[state])).reshape(-1)
            return self.actions[np.random.choice(best_actions)]

    def transition(self, state, actions, new_state, reward):
        # self.dataset[state] = actions[:3]
        # self.i += 1
        # if self.i == 10000:
        #     np.save('greedy_dataset.npy', self.dataset)
        a = self.actions.index(actions[self.id])
        self.qtable[state][a] = (1 - self.lrate) * self.qtable[state][a] + \
                                self.lrate * (reward + self.discount * max(self.qtable[new_state]))

        if new_state.terminal:
            self.qtable[new_state][:] = reward
