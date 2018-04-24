import random

from pursuit.agents.base_agent import Agent


class EpsRandomAgent(Agent):

    def __init__(self, id, agent, epsilon=0.1):
        super().__init__(id)
        self.eps = epsilon
        self.agent = agent

    def act(self, state):
        if random.random() < self.eps:
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            return directions[random.randint(0, 3)]
        else:
            return self.agent.act(state)