import random

from pursuit.agents.base_agent import Agent


class RandomAgent(Agent):
    def act(self, state):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        return directions[random.randint(0, 3)]
