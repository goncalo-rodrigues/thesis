import random

import numpy as np

from common.world import World
from pursuit.agents.adhoc import AdhocAgent
from pursuit.agents.adhoc_qlearning import AdhocQLearning
from pursuit.agents.greedy import GreedyAgent
from pursuit.agents.probabilistic_destinations import ProbabilisticDestinations
from pursuit.agents.teammate_aware import TeammateAwareAgent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function
from pursuit.visualizers.pygame_visualizer import PygameVisualizer
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from scipy.interpolate import interp1d


num_agents = 4
world_size = [5, 5]
adhoc = AdhocAgent(3)
agents = [GreedyAgent(i) for i in range(num_agents-1)] + [adhoc]
transition_f = get_transition_function(num_agents, world_size)
reward_f = get_reward_function(num_agents, world_size)
agent_colors = [(random.randint(0, 255), random.randint(0, 50), random.randint(0, 255)) for _ in range(num_agents)]
visualizer = PygameVisualizer(200, 200, agent_colors=agent_colors, agents=agents)
visualizers = (visualizer, )

world = World(PursuitState.random_state(num_agents, world_size), agents, transition_f, reward_f, visualizers)
iters = 200
results = np.zeros(iters)
for i in range(iters):
    world.initial_state = PursuitState.random_state(num_agents, world_size)
    results[i] = (world.run(0, 100)[0])
    if i >= 0 and i%1==0:
        plt.clf()
        # plt.plot([results[k:k+10].mean() for k in range(i-10)], label='Performance')
        n = len(adhoc.behavior_metric)
        plt.plot([sum(adhoc.behavior_metric[k:k+10])/10 for k in range(n-10)], label='Behavior')
        plt.plot([sum(adhoc.env_metric[k:k+10])/10 for k in range(n-10)], label='Environment')
        plt.legend()
        plt.draw()
        plt.pause(0.02)

print(sum(results))
