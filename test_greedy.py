import random

import numpy as np
from common.world import World
from pursuit.agents.ad_hoc.adhoc_perfect_model import AdhocPerfectModel
from pursuit.agents.handcoded.greedy import GreedyAgent
from pursuit.agents.handcoded.teammate_aware import TeammateAwareAgent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function
import matplotlib.pyplot as plt
import scipy.stats as st
# from pursuit.visualizers.pygame_visualizer import PygameVisualizer

random.seed(100)
np.random.seed(100)

# agent_colors = [(random.randint(0, 255), random.randint(0, 50), random.randint(0, 255)) for _ in range(num_agents)]
# visualizer = PygameVisualizer(200, 200, agent_colors=agent_colors, agents=agents)
# visualizers = (visualizer,)

num_agents = 4
for world_size in ((5, 5), (10, 10), (20, 20)):
    for k in (20, 30, 50):
        # agents = [GreedyAgent(i) for i in range(num_agents)]
        agents = [GreedyAgent(i) for i in range(num_agents-1)] + [AdhocPerfectModel(3, mcts_n=1000, mcts_k=k,mcts_c=k*1.0)]

        iters = 100
        results = []

        for i in range(iters):
            transition_f = get_transition_function(num_agents, world_size, random.Random(100+i))
            reward_f = get_reward_function(num_agents, world_size)
            world = World(PursuitState.random_state(num_agents, world_size, random.Random(100+i)), agents, transition_f, reward_f)
            timesteps, reward = world.run(0, 1000)
            results.append(timesteps)


        print(world_size)
        print(k)
        print(np.average(results))
        print(np.std(results))
        print(st.t.interval(0.9, len(results)-1, loc=np.mean(results), scale=st.sem(results)))
        # plt.hist(results, bins=100)
        # plt.show()
        print("\n")

