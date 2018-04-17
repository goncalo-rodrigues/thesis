import random

import numpy as np
from common.world import World
from pursuit.agents.handcoded.teammate_aware import TeammateAwareAgent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function
import matplotlib.pyplot as plt
# from pursuit.visualizers.pygame_visualizer import PygameVisualizer

random_instance = random.Random(100)
random.seed(100)
np.random.seed(100)

num_agents = 4
world_size = [5, 5]
agents = [TeammateAwareAgent(i) for i in range(num_agents)]
transition_f = get_transition_function(num_agents, world_size, random.Random(100))
reward_f = get_reward_function(num_agents, world_size)
agent_colors = [(random.randint(0, 255), random.randint(0, 50), random.randint(0, 255)) for _ in range(num_agents)]
# visualizer = PygameVisualizer(200, 200, agent_colors=agent_colors, agents=agents)
# visualizers = (visualizer,)

world = World(PursuitState.random_state(num_agents, world_size, random_instance), agents, transition_f, reward_f)
iters = 1000
results = []

for i in range(iters):
    world.initial_state = PursuitState.random_state(num_agents, world_size, random_instance)
    timesteps, reward = world.run(0, 1000)
    results.append(timesteps)


print(np.average(results))
print(np.std(results))
plt.hist(results)
plt.show()