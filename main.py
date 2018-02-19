import random

from common.world import World
from pursuit.agents.greedy import GreedyAgent
from pursuit.agents.random import RandomAgent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function
from pursuit.visualizers.pygame_visualizer import PygameVisualizer

num_agents = 4
world_size = [10, 10]
agents = [GreedyAgent(i) for i in range(num_agents)]
transition_f = get_transition_function(num_agents, world_size)
reward_f = get_reward_function(num_agents, world_size)
agent_colors = [(random.randint(0, 255), random.randint(0, 50), random.randint(0, 255)) for _ in range(num_agents)]
visualizer = PygameVisualizer(400, 400, agent_colors=agent_colors)


world = World(PursuitState.random_state(num_agents, world_size), agents, transition_f, reward_f)
results = []
for i in range(10000):
    world.initial_state = PursuitState.random_state(num_agents, world_size)
    results.append(world.run(0, 1000)[0])
print(sum(results))
