from common.world import World
from pursuit.agents.random import RandomAgent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function
from pursuit.visualizers.pygame_visualizer import PygameVisualizer

world_size = [10, 10]
agents = [RandomAgent()]*4
transition_f = get_transition_function(4, world_size)
reward_f = get_reward_function(4, world_size)
agent_colors = ((33, 155, 64), (210, 0, 0), (0, 75, 190), (200, 200, 200))
visualizer = PygameVisualizer(400, 400, agent_colors=agent_colors)


world = World(PursuitState.random_state(4, world_size), agents, transition_f, reward_f, (visualizer, ))
world.run(1)

