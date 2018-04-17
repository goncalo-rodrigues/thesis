from common.world import World
from pursuit.agents.handcoded.teammate_aware import TeammateAwareAgent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function
from pursuit.visualizers.pygame_visualizer import PygameVisualizer

num_agents = 4
world_size = (5, 5)
agents = [TeammateAwareAgent(i) for i in range(num_agents)]
prey_moves = [(-1, 0), (1, 0), (0, 0)]
transition_f = get_transition_function(num_agents, world_size, prey_moves=prey_moves)
reward_f = get_reward_function(num_agents, world_size)
agent_colors = [(255, 0, 0), (175, 0, 75), (75, 0, 175), (0, 0, 255)]
visualizer = PygameVisualizer(200, 200, agent_colors=agent_colors, agents=agents)
visualizers = (visualizer,)

initial_state = PursuitState(((0, 1), (1, 0), (0, 3), (1, 2)), ((0, 0),), world_size)

world = World(initial_state, agents, transition_f, reward_f, visualizers=visualizers)
print(world.run(1, 100))

# expected actions
# RIGHT LEFT UP DOWN NOOP
# 4, 2, 2, 4 DOWN LEFT LEFT DOWN
# 4, 2, 2, 1 DOWN LEFT LEFT RIGH
# 4, 3, 2, 1 DOWN UUUP LEFT RIGH
# 1, 3, 2, 3 RIGH UUUP LEFT UUUP
# 1, 3, 2, 1 RIGH UUUP LEFT RIGH
# 1, 3, 2, 1 RIGH UUUP LEFT RIGH