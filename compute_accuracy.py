import numpy as np
from matplotlib import patches

from common.world import World
from pursuit.agents.ad_hoc.adhoc import AdhocAgent, ACTIONS
from pursuit.agents.handcoded.teammate_aware import TeammateAwareAgent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function
import matplotlib.pyplot as plt

agent = TeammateAwareAgent(0)
world_size = (10, 10)
adhoc_filename = 'adhoc_dataset/10x10ta_random_200'
adhoc = AdhocAgent.load(adhoc_filename)
positions = [(3, 3), (3, 7), (7, 3)]
prey = (5, 5)
result = np.zeros(world_size)

for x in range(world_size[0]):
    for y in range(world_size[1]):
        if (x, y) in positions:
            continue
        initial_state = PursuitState(tuple([(x, y)] + positions), (prey,), world_size)
        adhoc.b_model.predict(initial_state)
        predicted_action_dist = adhoc.b_model.cache[initial_state][0]

        true_action = agent.act(initial_state)

        result[x, y] = predicted_action_dist[ACTIONS.index(true_action)]

fig,ax = plt.subplots(1)
im = ax.imshow(result, interpolation='nearest')
fig.colorbar(im)
for x, y in positions:
    rect = patches.Rectangle((x-0.5, y-0.5), 0.95, 0.95, linewidth=1, edgecolor='r', facecolor='black')
    ax.add_patch(rect)

rect = patches.Rectangle((prey[0]-0.5, prey[1]-0.5), 0.95, 0.95, linewidth=1, edgecolor='r', facecolor='red')
ax.add_patch(rect)
plt.show()