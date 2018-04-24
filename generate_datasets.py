from pathlib import Path

import os

from pursuit.agents.handcoded.eps_random import EpsRandomAgent
from pursuit.agents.handcoded.greedy import GreedyAgent
from pursuit.agents.handcoded.random import RandomAgent
from pursuit.agents.handcoded.teammate_aware import TeammateAwareAgent
from utils import save_run

path = Path('datasets')
os.makedirs(str(path), exist_ok=True)
#
# save_run(path / '5x5_greedy', 10000, [GreedyAgent(i) for i in range(4)], world_size=(5, 5))
# save_run(path / '5x5_greedy_random', 10000, [GreedyAgent(i) for i in range(3)] + [EpsRandomAgent(3, RandomAgent(3), 0.5)], world_size=(5, 5))
# save_run(path / '5x5_ta', 10000, [TeammateAwareAgent(i) for i in range(4)], world_size=(5, 5))

# save_run(path / '10x10_greedy', 10000, [GreedyAgent(i) for i in range(4)], world_size=(10, 10))
save_run(path / '10x10_greedy_random', 1000, [GreedyAgent(i) for i in range(3)] + [EpsRandomAgent(3, RandomAgent(3), 0.25)], world_size=(10, 10))
# save_run(path / '10x10_ta', 10000, [TeammateAwareAgent(i) for i in range(4)], world_size=(10, 10))

# save_run(path / '20x20_greedy', 10000, [GreedyAgent(i) for i in range(4)], world_size=(20, 20))
save_run(path / '20x20_greedy_random', 1000, [GreedyAgent(i) for i in range(3)] + [EpsRandomAgent(3, RandomAgent(3), 0.1)], world_size=(20, 20))
# save_run(path / '20x20_ta', 10000, [TeammateAwareAgent(i) for i in range(4)], world_size=(20, 20))
