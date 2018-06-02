import random
from pathlib import Path

import numpy as np

from common.world import World
from pursuit.agents.ad_hoc.adhoc import AdhocAgent
from pursuit.agents.handcoded.greedy import GreedyAgent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function
import os

folder = Path('adhoc_dataset')
os.makedirs(str(folder), exist_ok=True)
random_instance = random.Random(100)
random.seed(100)
np.random.seed(100)
world_size = (10, 10)
mcts_k = 10
mcts_n = 1000
mcts_c = mcts_k*1
bsize = (64, 64)
esize = (64, 64)
agent_type = GreedyAgent

num_agents = 4
adhoc = AdhocAgent(3, mcts_c=mcts_c, mcts_k=mcts_k, mcts_n=mcts_n, behavior_model_size=bsize,
                   environment_model_size=esize)
adhoc = AdhocAgent.load('adhoc_dataset/10x10greedy_random_200')
agents = [agent_type(i) for i in range(num_agents - 1)] + [adhoc]
transition_f = get_transition_function(num_agents, world_size, random.Random(100))
reward_f = get_reward_function(num_agents, world_size)

world = World(PursuitState.random_state(num_agents, world_size, random_instance), agents, transition_f, reward_f)
save_episodes = (1, 5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500)
current_episode = 0
for episodes in save_episodes:
    for current_episode in range(current_episode, episodes):

            world.initial_state = PursuitState.random_state(num_agents, world_size, random_instance)
            if current_episode > 200:
                timesteps, reward = world.run(0, 100)
                print(timesteps)
    if current_episode > 200:
        adhoc.save(str(folder / ('10x10greedy_random_' + str(episodes))))