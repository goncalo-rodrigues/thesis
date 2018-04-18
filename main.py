import random
from pathlib import Path
import os
import tqdm

import numpy as np
from multiprocessing import Process, Queue

from common.world import World
from pursuit.agents.ad_hoc.adhoc import AdhocAgent
from pursuit.agents.handcoded.greedy import GreedyAgent
from pursuit.agents.handcoded.teammate_aware import TeammateAwareAgent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function


def run(q, threadid):
    random_instance = random.Random(100+threadid)
    random.seed(100+threadid)
    np.random.seed(100+threadid)

    num_agents = 4
    adhoc = AdhocAgent(3, mcts_c=mcts_c, mcts_k=mcts_k, mcts_n=mcts_n, behavior_model_size=bsize,
                       environment_model_size=esize)
    agents = [agent_type(i) for i in range(num_agents - 1)] + [adhoc]
    transition_f = get_transition_function(num_agents, world_size, random.Random(100))
    reward_f = get_reward_function(num_agents, world_size)

    world = World(PursuitState.random_state(num_agents, world_size, random_instance), agents, transition_f, reward_f)
    results = []
    bmodelmetric = []
    emodelmetric = []
    emodelmetric_prey = []
    try:
        for i in range(episodes):
            world.initial_state = PursuitState.random_state(num_agents, world_size, random_instance)
            timesteps, reward = world.run(0, 200)
            results.append(timesteps)
            timesteps = max(1, timesteps)
            bmodelmetric.append(sum(adhoc.b_model.metric[-timesteps:]) / timesteps)
            emodelmetric.append(sum(adhoc.e_model.metric[-timesteps:]) / timesteps)
            emodelmetric_prey.append(sum(adhoc.e_model.metric_prey[-timesteps:]) / timesteps)
            q.put(1)
    finally:
        np.save(str(results_folder / 'results_{}'.format(threadid)), np.array(results))
        np.save(str(results_folder / 'eaccuracy_{}'.format(threadid)), np.array(emodelmetric))
        np.save(str(results_folder / 'baccuracy_{}'.format(threadid)), np.array(bmodelmetric))
        np.save(str(results_folder / 'eaccuracyprey_{}'.format(threadid)), np.array(emodelmetric_prey))


def progress_listener(q):
    progress_bar = tqdm.tqdm(total=n_threads * episodes)
    for _ in iter(q.get, None):
        progress_bar.update(1)
    progress_bar.close()

threads = []
n_threads = 1
q = Queue()

base_path = Path('.')
results_folder = base_path / 'results_adhoc_greedycorrected_1'
if results_folder.exists():
    results_folder.rmdir() # will throw error if not empty
os.makedirs(str(results_folder))

# these variables will be used by the function above
mcts_c = 1.41
mcts_k = 10
mcts_n = 100
bsize = (64, 64)
esize = (64, 64)
world_size = (5, 5)
agent_type = GreedyAgent
episodes = 200

progress_thread = Process(target=progress_listener, args=(q, ))
progress_thread.start()

for j in range(n_threads):
    threads.append(Process(target=run, args=(q, j, )))
    threads[-1].start()

for j in range(n_threads):
    threads[j].join()

q.put(None)
progress_thread.join()
