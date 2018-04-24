import random
from collections import defaultdict
from multiprocessing.pool import Pool
from pathlib import Path
import os
import tqdm

import numpy as np
from multiprocessing import Process, Queue, Manager

from common.world import World
from pursuit.agents.ad_hoc.adhoc import AdhocAgent
from pursuit.agents.ad_hoc.adhoc_after_n_episodes import AdhocAfterNAgent
from pursuit.agents.handcoded.greedy import GreedyAgent
from pursuit.agents.handcoded.teammate_aware import TeammateAwareAgent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function
from utils import load_run

def run(progress_q, results_q, threadid, adhoc_filename, episodes, results_folder, world_size):
    random_instance = random.Random(100+threadid)
    random.seed(100+threadid)
    np.random.seed(100+threadid)

    num_agents = 4
    # adhoc = AdhocAgent(3, mcts_c=mcts_c, mcts_k=mcts_k, mcts_n=mcts_n, behavior_model_size=bsize,
    #                    environment_model_size=esize)
    # load_run(dataset_folder / dataset_name, adhoc, episodes, fit=False, compute_metrics=False)
    adhoc = AdhocAgent.load(adhoc_filename)
    agents = [agent_type(i) for i in range(num_agents - 1)] + [adhoc]
    transition_f = get_transition_function(num_agents, world_size, random.Random(100))
    reward_f = get_reward_function(num_agents, world_size)

    world = World(PursuitState.random_state(num_agents, world_size, random_instance), agents, transition_f, reward_f)
    timesteps, reward = world.run(0, 500)
    progress_q.put(1)

    results_q.put((str(results_folder / 'results_eps{}'.format(episodes)), timesteps))
    results_q.put((str(results_folder / 'eaccuracy_eps{}'.format(episodes)), np.average(adhoc.e_model.metric)))
    results_q.put((str(results_folder / 'baccuracy_eps{}'.format(episodes)), np.average(adhoc.b_model.metric)))
    results_q.put((str(results_folder / 'eaccuracyprey_eps{}'.format(episodes)), np.average(adhoc.e_model.metric_prey)))


def progress_listener(q):
    progress_bar = tqdm.tqdm(total=n_threads*5)
    for _ in iter(q.get, None):
        progress_bar.update(1)
    progress_bar.close()


def aggregator(q, n_threads):
    results = defaultdict(list)
    try:
        for filename, value in iter(q.get, None):
            results[filename].append(value)
            if len(results[filename]) == n_threads:
                np.save(filename, results[filename])
                results.pop(filename)
    finally:
        for filename, array in results.items():
            np.save(filename, array)


n_threads = 100
progress_q = Manager().Queue()
results_q = Manager().Queue()


base_path = Path('.')
dataset_folder = base_path / 'datasets'


# these variables will be used by the function above
mcts_c = 1.41
mcts_k = 10
mcts_n = 100
bsize = (64, 64)
esize = (64, 64)
agent_type = GreedyAgent

progress_thread = Process(target=progress_listener, args=(progress_q, ))
results_thread = Process(target=aggregator, args=(results_q, n_threads))
progress_thread.start()
results_thread.start()
threads = []
pool = Pool(4)

for world_size in ((5, 5), ):
    dataset_name = '{}x{}_greedy_random'.format(*world_size)
    results_folder = base_path / '{}x{}_greedy_test'.format(*world_size)
    if results_folder.exists():
        results_folder.rmdir()  # will throw error if not empty
    os.makedirs(str(results_folder))
    for episodes in (1, 5, 20, 50, 200):
        adhoc = AdhocAgent(3, mcts_c=mcts_c, mcts_k=mcts_k, mcts_n=mcts_n, behavior_model_size=bsize,
                           environment_model_size=esize)
        load_run(dataset_folder / dataset_name, adhoc, episodes, fit=False, compute_metrics=False)
        adhoc_filename = 'adhoc_' + str(episodes)
        adhoc.save(adhoc_filename)
        for j in range(n_threads):
            threads.append(pool.apply_async(run, args=(progress_q, results_q, j, adhoc_filename, episodes, results_folder, world_size)))

for thread in threads:
    thread.get()

progress_q.put(None)
results_q.put(None)
progress_thread.join()
results_thread.join()
