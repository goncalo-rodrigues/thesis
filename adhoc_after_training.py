import random
from copy import deepcopy
from pathlib import Path
import os
import tqdm
import numpy as np
from multiprocessing.dummy import Process, Queue

from common.world import World
from pursuit.agents.ad_hoc.adhoc import AdhocAgent
from pursuit.agents.ad_hoc.adhoc_after_n_episodes import AdhocAfterNAgent
from pursuit.agents.handcoded.greedy import GreedyAgent
from pursuit.agents.handcoded.teammate_aware import TeammateAwareAgent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function


def run(q, threadid, world_q):
    world = deepcopy(world_q.get())
    random_instance = random.Random(100+threadid)
    num_agents = 4
    try:
        world.initial_state = PursuitState.random_state(num_agents, world_size, random_instance)
        timesteps, reward = world.run(0, 200)

        bacc = sum(adhoc.b_model.metric[-timesteps:]) / timesteps
        eacc = sum(adhoc.e_model.metric[-timesteps:]) / timesteps
        eacc_prey = sum(adhoc.e_model.metric_prey[-timesteps:]) / timesteps
        q.put((timesteps, bacc, eacc, eacc_prey))
    except Exception as e:
        print(e)


def init(episodes, world_q):
    random_instance = random.Random(100)
    random.seed(100)
    np.random.seed(100)

    num_agents = 4
    adhoc = AdhocAfterNAgent(agent_type(3), episodes, 3,
                             mcts_c=mcts_c, mcts_k=mcts_k, mcts_n=mcts_n, behavior_model_size=bsize,
                             environment_model_size=esize)
    agents = [agent_type(i) for i in range(num_agents - 1)] + [adhoc]
    transition_f = get_transition_function(num_agents, world_size, random.Random(100))
    reward_f = get_reward_function(num_agents, world_size)

    world = World(PursuitState.random_state(num_agents, world_size, random_instance), agents, transition_f, reward_f)

    for _ in tqdm.tqdm(range(episodes)):
        world.initial_state = PursuitState.random_state(num_agents, world_size, random_instance)
        world.run(0, 200)

    for _ in range(n_threads):
        world_q.put(world)

    return world, adhoc


def result_listener(q, results_folder):
    timesteps = []
    bacc = []
    eacc = []
    eacc_prey = []

    try:
        for ts, b, e, ep in tqdm.tqdm(iter(q.get, None), total=n_threads):
            timesteps.append(ts)
            bacc.append(b)
            eacc.append(e)
            eacc_prey.append(ep)
    # in case of interruption
    finally:
        np.save(str(results_folder / 'results'), np.array(timesteps))
        np.save(str(results_folder / 'eaccuracy'), np.array(eacc))
        np.save(str(results_folder / 'baccuracy'), np.array(bacc))
        np.save(str(results_folder / 'eaccuracyprey'), np.array(eacc_prey))


threads = []
n_threads = 2
q = Queue()
world_q = Queue()

base_path = Path('.')
results_folder = base_path / 'results_adhoc_test'
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


results_thread = Process(target=result_listener, args=(q, results_folder))
results_thread.start()
for j in range(n_threads):
    threads.append(Process(target=run, args=(q, j, world_q)))
    threads[-1].start()

world, adhoc = init(episodes-1, world_q)

for j in range(n_threads):
    threads[j].join()

q.put(None)
results_thread.join()
