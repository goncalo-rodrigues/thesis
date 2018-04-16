import random

import numpy as np
from multiprocessing import Process

from common.world import World
from pursuit.agents.adhoc import AdhocAgent
from pursuit.agents.adhoc_qlearning import AdhocQLearning
from pursuit.agents.greedy import GreedyAgent
from pursuit.agents.probabilistic_destinations import ProbabilisticDestinations
from pursuit.agents.teammate_aware import TeammateAwareAgent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function
# from pursuit.visualizers.pygame_visualizer import PygameVisualizer
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from scipy.interpolate import interp1d

def run(threadid):
    mcts_c = 1.41
    mcts_k = 10
    mcts_n = 100
    bsize = (64, 64)
    esize = (64, 64)

    random_instance = random.Random(100+threadid)
    random.seed(100+threadid)
    np.random.seed(100+threadid)

    num_agents = 4
    world_size = [5, 5]
    adhoc = AdhocAgent(3, mcts_c=mcts_c, mcts_k=mcts_k, mcts_n=mcts_n, behavior_model_size=bsize,
                       environment_model_size=esize)
    agents = [GreedyAgent(i) for i in range(num_agents - 1)] + [adhoc]
    transition_f = get_transition_function(num_agents, world_size, random.Random(100))
    reward_f = get_reward_function(num_agents, world_size)
    agent_colors = [(random.randint(0, 255), random.randint(0, 50), random.randint(0, 255)) for _ in range(num_agents)]
    # visualizer = PygameVisualizer(200, 200, agent_colors=agent_colors, agents=agents)
    # visualizers = (visualizer,)

    world = World(PursuitState.random_state(num_agents, world_size, random_instance), agents, transition_f, reward_f)
    iters = 500
    results = []
    bmodelmetric = []
    emodelmetric = []
    emodelmetric_prey = []
    # fig, ax1 = plt.subplots(clear=True)
    try:
        for i in range(iters):
            world.initial_state = PursuitState.random_state(num_agents, world_size, random_instance)
            timesteps, reward = world.run(0, 100)
            results.append(timesteps)
            timesteps = max(1, timesteps)
            bmodelmetric.append(sum(adhoc.b_model.metric[-timesteps:]) / timesteps)
            emodelmetric.append(sum(adhoc.e_model.metric[-timesteps:]) / timesteps)
            emodelmetric_prey.append(sum(adhoc.e_model.metric_prey[-timesteps:]) / timesteps)

            # if i == iters-1:
            #     fig.clf()
            #     ax1 = fig.add_subplot(1, 1, 1)
            #     ax2 = ax1.twinx()
            #
            #     ax1.set_ylim([0.7, 1])
            #     ax2.set_ylim([5, 30])
            #     n = len(adhoc.b_model.metric)
            #     ax1.plot(bmodelmetric, label='Behavior')
            #     ax1.plot(emodelmetric, label='Environment')
            #     ax1.plot(emodelmetric_prey, label='Environment (prey)')
            #
            #     ax2.plot(results, 'red', label='Timesteps')
            #     fig.legend()
            #     plt.draw()
            #     plt.pause(0.02)
    finally:
        print(sum(results) / len(results))
        # plt.savefig('plot_{}'.format(threadid))
        np.save('results_adhoc_t500/results_{}'.format(threadid), np.array(results))
        np.save('results_adhoc_t500/eaccuracy_{}'.format(threadid), np.array(emodelmetric))
        np.save('results_adhoc_t500/baccuracy_{}'.format(threadid), np.array(bmodelmetric))
        np.save('results_adhoc_t500/eaccuracyprey_{}'.format(threadid), np.array(emodelmetric_prey))

# threads = []
#
# for j in range(30):
#     threads.append(Process(target=run, args=(j, )))
#     threads[-1].start()
#
# for j in range(30):
#     threads[j].join()
world_size = [10,10]
agents = [GreedyAgent(i) for i in range(4)]
transition_f = get_transition_function(4, world_size, random.Random(100))
reward_f = get_reward_function(4, world_size)
world = World(PursuitState.random_state(4, world_size), agents, transition_f, reward_f)
results = []
for i in range(10000):
    world.initial_state = PursuitState.random_state(4, world_size)
    timesteps, reward = world.run(0, 2000)
    results.append(timesteps)
print(np.average(results))
print(np.std(results))