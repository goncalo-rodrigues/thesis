import random

from common.world import World
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function
from pursuit.visualizers.transition_recorder import TransitionRecorder
import _pickle as pickle


def save_run(filename, number_episodes, agents, world_size=(5, 5), seed=100):
    random_instance = random.Random(seed)
    num_agents = len(agents)
    transition_f = get_transition_function(num_agents, world_size, random.Random(seed))
    reward_f = get_reward_function(num_agents, world_size)
    transition_recorder = TransitionRecorder()
    world = World(PursuitState.random_state(num_agents, world_size, random_instance), agents, transition_f, reward_f,
                  visualizers=(transition_recorder, ))

    for i in range(number_episodes):
        world.initial_state = PursuitState.random_state(num_agents, world_size, random_instance)
        _, _ = world.run(0, 1000)

    output_file = open(filename, 'wb')
    pickle.dump(transition_recorder.transitions, output_file)
    output_file.close()


def load_run(filename, agent, num_episodes, **transition_args):
    input_file = open(filename, 'rb')
    transitions = pickle.load(input_file)
    input_file.close()
    for transition in transitions:
        if num_episodes == 0:
            break
        if transition[2].terminal:
            num_episodes -= 1
        agent.transition(*transition, **transition_args)
