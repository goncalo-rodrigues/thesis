import random

from pursuit.helper import move
from pursuit.state import PursuitState


def get_transition_function(num_agents, world_size):

    def transition(state, actions):
        assert(len(actions) == num_agents)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        prey_positions = {}

        num_preys = len(state.prey_positions)
        preys = list(range(num_preys))

        while len(preys) > 0:
            prey_i = preys.pop()
            prey_pos = state.prey_positions[prey_i]
            prey_action = random.choice(directions)
            prey_new_pos = move(prey_pos, prey_action, world_size)

            # if collision is detected, just go to the original position
            if prey_new_pos in prey_positions or prey_new_pos in state.agent_positions:
                if prey_pos in prey_positions:
                    colliding_i = prey_positions.pop(prey_pos)
                    preys.append(colliding_i)
                prey_positions[prey_pos] = prey_i
            else:
                prey_positions[prey_new_pos] = prey_i

        agent_positions = {}
        agents = list(range(num_agents))
        while len(agents) > 0:
            agent_i = agents.pop()
            agent_pos = state.agent_positions[agent_i]
            agent_action = actions[agent_i]
            agent_new_pos = move(agent_pos, agent_action, world_size)

            # if collision is detected, just go to the original position
            if agent_new_pos in prey_positions or agent_new_pos in agent_positions:
                if agent_pos in agent_positions:
                    colliding_i = agent_positions.pop(agent_pos)
                    agents.append(colliding_i)
                agent_positions[agent_pos] = agent_i
            else:
                agent_positions[agent_new_pos] = agent_i

        ppos_array = [None]*num_preys
        apos_array = [None]*num_agents

        for pos, i in prey_positions:
            ppos_array[i] = pos
        for pos, i in agent_positions:
            apos_array[i] = pos

        return PursuitState(prey_positions=ppos_array, agent_positions=apos_array, world_size=world_size)

    return transition
