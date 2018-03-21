import random

from pursuit.helper import move
from pursuit.state import PursuitState


def get_transition_function(num_agents, world_size):

    def transition(state, actions):
        assert(len(actions) == num_agents)

        occupied_positions = set(state.prey_positions) | set(state.agent_positions)

        num_preys = len(state.prey_positions)

        apos_array = [None] * num_agents
        ppos_array = [None] * num_preys
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
        for i in range(num_preys):
            prey_pos = state.prey_positions[i]
            prey_action = random.choice(directions)
            prey_new_pos = move(prey_pos, prey_action, world_size)

            # if collision is detected, just go to the original position
            if prey_new_pos in occupied_positions:
                prey_new_pos = prey_pos

            occupied_positions.remove(prey_pos)
            occupied_positions.add(prey_new_pos)
            ppos_array[i] = prey_new_pos

        agents_indexs = list(range(num_agents))
        random.shuffle(agents_indexs)

        for i in agents_indexs:
            agent_pos = state.agent_positions[i]
            agent_action = actions[i]
            agent_new_pos = move(agent_pos, agent_action, world_size)

            # if collision is detected, just go to the original position
            if agent_new_pos in occupied_positions:
                agent_new_pos = agent_pos

            occupied_positions.remove(agent_pos)
            occupied_positions.add(agent_new_pos)
            apos_array[i] = agent_new_pos

        return PursuitState(prey_positions=tuple(ppos_array), agent_positions=tuple(apos_array), world_size=tuple(world_size))

    return transition
