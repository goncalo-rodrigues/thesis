import random

import numpy as np

from pursuit.helper import neighbors, cornered, distance, directionx, directiony


class PursuitState(object):
    def __init__(self, agent_positions, prey_positions, world_size):
        assert(isinstance(agent_positions, tuple))
        assert(isinstance(prey_positions, tuple))
        assert(isinstance(world_size, tuple))
        assert(len(world_size) == 2)
        self.agent_positions = agent_positions
        self.prey_positions = prey_positions
        self.terminal = True
        self.world_size = world_size
        self.occupied = None
        for prey in prey_positions:
            if not cornered(self, prey, world_size):
                self.terminal = False
                break



    @property
    def occupied_cells(self):
        if not self.occupied:
            self.occupied = set(self.agent_positions) | set(self.prey_positions)
        return self.occupied


    @staticmethod
    def random_state(num_agents, world_size):
        assert(num_agents >= 4)
        world_size = tuple(world_size)
        num_preys = num_agents // 4

        assert(world_size[0]*world_size[1] > num_agents + num_preys)
        filled_positions = set()

        ppos_array = [(0,0)] * num_preys
        apos_array = [(0,0)] * num_agents
        for i in range(num_preys):
            while True:
                pos = (random.randint(0, world_size[0]-1), random.randint(0, world_size[1]-1))
                if pos not in filled_positions:
                    break

            ppos_array[i] = pos
            filled_positions.add(pos)

        for i in range(num_agents):
            while True:
                pos = (random.randint(0, world_size[0]-1), random.randint(0, world_size[1]-1))
                if pos not in filled_positions:
                    break

            apos_array[i] = pos
            filled_positions.add(pos)

        return PursuitState(prey_positions=tuple(ppos_array), agent_positions=tuple(apos_array), world_size=world_size)

    def __repr__(self):
        s = "Agents:\n" + '\n'.join(str(p) for p in self.agent_positions)
        s += "\n\n"
        s += "Prey:\n" + '\n'.join(str(p) for p in self.prey_positions)
        return s

    def __hash__(self):
        return hash(self.agent_positions)

    def __eq__(self, other):
        return self.agent_positions == other.agent_positions and \
               self.prey_positions == other.prey_positions and \
               self.world_size == other.world_size

    def features(self):
        return np.concatenate((np.array(self.agent_positions), np.array(self.prey_positions))).reshape(-1)

    def features_relative_prey(self):
        prey = self.prey_positions[0]
        relative_pos = []
        w, h = self.world_size
        for pos in self.agent_positions:
            dx, dy = distance(prey, pos, w, h)
            dx = dx * directionx(prey, pos, w)
            dy = dy * directiony(prey, pos, h)
            relative_pos.append(np.array([dx, dy]))
        return np.concatenate(relative_pos)
