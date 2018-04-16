import random

from pursuit.helper import distance, direction, move, cornered, astar_distance, argmin, argmax
from pursuit.planning.astar import astar


class TeammateAwareAgent(object):
    def __init__(self, id):
        self.id = id
        self.last_prey_pos = None
        self.prey_id = None
        self.last_target = None


    def act(self, state):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        my_pos = state.agent_positions[self.id]
        w, h = state.world_size
        def choose_action():
            target = self.last_target
            # if already at destination, just follow the prey
            if my_pos == target:
                return direction(my_pos, self.last_prey_pos, w, h)

            action, dist = astar(my_pos, state.occupied_cells - {target}, target, (w, h))

            if action is None:
                return random.choice(directions)
            else:
                return action

        # if self.prey_id is not None and state.prey_positions[self.prey_id] == self.last_prey_pos:
        #     return choose_action()

        closest_prey, d, prey_id = None, None, 0
        for i, prey in enumerate(state.prey_positions):
            distance_to_prey = sum(distance(my_pos, prey, w, h))
            # get the closest non cornered prey
            if d is None or (not cornered(state, prey, (w, h)) and distance_to_prey < d):
                closest_prey, d, prey_id = prey, distance_to_prey, i

        self.prey_id = prey_id
        self.last_prey_pos = state.prey_positions[self.prey_id]
        # get the 4 agents closest to the prey
        # agents = sorted(state.agent_positions, key=lambda p: sum(distance(p, closest_prey, w, h)))
        # agents = agents[:4]
        agents = state.agent_positions

        # sort the agents by the worst shortest distance to the prey
        neighboring = [move(closest_prey, d, (w, h)) for d in directions]
        distances = [[sum(distance(a, p, w, h)) for p in neighboring] for a in agents]
        # distances = [(sorted((astar_distance(p, n, state.occupied_cells, (w, h)), i) for i, n in enumerate(neighboring)), j) for j, p in enumerate(agents)]

        # distances[i][j] is the distance of agent i to cell j
        # taken = set()
        target = 0
        for _ in range(len(agents)):
            min_dists = [min(d) for d in distances]
            min_inds = [argmin(d) for d in distances]
            selected_agent = argmax(min_dists)
            target = min_inds[selected_agent]
            # print('%d selected for %d' % (selected_agent, target))
            if selected_agent == self.id:
                break
            # remove the target from other agents
            for d in distances:
                d[target] = 2**31
            # remove the agent itself
            for i in range(len(distances[selected_agent])):
                distances[selected_agent][i] = -1


        self.last_target = neighboring[target]

        # print("%d, target: %s, distance: %d" % (self.id, self.last_target, d))

        return choose_action()


    def transition(self, state, actions, new_state, reward):
        pass