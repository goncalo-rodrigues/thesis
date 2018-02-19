import random

from pursuit.helper import distance, direction, directionx, directiony, move, cornered


class GreedyAgent(object):
    def __init__(self, id):
        self.id = id


    def act(self, state):
        w, h = state.world_size
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        my_pos = state.agent_positions[self.id]
        closest_prey, d = None, None
        for prey in state.prey_positions:
            distance_to_prey = sum(distance(my_pos, prey, w, h))
            # already neighboring some prey
            if distance_to_prey == 1:
                return direction(my_pos, prey, w, h)
            # get the closest non cornered prey
            if d is None or (not cornered(state, prey, (w, h)) and distance_to_prey < d):
                closest_prey, d = prey, distance_to_prey


        # unoccupied neighboring cells, sorted by proximity to agent
        targets = [move(closest_prey, d, (w, h)) for d in directions]
        targets = list(filter(lambda x: x not in state.occupied_cells, targets))

        if len(targets) == 0:
            return random.choice(directions)

        target = min(targets, key=lambda pos: sum(distance(my_pos, pos, w, h)))

        dx, dy = distance(my_pos, target, w, h)
        move_x = (directionx(my_pos, target, w), 0)
        move_y = (0, directiony(my_pos, target, h))
        pos_x = move(my_pos, move_x, (w, h))
        pos_y = move(my_pos, move_y, (w, h))

        # moving horizontally since there's a free cell
        if pos_x not in state.occupied_cells and (dx > dy or dx <= dy and pos_y in state.occupied_cells):
            return move_x
        # moving vertically since there's a free cell
        elif pos_y not in state.occupied_cells and (dx <= dy or dx > dy and pos_x in state.occupied_cells):
            return move_y
        # moving randomly since there are no free cells towards prey
        else:
            return random.choice(directions)

    def transition(self, state, actions, new_state, reward):
        pass