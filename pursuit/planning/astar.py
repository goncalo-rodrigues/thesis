from pursuit.helper import neighbors, distance, direction
from heapq import *


def astar(initial_pos, obstacles, target, world_size):
    w, h = world_size
    # each item in the queue contains (heuristic+cost, cost, position, parent)
    queue = [(distance(n, target, w, h)+1, 1, n, initial_pos)
             for n in neighbors(initial_pos, world_size) if n not in obstacles]
    queue = heapify(queue)
    # hashmap that maps each visited item to its parent
    visited = {initial_pos: None}

    while queue:
        _, cost, pos, parent = heappop(queue)
        if pos in visited:
            continue
        visited[pos] = parent

        if pos == target:
            break

        for n in neighbors(pos, world_size):
            if n not in obstacles:
                heappush(queue, (distance(n, target, w, h)+cost+1, cost + 1, n, pos))

    if target not in visited:
        return None

    current = target
    while visited[current] != initial_pos:
        current = visited[current]

    return direction(initial_pos, current, h, w)


