from pursuit.helper import neighbors, distance, direction
from heapq import *


def astar(initial_pos, obstacles, target, world_size):
    if initial_pos == target:
        return (0, 0), 0
    w, h = world_size
    obstacles = obstacles - {target}
    # each item in the queue contains (heuristic+cost, cost, position, parent)
    queue = [(sum(distance(n, target, w, h))+1, 1, n, initial_pos)
             for n in neighbors(initial_pos, world_size) if n not in obstacles]

    heapify(queue)
    # hashmap that maps each visited item to its parent
    visited = {initial_pos: None}

    while len(queue) > 0:
        _, cost, pos, parent = heappop(queue)
        if pos in visited:
            continue
        visited[pos] = parent

        if pos == target:
            break

        for n in neighbors(pos, world_size):
            if n not in obstacles:
                heappush(queue, (sum(distance(n, target, w, h))+cost+1, cost + 1, n, pos))

    if target not in visited:
        return None, w*h

    i = 1
    current = target
    while visited[current] != initial_pos:
        current = visited[current]
        i+=1

    return direction(initial_pos, current, h, w), i


