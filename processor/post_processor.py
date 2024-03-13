import numpy as np


class ShortestPathFinder:
    def __init__(self, city_map: dict = None, Q: np.ndarray = None):
        self.city_map = city_map
        self.Q = Q

    def find_shortest_path(self, initial_state, destination, max_path: int = 1000):
        state = initial_state
        path = [list(self.city_map['intersections'].keys())[state]]
        max = 0
        while state != destination and max < max_path:
            action = np.argmax(self.Q[state])
            next_intersection = list(self.city_map['intersections'].keys())[action]
            path.append(next_intersection)
            next_state = list(self.city_map['intersections'].keys()).index(next_intersection)
            state = next_state
            max += 1
        return path
