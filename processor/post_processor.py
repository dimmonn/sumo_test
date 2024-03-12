import numpy as np


class ShortestPathFinder:
    def __init__(self, city_map: dict = None, Q: np.ndarray = None):
        self.city_map = city_map
        self.Q = Q

    def find_shortest_path(self, initial_state, destination):
        state = initial_state
        path = [list(self.city_map['intersections'].keys())[state]]
        while state != destination:
            action = np.argmax(self.Q[state])
            next_state = list(self.city_map['intersections'].keys())[action]
            path.append(next_state)
            state = list(self.city_map['intersections'].keys()).index(next_state)
        return path
