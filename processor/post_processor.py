import numpy as np


class ShortestPathFinder:
    def __init__(self, city_map: dict = None, Q: np.ndarray = None):
        self.city_map = city_map
        self.Q = Q

    def find_shortest_path(self, initial_state, destination, max_path: int = 1000):
        state = initial_state
        path = [list(self.city_map['intersections'].keys())[state]]
        steps = 0
        while state != destination and steps < max_path:
            q_values = self.Q[state].copy()
            q_values[state] = -np.inf
            action = np.argmax(q_values)
            next_intersection = list(self.city_map['intersections'].keys())[action]
            path.append(next_intersection)
            next_state = list(self.city_map['intersections'].keys()).index(next_intersection)
            state = next_state
            steps += 1
        return path
