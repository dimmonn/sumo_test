import torch
import numpy as np
from factories.parser import AlgorithmParameters


class ReAlgorithmFactory:
    def __init__(self, city_map: dict = None, rewards: dict = None):
        self.lambda_functions = {
            'q_learning': self.q_learning,
            'dqn': self.dqn,
            'policy_gradient': self.policy_gradient,
            'monte_carlo': self.monte_carlo,
            'temporal_difference': self.temporal_difference
        }
        self.city_map = city_map
        self.rewards = rewards
        algo_props = AlgorithmParameters('app.properties')
        q_learning_parameters = algo_props.get_parameters('q_learning')
        self.alpha = float(q_learning_parameters['alpha'])
        self.gamma = float(q_learning_parameters['gamma'])
        self.epsilon = float(q_learning_parameters['epsilon'])
        self.num_intersections = len(city_map['intersections'])
        self.num_actions = self.num_intersections
        self.Q = np.zeros((self.num_intersections, self.num_actions))

    def q_learning(self, destination, num_episodes):
        for episode in range(num_episodes):
            state = np.random.randint(self.num_intersections)
            done = False
            total_reward = 0
            while not done:
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.num_actions)
                else:
                    action = np.argmax(self.Q[state])
                next_state, reward = self.take_action(state, self.city_map)
                self.Q[state, action] += self.alpha * (
                        reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])
                state = next_state
                total_reward += reward
                if state == destination:
                    done = True
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    def take_action(self, state, city_map):
        intersection = list(city_map['intersections'].keys())[state]
        neighbors = city_map['intersections'][intersection]['neighbors']
        next_intersection = np.random.choice(list(neighbors.keys()))
        traffic_condition = city_map['traffic_conditions'].get(f"{intersection}-{next_intersection}", 'normal')
        if traffic_condition == 'light':
            reward = self.rewards['waiting_penalty']
        elif traffic_condition == 'congested':
            reward = self.rewards['negative']
        elif traffic_condition == 'accident':
            reward = self.rewards['negative']
        else:
            reward = 0
        next_state = list(city_map['intersections'].keys()).index(next_intersection)

        return next_state, reward

    def dqn(self, dqn, city_map, destination, num_episodes):
        for episode in range(num_episodes):
            state = np.random.randint(self.num_intersections)
            done = False
            total_reward = 0
            while not done:
                state_tensor = torch.FloatTensor([state])
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.num_actions)
                else:
                    with torch.no_grad():
                        q_values = dqn(state_tensor)
                        action = np.argmax(q_values).item()
                next_state, reward = self.take_action(state, city_map)
                next_state_tensor = torch.FloatTensor([next_state])
                q_values_next = dqn(next_state_tensor)
                max_next_q_value = torch.max(q_values_next)
                target_q_value = reward + self.gamma * max_next_q_value
                q_values = self.dqn(state_tensor)
                loss = criterion(q_values[action], target_q_value)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                state = next_state
                total_reward += reward
                if state == destination:
                    done = True
            print(f"DQN - Episode {episode + 1}, Total Reward: {total_reward}")

    def policy_gradient(self, city_map, destination, num_episodes):
        pass

    def monte_carlo(self, city_map, destination, num_episodes):
        pass

    def temporal_difference(self, city_map, destination, num_episodes):
        pass

    def get_lambda_function(self, key):
        if key in self.lambda_functions:
            return self.lambda_functions[key]
        else:
            raise ValueError("Invalid key: {}".format(key))
