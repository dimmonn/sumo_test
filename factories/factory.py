import numpy as np
from factories.parser import AlgorithmParameters
import torch
import torch.nn as nn
import torch.optim as optim


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
        # the Q - table stores the expected rewards for taking each action in each state
        self.Q = np.zeros((self.num_intersections, self.num_actions))

        input_size = self.num_intersections
        output_size = self.num_actions
        self.dqn = DQN(input_size, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)

    def q_learning(self, destination, num_episodes):
        for episode in range(num_episodes):
            done, state, total_reward = self.init()
            while not done:
                action = self.agent_action_selection_based_on_greedy_policy(state)
                next_state, reward = self.take_action(state, self.city_map)
                self.Q[state, action] += self.alpha * (
                        reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])
                state = next_state
                total_reward += reward
                if state == destination:
                    done = True
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    def agent_action_selection_based_on_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(self.Q[state])
        return action

    def take_action(self, state, city_map):
        # selects a random neighbor of the current intersection and checks the traffic condition.
        intersection = list(city_map['intersections'].keys())[state]
        neighbors = city_map['intersections'][intersection]['neighbors']
        next_intersection = np.random.choice(list(neighbors.keys()))
        traffic_condition = city_map['traffic_conditions'].get(f"{intersection}-{next_intersection}", 'normal')
        # The reward is determined based on the traffic condition.
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

    def dqn(self, destination, num_episodes):
        for episode in range(num_episodes):
            done, state, total_reward = self.init()
            while not done:
                state_tensor = torch.FloatTensor([state])
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.num_actions)
                else:
                    with torch.no_grad():
                        q_values = self.dqn(state_tensor)
                        action = np.argmax(q_values).item()
                next_state, reward = self.take_action(state, self.city_map)
                next_state_tensor = torch.FloatTensor([next_state])
                q_values_next = self.dqn(next_state_tensor)
                max_next_q_value = torch.max(q_values_next)
                target_q_value = reward + self.gamma * max_next_q_value
                q_values = self.dqn(state_tensor)
                loss = self.criterion(q_values[action], target_q_value)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                state = next_state
                total_reward += reward
                if state == destination:
                    done = True
            print(f"DQN - Episode {episode + 1}, Total Reward: {total_reward}")

    def init(self):
        state = np.random.randint(self.num_intersections)
        done = False
        total_reward = 0
        return done, state, total_reward

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


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
