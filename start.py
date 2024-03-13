from factories.factory import ReAlgorithmFactory
from data.map import *
from processor.post_processor import ShortestPathFinder

factory = ReAlgorithmFactory(city_map, rewards)
q_learning = factory.get_lambda_function('q_learning')

q_learning(3, num_episodes=1000)
spf = ShortestPathFinder(city_map=city_map, Q=factory.Q)
path = spf.find_shortest_path(0, 3, 1000)
print(path)

dqn = factory.get_lambda_function('dqn')
dqn(3, num_episodes=1000)
print()
