# Reinforcement Learning for City Navigation

This project implements reinforcement learning algorithms to find optimal paths for navigation in a simulated city environment. The project consists of several components including data handling, algorithm implementation, and post-processing.

## Project Structure

The project is organized into the following directories and files:

- **data**: Contains `map.py` which defines the city environment and traffic conditions.
- **factories**: Includes `factory.py` and `parser.py`. `factory.py` provides a factory class to create reinforcement learning algorithms, while `parser.py` parses algorithm parameters from a configuration file.
- **processor**: Contains `post_processor.py`, which implements a class for post-processing, specifically for finding the shortest path.
- **start.py**: The main script that initializes the city map, sets up the reinforcement learning factory, trains the algorithms, and finds the shortest path.

## Usage

To run the project:

1. Ensure you have Python 3.x installed.
2. Install the required dependencies using `pip`:
    ```
    pip install torch numpy
    ```
3. Run the main script:
    ```
    python start.py
    ```

## File Details

- **map.py**: Defines the city map including intersections, neighbors, and traffic conditions.
- **factories/factory.py**: Implements a factory class to create reinforcement learning algorithms such as Q-learning and DQN.
- **factories/parser.py**: Parses algorithm parameters from a configuration file.
- **processor/post_processor.py**: Provides a class to find the shortest path in the city map using the Q-values learned by the algorithms.
- **start.py**: The entry point of the program, responsible for initializing the city map, training the algorithms, and finding the shortest path.
- **app.properties**: Configuration file containing algorithm parameters.

## Configuration

Algorithm parameters such as learning rates, discount factors, and exploration rates can be configured in the `app.properties` file. Each algorithm has its section in the configuration file with corresponding parameters.

## Acknowledgments

This project is inspired by https://github.com/gmum/ml2023-24/tree/main/lectures.
