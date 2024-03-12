import configparser


class AlgorithmParameters:
    def __init__(self, file_path):
        self.config = configparser.ConfigParser()
        self.config.read(file_path)

    def get_parameters(self, algorithm):
        if algorithm not in self.config.sections():
            raise ValueError(f"Algorithm '{algorithm}' not found in the configuration file.")
        parameters = {}
        for key, value in self.config[algorithm].items():
            parameters[key] = str(value)
        return parameters
