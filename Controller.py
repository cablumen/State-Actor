import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Architectures import Architectures
from AtomicModel import AtomicModel
from CompositeModel import CompositeModel
from DataManager import DataManager
from Logger import Logger

class Controller:
    def __init__(self):
        self.__logger = Logger()
        self.__data_manager = DataManager()

        # Intialize experiments to run
        self.__atomic_experiments = [Architectures.FC_1_32]
        self.__composite_experiments = [(Architectures.COMP_FC_1_16, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]

        self.run_experiments()

    def run_experiments(self):
        models = []

        # intialize models
        for architecture in self.__atomic_experiments:
            models.append(AtomicModel(self.__logger, self.__data_manager, architecture))

        for (architecture, digits) in self.__composite_experiments:
            models.append(CompositeModel(self.__logger, self.__data_manager, architecture, digits))

        # train models
        for model in models:
            model.train()

        # evaluate models
        for model in models:
            model.evaluate()

if __name__ == '__main__':
    Controller()