import numpy as np
import tensorflow as tf

import Settings
from Settings import LogLevel

class ActorModel:
    def __init__(self, logger, data_manager, architecture):
        self.name = architecture.name

        self.__logger = logger
        self.__data_manager = data_manager

        self.__architecture = architecture.value
        self.__model = None
        self.reset()

        self.__model_verbosity = True if Settings.LogLevel == 0 else False

    def train(self):
        self.__logger.print("ActorModel(train): " + self.name, LogLevel.INFO)
        if self.__data_manager.is_actor_training_ready():
            train_x, train_y, test_x, test_y = self.__data_manager.get_actor_data()
            training_history = self.__model.fit(train_x, train_y, batch_size=Settings.BATCH_SIZE, epochs=Settings.EPOCHS, validation_data=(test_x, test_y), verbose=self.__model_verbosity)
            self.__logger.log_training(training_history)
        
    def predict(self, state):
        self.__logger.print("ActorModel(predict): " + self.name, LogLevel.INFO)
        return self.__model.predict(state, verbose=self.__model_verbosity)

    def predict_batch(self, state_batch):
        self.__logger.print("ActorModel(predict_batch): " + self.name, LogLevel.INFO)
        return self.__model.predict(state_batch, batch_size=Settings.BATCH_SIZE, verbose=self.__model_verbosity)

    def evaluate(self):
        self.__logger.print("ActorModel(evaluate): " + self.name, LogLevel.INFO)
        if self.__data_manager.is_actor_training_ready():
            _, _, test_x, test_y = self.__data_manager.get_actor_data()
            predict_y = self.predict_batch(test_x)
            mse = ((np.square(predict_y - test_y)).mean(axis=1)).mean()
            self.__logger.log_evaluation(mse)

    def reset(self):
        actor = tf.keras.models.clone_model(self.__architecture)
        actor.compile(optimizer='adam', loss='mse', metrics=['mse'], jit_compile=True)
        self.__model = actor
