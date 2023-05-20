import numpy as np
import tensorflow as tf

import Settings
from Settings import LogLevel

class ActorModel:
    def __init__(self, logger, data_manager, architecture):
        self.name = architecture.name

        self.logger = logger
        self.data_manager = data_manager

        self.architecture = architecture.value
        self.model = None
        self.reset()

        self.model_verbosity = True if Settings.LogLevel == 0 else False

    @tf.function(jit_compile=True)
    def train(self):
        self.logger.print("ActorModel(train): " + self.name, LogLevel.INFO)
        if self.data_manager.is_actor_training_ready():
            train_x, train_y, test_x, test_y = self.data_manager.get_actor_data()
            training_history = self.model.fit(train_x, train_y, batch_size=Settings.BATCH_SIZE, epochs=Settings.EPOCHS, validation_data=(test_x, test_y), verbose=self.model_verbosity)
            self.logger.log_training(training_history)

    def predict(self, state):
        self.logger.print("ActorModel(predict): " + self.name, LogLevel.INFO)
        return self.model(state)

    def evaluate(self):
        self.logger.print("ActorModel(evaluate): " + self.name, LogLevel.INFO)
        if self.data_manager.is_actor_training_ready():
            _, _, test_x, test_y = self.data_manager.get_actor_data()
            predict_y = self.predict(test_x)
            mse = ((np.square(predict_y - test_y)).mean(axis=1)).mean()
            self.logger.log_evaluation(mse)

    def reset(self):
        actor = tf.keras.models.clone_model(self.architecture)
        actor.compile(optimizer='adam', loss='mse', metrics=['mse'], jit_compile=True)
        self.model = actor
