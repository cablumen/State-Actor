import tensorflow as tf

import Settings
from Settings import LogLevel

GAMMA = 0.95

class CriticModel:
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
        self.logger.print("CriticModel(train): " + self.name, LogLevel.INFO)
        if self.data_manager.is_critic_training_ready():
            states, next_states, rewards, dones = self.data_manager.get_critic_data()
            targets = rewards + GAMMA * self.predict(next_states) * (1 - dones)
            self.model.fit(states, targets, epochs=Settings.EPOCHS, verbose=self.model_verbosity)

    def predict(self, state):
        self.logger.print("CriticModel(predict): " + self.name, LogLevel.INFO)
        return self.model(state)

    def reset(self):
        critic_model = tf.keras.models.clone_model(self.architecture)
        critic_model.compile(optimizer='adam', loss='mse', metrics=['mse'], jit_compile=True)
        self.model = critic_model
