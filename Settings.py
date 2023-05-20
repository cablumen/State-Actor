from enum import Enum

LOG_LEVEL = 3                       # minimum log level to print to terminal. see LogLevel below
LOG_SESSION_REWARD = True           # whether to create a log file for session rewards
LOG_EPISODE_REWARD = False          # whether to create a log file for episode rewards
LOG_TRAINING = True                 # whether to create a log file for actor training
LOG_EVALUATION = True               # whether to create a log file for actor evaluation

BATCH_SIZE = 512                    # batch size for training and prediction
EPOCHS = 40                         # epochs to train models and sub-models

#   exploration parameters
SESSION_COUNT = 10                  # session per architecture
EPISODE_COUNT = 200                 # episodes per session
MAX_TRAINING_STEPS = 300            # max steps per episode

#   epsilon exploration parameters
EPSILON_START = 1.0
EPSILON_DECAY = 0.975
EPSILON_MIN = 0.01

#   environment parameters
ACTION_SIZE = 2
OBSERVATION_SIZE = 4


class LogLevel(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2
    SILENT = 3
