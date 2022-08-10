from enum import Enum

LOG_LEVEL = 0       # minimum log level to print to terminal. see LogLevel below

LABELS = 10         # labels in dataset

BATCH_SIZE = 128    # batch size for training and prediction
EPOCHS = 4          # epochs to train models and sub-models


class LogLevel(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2
    SILENT = 3