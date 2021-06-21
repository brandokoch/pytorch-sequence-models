from logging import error, log
from sys import exc_info
import os

import torch
import torch.nn as nn
import torch.optim as optim

from utils.logconf import logging
import wandb

import config
from datasets import get_dataloaders
from learner import Learner
import model_dispatcher
import callback_dispatcher


# Configure Logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Disable tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class SeqModelTrainingApp:
    def __init__(self):

        log.info('----- Training Started -----')

        # Device handling
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        log.info(f'GPU availability: {self.use_cuda}')
        log.info(f'Device name is {torch.cuda.get_device_name()}')

    def main(self):
        train_dl, val_dl = get_dataloaders(config.DATASET)

        try:
            model = model_dispatcher.models[config.MODEL]
            loss_func = getattr(nn, config.LOSS)()
            opt_func = getattr(optim, config.OPTIMIZER)
            cbs = callback_dispatcher.callbacks[config.MODEL]
        except Exception as e:
            log.error(
                "Exception occured: Configuration is invalid, check the README", exc_info=True)

        learner = Learner(model, train_dl, val_dl, loss_func,
                          config.LR, cbs, opt_func)
        learner.fit(config.EPOCHS)
        learner.save(os.path.join('runs', config.RUN_NAME, config.MODEL+'.pt'))


if __name__ == "__main__":
    wandb.init(project='rnn', name=config.MODEL)
    SeqModelTrainingApp().main()
