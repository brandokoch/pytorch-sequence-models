from sys import exc_info
from typing import final
import numpy as np
from sklearn import metrics
import logging
import torch
import wandb

# Configure Logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class MoveToGPUCallback():
    def before_batch(self):
        try:
            self.learner.batch[0] = self.learner.batch[0].to('cuda')
            self.learner.batch[1] = self.learner.batch[1].to('cuda')
        except Exception as e:
            log.error(
                "Exception occured: Can't move the batch to GPU", exc_info=True)

    def before_fit(self):
        try:
            self.learner.model = self.learner.model.to('cuda')
        except Exception as e:
            log.error(
                "Exception occured: Can't move the model to GPU", exc_info=True)

class TrackResultSentimentClf():

    def before_epoch(self):
        self.batch_cnt = 0
        self.loss_sum = 0
        self.ys = []
        self.preds = []

    def after_batch(self):
        self.batch_cnt += 1
        loss = self.learner.loss
        _, yb = self.learner.batch
        preds = self.learner.preds

        yb = yb.detach().cpu().numpy().tolist()
        preds = preds.detach().cpu().numpy().tolist()
        loss = loss.detach().cpu()

        self.loss_sum += loss
        self.ys.extend(yb)
        self.preds.extend(preds)

        # Tracking train loss by batch
        if self.learner.model.training:
            wandb.log({'Loss/Train': loss, 'epoch': self.learner.epoch_idx, 'batch': self.learner.batch_idx})


    def after_epoch(self):

        # Calculate avg epoch loss
        avg_loss = self.loss_sum/self.batch_cnt

        # Calculate accuracy
        def sigmoid(x): return 1 / (1+np.exp(-x))

        final_predictions = np.array(self.preds)
        final_targets = np.array(self.ys)

        final_predictions = sigmoid(final_predictions)
        final_predictions = final_predictions > 0.5
        accuracy = metrics.accuracy_score(final_targets, final_predictions)

        # Log
        if self.learner.model.training:
            log.info(f"Epoch: {self.learner.epoch_idx} | Training | Loss: {avg_loss:.4f} | Accuracy {accuracy}")
            wandb.log({'Acc/Train': accuracy, 'epoch': self.learner.epoch_idx})
        else:

            log.info(f"Epoch: {self.learner.epoch_idx} | Validation | Loss: {avg_loss:.4f} | Accuracy {accuracy}")
            wandb.log({'Acc/Val': accuracy, 'epoch': self.learner.epoch_idx})

            # Tracking validation loss by epoch
            wandb.log({'Loss/Val': avg_loss, 'epoch': self.learner.epoch_idx})

class TrackResultLM():

    def before_epoch(self):
        self.batch_cnt = 0
        self.loss_sum = 0
        self.ys = []
        self.preds = []

    def after_batch(self):
        self.batch_cnt += 1
        loss = self.learner.loss
        _, yb = self.learner.batch
        preds = self.learner.preds

        yb = yb.detach().cpu().numpy().tolist()
        preds = preds.detach().cpu().numpy().tolist()
        loss = loss.detach().cpu()

        self.loss_sum += loss
        self.ys.extend(yb)
        self.preds.extend(preds)

        # Tracking train loss by batch
        if self.learner.model.training:
            wandb.log({'Loss/Train': loss, 'epoch': self.learner.epoch_idx, 'batch': self.learner.batch_idx})


    def after_epoch(self):

        # Calculate avg epoch loss
        avg_loss = self.loss_sum/self.batch_cnt

        # Calculate accuracy 
        # (converting back to tensors so particualar PyTorch functions can be used)
        final_predictions = torch.tensor(self.preds)
        final_targets = torch.tensor(self.ys)

        final_predictions= torch.softmax(final_predictions, dim=1)
        final_predictions= torch.argmax(final_predictions, dim=1)

        final_predictions=torch.reshape(final_predictions, (-1,)).numpy()
        final_targets=torch.reshape(final_targets, (-1,)).numpy()

        accuracy = metrics.accuracy_score(final_targets, final_predictions)

        # Calculate perplexity 
        perplexity=np.exp(avg_loss)

        # Log
        if self.learner.model.training:
            log.info(f"Epoch: {self.learner.epoch_idx} | Training | Loss: {avg_loss:.4f} | Accuracy {accuracy:.2f} | Perplexity {perplexity:.2f}")
            wandb.log({'Ppx/Train': perplexity, 'epoch': self.learner.epoch_idx})
            wandb.log({'Acc/Train': accuracy, 'epoch':self.learner.epoch_idx})
        else:

            log.info(f"Epoch: {self.learner.epoch_idx} | Validation | Loss: {avg_loss:.4f} | Accuracy {accuracy:.2f} | Perplexity {perplexity:.2f}")
            wandb.log({'Ppx/Val': perplexity, 'epoch': self.learner.epoch_idx})
            wandb.log({'Acc/Val': accuracy, 'epoch':self.learner.epoch_idx})


            # Tracking validation loss by epoch
            wandb.log({'Loss/Val': avg_loss, 'epoch': self.learner.epoch_idx})