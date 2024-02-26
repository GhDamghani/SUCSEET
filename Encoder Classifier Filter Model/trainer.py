import torch
from tqdm import tqdm
import numpy as np
from multiprocessing import Process, Pipe
from checkpoint import Checkpoint, loop_timer


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        train_dataset,
        val_dataset,
        batch_size,
        epochs,
        logger,
        autosave_seconds,
        resume=False,
        checkpointfile=None,
        log_times=10,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs

        self.train_dataset = train_dataset
        self.test_mode = not bool(self.train_dataset)

        self.val_dataset = val_dataset

        self.criterion = criterion

        if not self.test_mode:
            self.logger = logger
            self.log_interval = (
                int(round(len(self.train_dataset) / self.batch_size / log_times))
                if log_times is not None
                else len(self.train_dataset) + 1
            )
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.patience = self.epochs
            self.autosave_seconds = autosave_seconds
        self.resume = resume
        if resume:
            self.checkpoint = Checkpoint.load(checkpointfile, self)
            logger('resuming from checkpoint: "{}"'.format(checkpointfile))
        else:
            self.checkpoint = Checkpoint()
            self.epoch_i = 0
            self.batch_i = 0

            if not self.test_mode:
                self.lowest_val_loss = float("inf")
                self.best_epoch = 0
                self.train_loss = 0

    def step(self, X, y):
        pred = self.model(X).cpu()
        loss = self.criterion(pred, y)
        corrects = torch.sum(torch.argmax(pred, -1) == y).item()
        return pred, loss, corrects

    def _train(self):
        self.model.train()
        if self.resume:
            self.train_dataset.offset = self.batch_i
        else:
            self.train_loss = 0
            self.train_corrects = 0
            self.train_total = 0
            interval_loss = 0
            interval_corrects = 0
            interval_total = 0
        iterator = self.train_dataset
        self.train_dataset.epoch_i = self.epoch_i
        with open("tqdm_batch.log", "w") as file:
            progress_bar = tqdm(
                iterator,
                file=file,
                bar_format="[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
            iterator = enumerate(progress_bar, start=self.train_dataset.offset)
            for self.batch_i, (X, y) in iterator:
                if self.autosave_reciever.poll():
                    if self.autosave_reciever.recv() == True:
                        self.checkpoint.update_set(self)
                        self.checkpoint.save(self)
                _, loss, corrects = self.step(X, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if torch.isnan(torch.tensor(loss.item())).item():
                    pass

                loss_ = loss.item() * y.numel()
                interval_loss += loss_
                self.train_loss += loss_
                self.train_corrects += corrects
                interval_corrects += corrects
                self.train_total += y.numel()
                interval_total += y.numel()

                if torch.isnan(torch.tensor(self.train_loss)).item():
                    pass

                if (self.batch_i + 1) % self.log_interval == 0:
                    accuracy = interval_corrects / interval_total
                    loss_str = f"Loss: {interval_loss/interval_total:.5g}"
                    acc_str = f"Accuracy: {accuracy:7.2%}"
                    left_str = f"{loss_str:12} | {acc_str:12}"
                    right_str = str(progress_bar)
                    self.logger(left_str, right=right_str)
                    interval_loss = 0
                    interval_corrects = 0
                    interval_total = 0
            self.logger(
                f"Epoch {self.epoch_i} Train loss: {self.train_loss/self.train_total:.5g}  Accuracy: {self.train_corrects/self.train_total:02.2%}"
            )
        self.resume = False

    def validate(self, return_data=False):
        self.model.eval()
        self.val_loss = 0
        total = 0
        corrects = 0
        if return_data:
            pred_labels = []
            y_labels = []
        with torch.no_grad():
            for X, y in self.val_dataset:
                pred, loss, corrects_ = self.step(X, y)
                self.val_loss += loss.item() * y.numel()
                corrects += corrects_
                total += y.numel()
                if return_data:
                    pred_labels.append(torch.argmax(pred, -1).numpy())
                    y_labels.append(y.cpu().numpy())
        self.val_acc = corrects / total
        self.val_loss /= total
        if not self.test_mode:
            if return_data:
                self.logger(
                    f"Evaluation Complete. Val loss: {self.val_loss:.5g} Accuracy: {self.val_acc:02.2%}"
                )
            else:
                self.logger(
                    f"Epoch {self.epoch_i}   Val loss: {self.val_loss:.5g} Accuracy: {self.val_acc:02.2%}"
                )
        if return_data:
            pred_labels = np.concatenate(pred_labels)
            y_labels = np.concatenate(y_labels)
            return pred_labels, y_labels

        return

    def train(self):
        self.autosave_reciever, autosave_sender = Pipe(duplex=False)
        autosave_prc = Process(
            target=loop_timer,
            args=(self.autosave_seconds, autosave_sender),
        )
        autosave_prc.start()
        self.lr = self.optimizer.param_groups[0]["lr"]
        with open("tqdm_epoch.log", "w") as file:
            progress_bar = tqdm(
                range(self.epoch_i, self.train_dataset.epochs),
                file=file,
                bar_format="[{elapsed}<{remaining}, " "{rate_fmt}{postfix}]",
            )

            for epoch_i in progress_bar:
                self.epoch_i = epoch_i

                epoch_str = f"Epoch: {self.epoch_i:04d}"
                lr_str = (
                    f"Lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:.4g}"
                )
                self.logger(f"{epoch_str} {lr_str} {progress_bar}", right="=")

                self._train()
                self.validate()

                if self.val_loss < self.lowest_val_loss:
                    self.best_epoch = epoch_i
                    self.lowest_val_loss = self.val_loss
                    torch.save(self.model.state_dict(), "model.pth")
                    self.logger("*")
                    self.logger(f"Best Epoch! Model Saved", right="*")
                    self.logger("*")
                else:
                    self.logger(f"Not a Progressive Epoch!")
                if self.epoch_i - self.best_epoch >= self.patience:
                    self.logger("&")
                    self.logger(
                        f"Early stopping after {self.patience} epochs of no new best epoch",
                    )
                    self.logger("&")
                    break

                #### Learning Rate Scheduler ####

                self.scheduler.step(self.val_loss)
                if self.optimizer.param_groups[0]["lr"] != self.lr:
                    self.lr = self.optimizer.param_groups[0]["lr"]
                    self.logger(
                        f"New learning rate: {self.lr:.4g}",
                        right=" ",
                    )
                self.logger("=")