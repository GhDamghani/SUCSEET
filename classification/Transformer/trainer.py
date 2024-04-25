import torch
from tqdm import tqdm
import numpy as np
from os.path import join


class Trainer:
    def __init__(
        self,
        model,
        loss,
        metrics,
        val_dataset,
        model_task,
        train_dataset=None,
        optimizer=None,
        batch_size=-1,
        epochs=None,
        logger=None,
        scheduler=None,
        log_times=10,
        device="cpu",
        output_path=".",
    ) -> None:
        self.model = model
        self.val_dataset = val_dataset
        self.model_task = model_task

        self.batch_size = batch_size
        self.epochs = epochs

        self.train_dataset = train_dataset
        self.test_mode = not bool(self.train_dataset)

        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.loss = loss
        self.metrics = metrics
        self.metrics_val = {key: 0 for key in self.metrics}
        self.metrics_val["loss"] = 0

        self.device = device
        self.output_path = output_path

        if not self.test_mode:
            self.log_interval = (
                int(round(len(self.train_dataset) / self.batch_size / log_times))
                if log_times is not None
                else len(self.train_dataset) + 1
            )
            self.patience = self.epochs

            self.metrics_train = {key: 0 for key in self.metrics}
            self.metrics_interval = {key: 0 for key in self.metrics}
            self.metrics_train["loss"] = 0
            self.metrics_interval["loss"] = 0

            self.lowest_val_loss = float("inf")
            self.best_epoch = 0

    def update_metrics(self, pred, y, *dicts):
        with torch.no_grad():
            for key, value in self.metrics.items():
                met = value(pred, y)
                for dict0 in dicts:
                    dict0[key] += met

    def totorch(self, X, y):
        X = torch.from_numpy(X).float().to(self.device)
        if self.model_task == "classification":
            y = torch.from_numpy(np.squeeze(y)).long()
        elif self.model_task == "regression":
            y = torch.from_numpy(y).float()
        return X, y

    def step(self, X, y):
        pred = self.model(X).cpu()
        loss = self.loss(pred, y)
        return pred, loss

    def train_(self):
        self.model.train()
        for key in self.metrics_train:
            self.metrics_train[key] = 0
            self.metrics_interval[key] = 0
        with open("tqdm_batch.log", "w") as file:
            progress_bar = tqdm(
                self.train_dataset.generate_batch(self.batch_size, shuffle=True),
                file=file,
                bar_format="[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
            iterator = enumerate(progress_bar)
            for self.batch_i, (X, y) in iterator:
                X, y = self.totorch(X, y)
                pred, loss = self.step(X, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.update_metrics(pred, y, self.metrics_train, self.metrics_interval)
                self.metrics_train["loss"] += loss.item() * len(y)
                self.metrics_interval["loss"] += loss.item() * len(y)

                if torch.isnan(torch.tensor(loss.item())).item():
                    pass

                if (self.batch_i + 1) % self.log_interval == 0:

                    loss_str = f"Loss: {self.metrics_interval['loss']/self.metrics_interval['total']:.5g}"
                    self.metrics_interval["loss"] = 0

                    if self.model_task == "classification":
                        accuracy_interval = (
                            self.metrics_interval["corrects"]
                            / self.metrics_interval["total"]
                        )
                        self.metrics_interval["corrects"] = 0
                        acc_str = f"Accuracy: {accuracy_interval:7.2%}"
                        left_str = f"{loss_str:12} | {acc_str:12}"
                    elif self.model_task == "regression":
                        left_str = f"{loss_str:12}"
                    self.metrics_interval["total"] = 0
                    right_str = str(progress_bar)
                    self.logger(left_str, right=right_str)

            loss_str = f"Train Loss: {self.metrics_train['loss']/self.metrics_train['total']:.5g}"
            left_str = f"Epoch {self.epoch_i} {loss_str}"
            if self.model_task == "classification":
                acc_str = f" Accuracy: {self.metrics_train['corrects']/self.metrics_train['total']:02.2%}"
                left_str += acc_str
            self.logger(left_str)

    def validate(self, return_data=False):
        self.model.eval()
        if return_data:
            pred_all = []
            y_all = []

        for key in self.metrics_val:
            self.metrics_val[key] = 0
        with torch.no_grad():
            for X, y in self.val_dataset.generate_batch(self.batch_size):
                X, y = self.totorch(X, y)
                pred, loss = self.step(X, y)

                self.update_metrics(pred, y, self.metrics_val)
                self.metrics_val["loss"] += loss.item() * len(y)

                if return_data:
                    pred_all.append(pred.numpy())
                    y_all.append(y.numpy())
        if not (self.test_mode):
            loss_str = (
                f"  Val Loss: {self.metrics_val['loss']/self.metrics_val['total']:.5g}"
            )
            if return_data:
                left_str = f"Evaluation Complete. {loss_str}"
            else:
                left_str = f"Epoch {self.epoch_i} {loss_str}"

            if self.model_task == "classification":
                acc_str = f" Accuracy: {self.metrics_val['corrects']/self.metrics_val['total']:02.2%}"
                left_str += acc_str
            self.logger(left_str)
        if return_data:
            pred_all = np.concatenate(pred_all)
            y_all = np.concatenate(y_all)
            return pred_all, y_all, self.metrics_val.copy()

        return

    def train(self):
        self.lr = self.optimizer.param_groups[0]["lr"]
        with open("tqdm_epoch.log", "w") as file:
            progress_bar = tqdm(
                range(self.epochs),
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

                self.train_()
                self.validate()

                if self.metrics_val["loss"] < self.lowest_val_loss:
                    self.best_epoch = epoch_i
                    self.lowest_val_loss = self.metrics_val["loss"]
                    torch.save(
                        self.model.state_dict(), join(self.output_path, "model.pth")
                    )
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
                if self.scheduler is not None:
                    self.scheduler.step(self.val_loss)
                    if self.optimizer.param_groups[0]["lr"] != self.lr:
                        self.lr = self.optimizer.param_groups[0]["lr"]
                        self.logger(
                            f"New learning rate: {self.lr:.4g}",
                            right=" ",
                        )
                    self.logger("=")
