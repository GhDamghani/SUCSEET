import pandas as pd
import matplotlib.pyplot as plt
from os.path import join


def plot_metrics(config):
    train_log_file = join(config.output_path, "train_log.csv")
    train_log = pd.read_csv(train_log_file)
    val_log_file = join(config.output_path, "val_log.csv")
    val_log = pd.read_csv(val_log_file)

    for column in train_log.columns[1:]:
        plt.figure(figsize=(8, 3))
        plt.plot(train_log[column].ravel() / train_log["total"].ravel(), label="train")
        plt.plot(val_log[column].ravel() / val_log["total"].ravel(), label="val")
        plt.legend()
        plt.xlabel("Epoch")
        if column == "corrects":
            column = "accuracy"
        plt.ylabel(column)
        plt.title(f"{column} vs. Epoch")
        plt.tight_layout()
        plt.savefig(join(config.output_path, f"{column}.png"))


def main(config):

    plot_metrics(config)


if __name__ == "__main__":
    import config

    config = config.Config

    main(config)
