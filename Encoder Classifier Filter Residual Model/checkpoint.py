from numpy import savez, load, ndarray
from datetime import datetime
from os import remove
from torch import save as torch_save, load as torch_load
from time import sleep


class Checkpoint:
    kwargs = {
        "best_epoch",
        "lowest_val_loss",
        "epoch_i",
        "batch_i",
        "train_loss",
        "train_corrects",
        "train_total",
        "lr",
    }

    def __init__(self):
        pass

    def save(self, trainer):
        if hasattr(self, "previousfile"):
            try:
                remove(self.previousfile + ".npz")
                remove(self.previousfile + ".pth")
            except FileNotFoundError:
                pass
        current_datetime = datetime.now()
        save_filename = "checkpoint_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        self.previousfile = save_filename
        atts_dict = {x: getattr(self, x) for x in Checkpoint.kwargs}
        savez(save_filename, **atts_dict)
        torch_save(trainer.model.state_dict(), save_filename + ".pth")
        return save_filename

    def update_set(self, trainer):
        for kw in Checkpoint.kwargs:
            setattr(self, kw, getattr(trainer, kw))

    def update_get(self, trainer):
        for kw in Checkpoint.kwargs:
            setattr(trainer, kw, getattr(self, kw))

    @classmethod
    def load(cls, filename, trainer):
        with load(filename + ".npz", allow_pickle=True) as loaded:
            out = cls()
            for key, val in loaded.items():
                if key not in Checkpoint.kwargs:
                    continue
                if isinstance(val, ndarray) and val.shape == ():
                    val = val.item()
                setattr(out, key, val)
        out.update_get(trainer)
        trainer.model.load_state_dict(torch_load(filename + ".pth"))
        return out


def loop_timer(time, sender):
    while True:
        sleep(time)
        sender.send(True)


if __name__ == "__main__":
    a = Checkpoint()
    a.bar = 2
    a.bath = "grow"
    a.car = False
    name = a.save()
    b = Checkpoint.load(name + ".npz")
    print(b.get_atts())
