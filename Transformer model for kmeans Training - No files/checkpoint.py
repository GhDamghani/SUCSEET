from numpy import savez, load, ndarray
from datetime import datetime
from os import remove
from torch import save as torch_save


class ModelState:
    def save(self, model):
        if hasattr(self, "previousfile"):
            try:
                remove(self.previousfile + ".npz")
                remove(self.previousfile + ".pth")
            except FileNotFoundError:
                pass
        current_datetime = datetime.now()
        save_filename = "checkpoint_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        self.previousfile = save_filename
        atts = self.get_atts()
        atts_dict = {x: getattr(self, x) for x in atts}
        savez(save_filename, **atts_dict)
        torch_save(model.state_dict(), save_filename + ".pth")
        return save_filename

    def make_checkpoint(self, model_namespace):
        self.best_epoch = model_namespace.best_epoch
        self.lowest_test_loss = model_namespace.lowest_test_loss
        self.progressive_epoch = model_namespace.progressive_epoch
        self.last_test_loss = model_namespace.last_test_loss
        self.lr = model_namespace.lr
        self.epoch_i = model_namespace.epoch_i
        self.batch_i = model_namespace.batch_i

    def get_atts(self):
        return tuple(
            filter(
                lambda x: not (x.startswith("__") or callable(getattr(self, x))),
                dir(self),
            )
        )

    @classmethod
    def load(cls, filename):
        with load(filename, allow_pickle=True) as loaded:
            out = cls()
            for key, val in loaded.items():
                if isinstance(val, ndarray) and val.shape == ():
                    val = val.item()
                setattr(out, key, val)
        return out


""" if __name__ == "__main__":
    a = ModelState()
    a.bar = 2
    a.bath = "grow"
    a.car = False
    name = a.save()
    b = ModelState.load(name + ".npz")
    print(b.get_atts()) """
