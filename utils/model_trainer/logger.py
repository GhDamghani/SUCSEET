import logging
import logging.config
from functools import partial


class Logger:
    def __init__(self, name):
        self.root_logger = logging.getLogger(name)
        self.root_logger.setLevel(logging.DEBUG)

        self.handler = logging.FileHandler(name, "w", "utf-8")
        self.handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(message)s", "%m/%d/%Y %H:%M:%S")
        self.handler.setFormatter(formatter)
        self.root_logger.addHandler(self.handler)
        self.console_width = 80

    def close(self):
        self.root_logger.removeHandler(self.handler)
        self.handler.close()

    def log(self, s, right=None, model=False):
        if model:
            return self.root_logger.info(f"\n{s}")
        s_len = len(s)
        if s_len == 1:
            return self.log(self.console_width * s)
        else:
            if right is None:
                return self.root_logger.info(f"| {s:<{self.console_width}} |")
            elif len(right) == 1:
                s = f"{(self.console_width//2 - s_len // 2 - 1) * right} {s} {(self.console_width//2 - (s_len - (s_len // 2)) - 1) * right}"
                return self.log(s)
            else:
                s = f'{s}{(self.console_width-s_len-len(right))*" "}{right}'
                return self.log(s)
