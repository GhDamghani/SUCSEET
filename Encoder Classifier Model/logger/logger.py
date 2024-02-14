import logging
import logging.config
from pathlib import Path
from utils import read_json
from functools import partial


def logger_fcn(s, logger, right=None, model=False):
    if model:
        return logger.info(f"\n{s}")
    console_width = 80
    s_len = len(s)
    if s_len == 1:
        return logger_fcn(console_width * s, logger)
    else:
        if right is None:
            return logger.info(f"| {s:<{console_width}} |")
        elif len(right) == 1:
            s = f"{(console_width//2 - s_len // 2 - 1) * right} {s} {(console_width//2 - (s_len - (s_len // 2)) - 1) * right}"
            return logger_fcn(s, logger)
        else:
            s = f'{s}{(console_width-s_len-len(right))*" "}{right}'
            return logger_fcn(s, logger)


def get_logger(logging_file, filemode):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=logging_file,
        filemode=filemode,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        encoding="utf-8",
        level=logging.DEBUG,
    )

    logger = partial(logger_fcn, logger=logger)
    return logger
