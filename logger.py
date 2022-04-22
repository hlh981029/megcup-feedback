import os
import sys
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, rank=None, name='', action='train', debug=False):
    # create logger
    logger = logging.getLogger('megcup')
    level = logging.INFO if not debug else logging.DEBUG
    logger.setLevel(level)
    logger.propagate = False

    # create formatter
    fmt = f'[%(asctime)s {name}] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored(f'[%(asctime)s {name}]', 'green') + \
        colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'{action}.log'), mode='a')

    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
