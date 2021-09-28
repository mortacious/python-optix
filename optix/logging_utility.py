import logging


class Logger(object):
    _optix_to_logging_level = {
        4: logging.DEBUG,
        3: logging.WARN,
        2: logging.ERROR,
        1: logging.FATAL,
    }

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def __call__(self, level, tag, msg):
        self.logger.log(self._optix_to_logging_level[level], msg, extra={'tag': tag})
