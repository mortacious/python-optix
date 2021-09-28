import logging


class Logger(object):
    """
    Helper class to translate the optix logging calls to python logging calls.
    It takes any Logger compatible object and outputs all messages with the following log levels:

    OptiX   | Logging
    -----------------
      1     |  FATAL
      2     |  ERROR
      3     |  WARN
      4     |  DEBUG

    See also https://raytracing-docs.nvidia.com/optix7/api/html/group__optix__types.html#gafa74ebb0b1ab57289a4d5a080cd4c090
    for more information.

    Parameters
    ----------
    logger: logging.Logger or compatible object
        The logger to output the messages to
    """
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
