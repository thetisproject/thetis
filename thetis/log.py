"""
Loggers for Thetis

Creates two logger instances, one for general model output and one for debug,
warning, error etc. messages.

To print to the model output stream, use :func:`~.print_output`.

Debug, warning etc. messages are issued with :func:`~.debug`, :func:`~.info`,
:func:`~.warning`, :func:`~.error`, :func:`~.critical` methods.
"""
from thetis.utility import COMM_WORLD
import logging

__all__ = ('logger', 'output_logger',
           'debug', 'info', 'warning', 'error', 'critical', 'print_output')


def get_new_logger(name, fmt='%(levelname)s %(message)s'):
    logger = logging.getLogger(name)

    for handler in logger.handlers:
        logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))

    if COMM_WORLD.rank != 0:
        handler = logging.NullHandler()

    logger.addHandler(handler)
    return logger


# create conventional logger for error etc messages
logger = get_new_logger('thetis')
logger.setLevel(logging.DEBUG)
log = logger.log
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

# create additional logger for model output
output_logger = get_new_logger('thetis_output', fmt='%(message)s')
output_logger.setLevel(logging.INFO)
print_output = output_logger.info
