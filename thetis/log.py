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
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
import os

__all__ = ('logger', 'output_logger',
           'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL',
           'log', 'debug', 'info', 'warning', 'error', 'critical',
           'print_output', 'set_thetis_loggers', 'thetis_log_level',
           'set_log_directory')

logger_format = {
    'thetis': '%(name)s:%(levelname)s %(message)s',
    'thetis_output': '%(message)s',
}


class ThetisLogConfig:
    """Module-wide config object"""
    filename = None


thetis_log_config = ThetisLogConfig()


def set_thetis_loggers(comm=COMM_WORLD):
    """Set stream handlers for log messages.

    :kwarg comm: The communicator the handler should be collective
         over. If provided, only rank-0 on that communicator will
         write to the log, other ranks will use a :class:`logging.NullHandler`.
         If set to ``None``, all ranks will write to log.
    """
    for name, fmt in logger_format.items():
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)

        if comm is None or comm.rank == 0:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(fmt=fmt))
        else:
            handler = logging.NullHandler()
        logger.addHandler(handler)


def set_log_directory(output_directory, comm=COMM_WORLD, mode='w'):
    """
    Forward all log output to `output_directory/log` file.

    When called, a new empty log file is created.

    If called twice with a different `output_directory`, a warning is raised,
    and the new log file location is assigned. The old log file or the
    `output_directory` are not removed.

    :arg output_directory: the directory where log file is stored
    :kwarg comm: The communicator the handler should be collective
         over. If provided, only rank-0 on that communicator will
         write to the log, other ranks will use a :class:`logging.NullHandler`.
         If set to ``None``, all ranks will write to log.
    :kwarg mode: write mode, 'w' removes previous log file (if any), otherwise
        appends to it. Default: 'w'.
    """
    def rm_all_file_handlers():
        for name in logger_format:
            logger = logging.getLogger(name)
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)

    def assign_file_handler(logfile):
        if comm.rank == 0:
            if os.path.exists(output_directory):
                if not os.path.isdir(output_directory):
                    raise IOError('file with same name exists', output_directory)
            else:
                os.makedirs(output_directory)

        for name, fmt in logger_format.items():
            logger = logging.getLogger(name)
            if comm.rank == 0:
                new_handler = logging.FileHandler(logfile, mode='a')
                new_handler.setFormatter(logging.Formatter(fmt))
            else:
                new_handler = logging.NullHandler()
            logger.addHandler(new_handler)

    logfile = os.path.join(output_directory, 'log')
    if thetis_log_config.filename == logfile:
        # no change
        return
    changed = (thetis_log_config.filename is not None
               and thetis_log_config.filename != logfile)
    if changed:
        old_file = str(thetis_log_config.filename)
        rm_all_file_handlers()
    if mode == 'w' and os.path.isfile(logfile):
        # silently remove previous log
        if comm.rank == 0:
            os.remove(logfile)
    thetis_log_config.filename = logfile
    assign_file_handler(logfile)
    if changed:
        msg = (f'Setting a log file "{logfile}" that differs from previous '
               f'"{old_file}", removing old handler')
        warning(msg)  # to new log


def thetis_log_level(level):
    """Set the log level for Thetis logger.

    This controls what level of logging messages are printed to
    stderr. The higher the level, the fewer the number of messages.

    :arg level: The level to use.
    """
    logger = logging.getLogger('thetis')
    logger.setLevel(level)


# logger for error, warning etc messages
logger = logging.getLogger('thetis')
log = logger.log
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

# logger for model output messages with no prefix, used with print_output
output_logger = logging.getLogger('thetis_output')
output_logger.setLevel(INFO)
print_output = output_logger.info
