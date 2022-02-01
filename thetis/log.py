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
import io

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
    mem_buffer = None


thetis_log_config = ThetisLogConfig()


class BufferHandler(logging.StreamHandler):
    pass


def set_thetis_loggers(comm=COMM_WORLD):
    """Set stream handlers for log messages.

    :kwarg comm: The communicator the handler should be collective
         over. Only rank-0 on that communicator will write to the log, other
         ranks will use a :class:`logging.NullHandler`.
    """
    def add_stream_handler(buffered=False):
        if comm is None or comm.rank == 0:
            if buffered:
                handler = BufferHandler(thetis_log_config.mem_buffer)
            else:
                handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(fmt=fmt))
        else:
            handler = logging.NullHandler()
        logger.addHandler(handler)

    if thetis_log_config.mem_buffer is None:
        thetis_log_config.mem_buffer = io.StringIO()

    for name, fmt in logger_format.items():
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)

        add_stream_handler()
        add_stream_handler(buffered=True)


def set_log_directory(output_directory, comm=COMM_WORLD, mode='w'):
    """
    Forward all log output to `output_directory/log` file.

    When called, a new empty log file is created.

    If called twice with a different `output_directory`, a warning is raised,
    and the new log file location is assigned. The old log file or the
    `output_directory` are not removed.

    :arg output_directory: the directory where log file is stored
    :kwarg comm: The communicator the handler should be collective
         over. Only rank-0 on that communicator will write to the log, other
         ranks will use a :class:`logging.NullHandler`.
    :kwarg mode: write mode, 'w' removes previous log file (if any), otherwise
        appends to it. Default: 'w'.
    """
    def create_directory(dir):
        if comm.rank == 0:
            if os.path.exists(dir):
                if not os.path.isdir(dir):
                    raise IOError('file with same name exists', dir)
            else:
                os.makedirs(dir)

    def rm_handlers(cls=logging.FileHandler):
        for name in logger_format:
            logger = logging.getLogger(name)
            for handler in logger.handlers:
                if isinstance(handler, cls):
                    logger.removeHandler(handler)

    def rm_file_handlers():
        rm_handlers(cls=logging.FileHandler)

    def rm_buf_handlers():
        rm_handlers(cls=BufferHandler)

    def assign_file_handler(logfile):

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
    different_file = thetis_log_config.filename is not None
    if different_file:
        old_file = str(thetis_log_config.filename)
        rm_file_handlers()
    if mode == 'w' and os.path.isfile(logfile):
        # silently remove previous log
        if comm.rank == 0:
            os.remove(logfile)
    create_directory(output_directory)
    if comm.rank == 0:
        buffer_content = thetis_log_config.mem_buffer.getvalue()
        with open(logfile, 'w') as f:
            f.write(buffer_content)
    rm_buf_handlers()
    thetis_log_config.filename = logfile
    assign_file_handler(logfile)
    if different_file:
        msg = (f'Setting a log file "{logfile}" that differs from previous '
               f'"{old_file}", removing old handler')
        warning(msg)  # to new log


def thetis_log_level(level):
    """Set the log level for Thetis logger.

    This controls what level of logging messages are printed to
    stderr. The higher the level, the fewer the number of messages.

    :arg level: The level to use, one of 'DEBUG', 'INFO', 'WARNING', 'ERROR',
        'CRITICAL'.
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
