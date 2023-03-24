"""
Setup logger
"""
import logging
import py_jupedsim as jps


def log_debug(msg):
    """debug messages"""

    logging.debug(msg)


def log_info(msg):
    """info messages"""
    logging.info(msg)


def log_warn(msg):
    """warning messages"""
    logging.warning(msg)


def log_error(msg):
    """error messages"""
    logging.error(msg)


logging.basicConfig(level=logging.DEBUG, format="%(levelname)s : %(message)s")


def init_logger():
    # jps.set_debug_callback(log_debug)
    jps.set_info_callback(log_info)
    jps.set_warning_callback(log_warn)
    jps.set_error_callback(log_error)
