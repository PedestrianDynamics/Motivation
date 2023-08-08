"""
Setup logger
"""
import logging
import jupedsim as jps


def log_debug(msg: str) -> None:
    """debug messages"""

    logging.debug(msg)


def log_info(msg: str) -> None:
    """info messages"""
    logging.info(msg)


def log_warning(msg: str) -> None:
    """warning messages"""
    logging.warning(msg)


def log_error(msg: str) -> None:
    """error messages"""
    logging.error(msg)


logging.basicConfig(level=logging.DEBUG, format="%(levelname)s : %(message)s")


def init_logger() -> None:
    """init logger"""

    # jps.set_debug_callback(log_debug)
    jps.set_info_callback(log_info)
    jps.set_warning_callback(log_warning)
    jps.set_error_callback(log_error)
