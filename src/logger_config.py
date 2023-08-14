"""
Setup logger
"""
import logging
import sys

import jupedsim as jps

# Create a handler that writes INFO and DEBUG messages to stdout
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)

# Create a handler that writes WARNING, ERROR, and CRITICAL messages to stderr
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)

# Create a formatter and set it for both handlers
formatter = logging.Formatter("%(levelname)s : %(message)s")
stdout_handler.setFormatter(formatter)
stderr_handler.setFormatter(formatter)

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set the lowest level to log for the root logger

# Add both handlers to the logger
logger.addHandler(stdout_handler)
logger.addHandler(stderr_handler)


# Custom log functions
def log_debug(msg: str) -> None:
    """debug messages"""
    logger.debug(msg)


def log_info(msg: str) -> None:
    """info messages"""
    logger.info(msg)


def log_warning(msg: str) -> None:
    """warning messages"""
    logger.warning(msg)


def log_error(msg: str) -> None:
    """errror messages"""
    logger.error(msg)


def init_logger() -> None:
    """init logger"""

    # jps.set_debug_callback(log_debug)
    jps.set_info_callback(log_info)
    jps.set_warning_callback(log_warning)
    jps.set_error_callback(log_error)
