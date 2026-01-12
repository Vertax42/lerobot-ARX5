"""Unified logging configuration using spdlog."""

import spdlog

# Default spdlog pattern with color markers: [HH:MM:SS] [logger_name] [level] message
# %^ and %$ mark the colored region (level name will be colored)
SPDLOG_PATTERN = "[%H:%M:%S] [%n] [%^%l%$] %v"


def get_logger(name: str) -> spdlog.ConsoleLogger:
    """Create a spdlog ConsoleLogger with unified format and colors.

    Args:
        name: Logger name

    Returns:
        Configured spdlog.ConsoleLogger instance with colored output
    """
    logger = spdlog.ConsoleLogger(name, colored=True, multithreaded=True)
    logger.set_pattern(SPDLOG_PATTERN)
    logger.set_level(spdlog.LogLevel.INFO)
    return logger
