import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
import colorlog


class Logger:
    def __init__(self, module_name: str, log_level: str = "INFO", max_log_size: int = 5 * 1024 * 1024):
        self.module_name = module_name
        self.log_dir = os.path.join("logs", module_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Standard log format with colors
        log_format = "%(log_color)s[%(levelname)s]%(reset)s | %(bold)s%(module)s%(reset)s | %(asctime)s | %(message)s"
        formatter = colorlog.ColoredFormatter(
            log_format,
            secondary_log_colors={
                'asctime': {'color': 'white'},  # Ensure timestamp remains grey
            },
            datefmt="%Y-%m-%d %H:%M:%S"  # Log date and time till seconds
        )


        # JSON log format
        json_formatter = jsonlogger.JsonFormatter(
            "%(levelname)s %(module)s %(asctime)s %(message)s")

        # Create log file handler
        log_file = os.path.join(self.log_dir, f"{module_name}.log")
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_log_size, backupCount=5)
        file_handler.setFormatter(logging.Formatter(
            "[%(levelname)s] | %(module)s | %(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))

        # Create JSON log file handler
        json_log_file = os.path.join(self.log_dir, f"{module_name}.json")
        json_handler = RotatingFileHandler(
            json_log_file, maxBytes=max_log_size, backupCount=5)
        json_handler.setFormatter(json_formatter)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        # Configure logger
        self.logger = logging.getLogger(module_name)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self.logger.addHandler(file_handler)
        self.logger.addHandler(json_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger


# Example usage
if __name__ == "__main__":
    logger = Logger(__name__).get_logger()
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")