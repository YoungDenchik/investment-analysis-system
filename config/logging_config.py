# config/logging_config.py
import logging
import sys

def setup_logging():
    """Set up basic logging configuration."""
    # Set up logging level, format, handlers, etc.
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        stream=sys.stdout
    )
