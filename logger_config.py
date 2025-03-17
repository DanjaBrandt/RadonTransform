import logging
import os
from utils import functions


def setup_logger(mode):
    """
    Configures and returns a logger with a unique log file name based on the processing mode.

    :param mode: The mode of operation (e.g., 'radon', 'fourier', 'display')
    :return: Configured logger
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Generate a unique log file name based on the mode
    log_file = functions.get_unique_name(os.path.join(log_dir, f"{mode}_log"), extension=".log")

    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    return logging.getLogger(mode)
