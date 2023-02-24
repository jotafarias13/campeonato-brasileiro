import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%Y %H:%M:%S",
)

file_handler = logging.FileHandler(filename="main.log", mode="a")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
