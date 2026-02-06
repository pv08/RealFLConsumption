import logging

LOGGER_NAME = "logger"
LOGGER = logging.getLogger(LOGGER_NAME)
LOGGER.setLevel(logging.DEBUG)

DEFAULT_FORMATTER = logging.Formatter(
    "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
)

if not LOGGER.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(DEFAULT_FORMATTER)
    LOGGER.addHandler(console_handler)
    LOGGER.propagate = False

logger = logging.getLogger(LOGGER_NAME)
log = logger.log