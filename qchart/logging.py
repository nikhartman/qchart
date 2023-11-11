import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from qchart.config import config

def get_log_directory():
    log_directory = Path(config['logging']['directory'])
    log_directory.mkdir(parents=True, exist_ok=True)
    return log_directory

filename = Path(get_log_directory(), 'qchart.log')
logger = logging.getLogger('qchart')
log_handler = RotatingFileHandler(filename, mode='a', 
    maxBytes=1024*16, backupCount=10, encoding='utf-8')
log_handler.setFormatter(
    logging.Formatter(
        '%(asctime)s %(levelname)s: '
        '%(message)s '
        '[in %(pathname)s:%(lineno)d]'
    )
)
log_level = logging.getLevelName(config['logging']['level'])
logger.setLevel(log_level)
logger.addHandler(log_handler)