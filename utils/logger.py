import sys
import torch
import logging
import logging.config

VRAM_LEVEL = 25
logging.addLevelName(VRAM_LEVEL, "VRAM")

class CustomLogger(logging.Logger):
    """A custom logger class with a 'header' method."""
    def header(self, message, *args, **kwargs):
        header_color = '\x1b[1;30;47m'
        reset_color = '\x1b[0m'
        
        padded_message = f'# {message.strip()} '
        banner = f'{header_color}{padded_message}{reset_color}'
        
        print()
        self.info(banner, *args, **kwargs)
        
    def subheader(self, message, *args, **kwargs):
        header_color = '\x1b[1;32m'
        reset_color = '\x1b[0m'
        
        padded_message = f'## {message.strip()}'
        banner = f'{header_color}{padded_message}{reset_color}'
        
        self.info(banner, *args, **kwargs)
        
    def single(self, message, *args, **kwargs):       
        print()
        self.info(message, *args, **kwargs)
        
    def vram(self, message, *args, **kwargs):
        if self.isEnabledFor(VRAM_LEVEL):
            self._log(VRAM_LEVEL, message, args, **kwargs)
        
logging.setLoggerClass(CustomLogger)
  
def setup_logging(log_filepath, vram_only=False):
    """
    Configures logging to save to a custom file in the 'logs' directory.
    """

    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'colored': {
                '()': 'colorlog.ColoredFormatter',
                'format': '%(log_color)s[%(asctime)s][%(levelname)s] - %(message)s%(reset)s',
                'log_colors': {
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                    'VRAM': 'purple',
                },
                'datefmt': '%H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': VRAM_LEVEL if vram_only else 'INFO',
                'formatter': 'colored',
                'stream': sys.stdout,
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'standard',
                'filename': log_filepath,
                'mode': 'w',
            },
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': True
            }
        }
    }

    logging.config.dictConfig(LOGGING_CONFIG)
    if not vram_only:
        root_logger = logging.getLogger('')
        for handler in root_logger.handlers:
            if getattr(handler, 'name', None) == 'console':
                handler.addFilter(lambda record: record.levelno != VRAM_LEVEL)
