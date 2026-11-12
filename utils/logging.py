import logging
import os 
import sys 

def init_logging(logger, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logfile= "training.log"
    logger.logger.handlers = []
    logger.logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("(%(asctime)s): %(message)s", datefmt='%Y-%m-%d %H:%M')
    
    handler_file = logging.FileHandler(os.path.join(output_dir, logfile))
    handler_file.setFormatter(formatter)
    logger.logger.addHandler(handler_file)

    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    logger.logger.addHandler(handler_stream)
