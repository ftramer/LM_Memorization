import os
import logging
from logging.handlers import TimedRotatingFileHandler


def get_log_object():
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s,%(msecs)3d %(levelname)-8s [%(filename)s:%(lineno)d] - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = TimedRotatingFileHandler('logs/' + os.path.basename(__file__) + '.log', when='midnight', interval=1)
    fh.suffix = '%Y%m%d'
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    log.addHandler(fh)
    log.addHandler(ch)

    return log
