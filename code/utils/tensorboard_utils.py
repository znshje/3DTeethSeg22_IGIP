import os.path

from utils.singleton import singleton
from tensorboardX import SummaryWriter
from loguru import logger


@singleton
class TensorboardUtils:
    def __init__(self, output_dir: str):
        self.writer = SummaryWriter(output_dir)
        logger.info('TensorboardUtils inited. Run following command to use tensorboard:\n\ttensorboard --logdir={} --bind_all',
                    os.path.abspath(output_dir))
