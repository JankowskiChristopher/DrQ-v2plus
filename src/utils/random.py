import argparse
import random

import numpy as np
import torch
from omegaconf import DictConfig


class Random:
    """
    Class responsible for setting all seeds and other random operations
    :ivar args: the arguments from the argument parser
    """

    def __init__(self, args: DictConfig):
        self.args = args

    def set_all_seeds(self) -> None:
        """
        Function sets all seeds except environent seed
        :return:
        """
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        # As such it seems good practice to turn off cudnn.benchmark when turning on cudnn.deterministic see
        # https://discuss.pytorch.org/t/what-is-the-differenc-between-cudnn-deterministic-and-cudnn-benchmark/38054/3
        torch.backends.cudnn.deterministic = self.args.torch_deterministic
        if self.args.torch_deterministic:
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
