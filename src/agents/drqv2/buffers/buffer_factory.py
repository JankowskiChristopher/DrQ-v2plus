import logging
import tempfile
from typing import Optional, Tuple

import numpy as np
from dm_env import specs
from omegaconf import DictConfig

from agents.drqv2.buffers.replay_buffer import AbstractReplayBuffer
from agents.drqv2.buffers.torch_ram_buffer import TorchRamBuffer
from agents.drqv2.utils.dmc import ExtendedTimeStepWrapper
from src.agents.drqv2.buffers.dataloader_replay_buffer import DataloaderReplayBuffer
from src.agents.drqv2.buffers.np_replay_buffer import EfficientReplayBuffer
from utils.writer import Writer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BufferFactory:
    def __init__(self, args: DictConfig, env: ExtendedTimeStepWrapper):
        self.args = args
        self.replay_buffer_name = self.args.replay_buffer_name
        self.temp_directory = None
        self.train_env = env

    def get_buffer(self) -> Tuple[AbstractReplayBuffer, Optional[str]]:
        logger.info(f"Creating {self.replay_buffer_name} replay buffer")

        if self.replay_buffer_name == "np_replay_buffer":
            self.temp_directory = tempfile.mkdtemp(dir=Writer.get_entropy_directory())
            logger.info(f"Starting with temp_directory: {self.temp_directory}")

            return EfficientReplayBuffer(self.args, self.temp_directory), self.temp_directory

        elif self.replay_buffer_name == "torch_ram_buffer":
            return TorchRamBuffer(self.args), self.temp_directory

        elif self.replay_buffer_name == "original":
            data_specs = (
                self.train_env.observation_spec(),
                self.train_env.action_spec(),
                specs.Array((1,), np.float32, "reward"),
                specs.Array((1,), np.float32, "discount"),
            )

            return DataloaderReplayBuffer(self.args, data_specs), None

        else:
            raise ValueError(f"Unknown replay buffer: {self.replay_buffer_name}")
