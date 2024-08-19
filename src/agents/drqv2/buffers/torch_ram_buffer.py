import logging
from typing import Tuple

import numpy as np
import torch

from src.agents.drqv2.buffers.replay_buffer import AbstractReplayBuffer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TorchRamBuffer(AbstractReplayBuffer):
    def __init__(self, args):
        assert (
            0 < args.gpu_replay_buffer_size <= args.replay_buffer_size
        ), f"Wrong buffer sizes: {args.gpu_replay_buffer_size} vs {args.replay_buffer_size}"

        self.args = args

        self.buffer_size = args.replay_buffer_size
        # each episode we add 3 frames. Times 2 for action repeat.
        self.buffer_size += (self.buffer_size // 1000) * self.args.frame_stack * self.args.action_repeat
        self.gpu_buffer_size = args.gpu_replay_buffer_size
        self.index = -1
        self.traj_index = 0
        self.discount = args.discount
        self.full = False
        # fixed since we can only sample transitions that occur nstep earlier
        # than the end of each episode or the last recorded observation
        self.discount_vec = torch.pow(args.discount, torch.arange(args.nstep)).to(torch.float32).to(args.device)
        self.next_dis = args.discount**args.nstep

    def _initial_setup(self, time_step):
        self.index = 0
        self.obs_shape = list(time_step.observation.shape)
        logger.debug(f"Observation shape: {self.obs_shape}")
        self.ims_channels = self.obs_shape[0] // self.args.frame_stack
        self.act_shape = time_step.action.shape

        # Arrays are big, so they are stored on GPU and in RAM. Tiny intersection is stored both in ram and on GPU.
        self.gpu_obs_offset = -self.args.frame_stack
        self.gpu_observations = torch.zeros(
            size=[
                self.gpu_buffer_size + self.args.frame_stack + self.args.nstep,
                self.ims_channels,
                *self.obs_shape[1:],
            ],
            dtype=torch.uint8,
            device=self.args.device,
        )

        self.ram_obs_offset = self.gpu_buffer_size - self.args.frame_stack
        self.ram_observations = torch.zeros(
            size=[
                self.buffer_size - self.gpu_buffer_size + self.args.frame_stack + self.args.nstep,
                self.ims_channels,
                *self.obs_shape[1:],
            ],
            dtype=torch.uint8,
            device="cpu",
        )

        # As the size is small, all these arrays are stored in ram.
        self.actions = torch.zeros(
            size=[self.buffer_size, *self.act_shape], dtype=torch.float32, device=self.args.device
        )
        self.rewards = torch.zeros(size=[self.buffer_size], dtype=torch.float32, device=self.args.device)
        self.discounts = torch.zeros(size=[self.buffer_size], dtype=torch.float32, device=self.args.device)

        # which timesteps can be validly sampled (Not within nstep from end of
        # an episode or last recorded observation)
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)

    def add_data_point(self, time_step):
        first = time_step.first()
        latest_obs = time_step.observation[-self.ims_channels :]
        if first:
            # if first observation in a trajectory, record frame_stack copies of it
            end_index = self.index + self.args.frame_stack
            end_invalid = end_index + self.args.frame_stack + 1
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    self.full = True

                self._save_observation_with_frame_stack(latest_obs)
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.index : self.buffer_size] = False
                self.valid[0:end_invalid] = False
            else:
                self._save_observation_with_frame_stack(latest_obs)
                self.valid[self.index : end_invalid] = False
            self.traj_index = 1
        else:
            self._save_observation(latest_obs, self.index)
            self.actions[self.index] = torch.from_numpy(time_step.action).to(self.args.device)
            self.rewards[self.index] = time_step.reward
            self.discounts[self.index] = time_step.discount
            self.valid[(self.index + self.args.frame_stack) % self.buffer_size] = False
            if self.traj_index >= self.args.nstep:
                self.valid[(self.index - self.args.nstep + 1) % self.buffer_size] = True
            self.index += 1
            self.traj_index += 1
            if self.index == self.buffer_size:
                self.index = 0
                self.full = True

    def add(self, time_step):
        if self.index == -1:
            self._initial_setup(time_step)
        self.add_data_point(time_step)

    def __next__(self):
        # sample only valid indices
        indices = np.random.choice(self.valid.nonzero()[0], size=self.args.batch_size)
        indices_of_indices = np.arange(indices.shape[0])
        gpu_indices = indices[indices < self.gpu_buffer_size]
        gpu_indices_of_indices = indices_of_indices[indices < self.gpu_buffer_size]
        ram_indices = indices[indices >= self.gpu_buffer_size]
        ram_indices_of_indices = indices_of_indices[indices >= self.gpu_buffer_size]
        return self.gather_nstep_indices(
            indices, gpu_indices, ram_indices, gpu_indices_of_indices, ram_indices_of_indices
        )

    def _save_observation(self, observation: np.ndarray, index: int):
        # Check, seems ok.
        if index < self.gpu_buffer_size:
            self.gpu_observations[index - self.gpu_obs_offset] = torch.from_numpy(np.copy(observation)).to(
                self.args.device
            )

            if index >= self.ram_obs_offset:
                logger.info(f"Saving observation to ram at index {index}.")
                self._save_observation_to_disk(index, np.copy(observation))
            if index < self.args.nstep:
                logger.info(f"Saving observation to ram at index {index}.")
                self._save_observation_to_disk(self.ram_obs_offset - (self.args.nstep - index), np.copy(observation))
        else:
            self._save_observation_to_disk(index, np.copy(observation))

            if index < self.gpu_buffer_size + self.args.nstep:
                logger.info(f"Saving observation to gpu at index {index}.")
                self.gpu_observations[index - self.gpu_obs_offset] = torch.from_numpy(np.copy(observation)).to(
                    self.args.device
                )
            if index >= self.buffer_size - self.args.frame_stack:
                logger.info(f"Saving observation to gpu at index {index}.")
                self.gpu_observations[(index - self.gpu_obs_offset) % self.buffer_size] = torch.from_numpy(
                    np.copy(observation)
                ).to(self.args.device)

    def _save_observation_with_frame_stack(self, observation: np.ndarray):
        for i in range(self.args.frame_stack):
            self._save_observation(np.copy(observation), self.index + i)
        self.index += self.args.frame_stack
        if self.index >= self.buffer_size:
            self.index = self.index % self.buffer_size
            self.full = True

    def _save_observation_to_disk(self, index: int, observation: np.ndarray):
        self.ram_observations[index - self.ram_obs_offset] = torch.from_numpy(observation).to("cpu")

    def _prepare_gather_indices(self, indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # offsets = np.arange(-self.args.frame_stack, self.args.nstep)
        # stacked_offsets = np.tile(offsets, (len(indices), 1))
        # all_gather_ranges = indices[:, None] + stacked_offsets

        offsets = torch.arange(-self.args.frame_stack, self.args.nstep, dtype=torch.int64)
        stacked_offsets = offsets.repeat(len(indices), 1)
        all_gather_ranges = torch.from_numpy(indices).unsqueeze(1) + stacked_offsets

        gather_ranges = all_gather_ranges[:, self.args.frame_stack :]  # bs x nstep
        obs_gather_ranges = all_gather_ranges[:, : self.args.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.args.frame_stack :]

        logger.debug("Shapes in _prepare_gather_indices")
        logger.debug(
            f"gather_ranges: {gather_ranges.shape}, obs_gather_ranges: {obs_gather_ranges.shape}, "
            f"nobs_gather_ranges: {nobs_gather_ranges.shape}"
        )
        return gather_ranges, obs_gather_ranges, nobs_gather_ranges

    def gather_nstep_indices(
        self,
        indices: np.ndarray,
        gpu_indices: np.ndarray,
        ram_indices: np.ndarray,
        gpu_indices_of_indices: np.ndarray,
        ram_indices_of_indices: np.ndarray,
    ):
        logger.debug(
            f"Gathering nstep indices for {indices.shape[0]} indices."
            f"and {gpu_indices.shape[0]} gpu indices and {ram_indices.shape[0]} ram indices."
        )
        gather_ranges, _, nobs_gather_ranges = self._prepare_gather_indices(indices)

        all_rewards = self.rewards[gather_ranges]
        rew = torch.sum(all_rewards * self.discount_vec, dim=1, keepdim=True)

        # GPU observations
        if gpu_indices.shape[0] > 0:
            _, gpu_obs_gather_ranges, gpu_nobs_gather_ranges = self._prepare_gather_indices(gpu_indices)

            logger.debug(f"gpu_obs_gather_ranges {gpu_obs_gather_ranges}")
            logger.debug(
                f"max gpu_obs_gather_ranges {torch.max(gpu_obs_gather_ranges)} and min {torch.min(gpu_obs_gather_ranges)}"
            )
            gpu_obs = torch.reshape(
                self.gpu_observations[gpu_obs_gather_ranges - self.gpu_obs_offset], [-1, *self.obs_shape]
            )
            gpu_nobs = torch.reshape(
                self.gpu_observations[gpu_nobs_gather_ranges - self.gpu_obs_offset], [-1, *self.obs_shape]
            )
        else:
            gpu_obs = gpu_nobs = None

        # RAM observations
        if ram_indices.shape[0] > 0:
            _, ram_obs_gather_ranges, ram_nobs_gather_ranges = self._prepare_gather_indices(ram_indices)

            logger.debug(f"ram_indices: {ram_indices}")
            logger.debug(
                f"ram_obs_gather_ranges: {ram_obs_gather_ranges.shape}, "
                f"ram_nobs_gather_ranges: {ram_nobs_gather_ranges.shape}"
            )
            logger.debug(f"ram_obs_offset: {self.ram_obs_offset}, ram_observations {self.ram_observations.shape}")
            logger.debug(
                f"max ram_obs_gather_ranges {torch.max(ram_obs_gather_ranges)} and min {torch.min(ram_obs_gather_ranges)}"
            )

            logger.debug(ram_obs_gather_ranges - self.ram_obs_offset)
            ram_obs = torch.reshape(
                self.ram_observations[ram_obs_gather_ranges - self.ram_obs_offset], [-1, *self.obs_shape]
            ).to(self.args.device)

            ram_nobs = torch.reshape(
                self.ram_observations[ram_nobs_gather_ranges - self.ram_obs_offset], [-1, *self.obs_shape]
            ).to(self.args.device)
        else:
            ram_obs = ram_nobs = None

        # Final Observations
        obs = torch.zeros(size=(indices.shape[0], *self.obs_shape), dtype=torch.uint8, device=self.args.device)
        nobs = torch.zeros(size=(indices.shape[0], *self.obs_shape), dtype=torch.uint8, device=self.args.device)

        if gpu_obs is not None:
            obs[gpu_indices_of_indices] = gpu_obs
            nobs[gpu_indices_of_indices] = gpu_nobs

        if ram_obs is not None:
            obs[ram_indices_of_indices] = ram_obs
            nobs[ram_indices_of_indices] = ram_nobs

        assert obs.shape == nobs.shape, f"Shapes {obs.shape} and {nobs.shape} do not match."
        assert obs.shape == (indices.shape[0], *self.obs_shape), (
            f"Shapes {obs.shape} and {(indices.shape[0], *self.obs_shape)} do not " f"match."
        )

        # Actions
        act = self.actions[indices]
        dis = torch.unsqueeze(self.next_dis * self.discounts[nobs_gather_ranges[:, -1]], dim=-1)

        ret = (obs, act, rew, dis, nobs)
        logger.debug(f"Return shapes {[r.shape for r in ret]}")
        return ret

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.index
