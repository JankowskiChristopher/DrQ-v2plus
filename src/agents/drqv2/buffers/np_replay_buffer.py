import logging
from typing import Dict, List, Tuple

import numpy as np

from src.agents.drqv2.buffers.replay_buffer import AbstractReplayBuffer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EfficientReplayBuffer(AbstractReplayBuffer):
    def __init__(self, args, temp_dir: str, data_specs=None):
        logger.info(f"Using EfficientReplayBuffer with temp_dir: {temp_dir}")

        assert (
            0 < args.ram_replay_buffer_size <= args.replay_buffer_size
        ), f"Wrong buffer sizes: {args.ram_buffer_size} vs {args.replay_buffer_size}"
        self.args = args
        self.temp_dir = temp_dir

        self.buffer_size = args.replay_buffer_size
        self.ram_buffer_size = args.ram_replay_buffer_size
        self.index = -1
        self.traj_index = 0
        self._recorded_frames = args.frame_stack + 1
        self.discount = args.discount
        self.full = False
        # fixed since we can only sample transitions that occur nstep earlier
        # than the end of each episode or the last recorded observation
        self.discount_vec = np.power(args.discount, np.arange(args.nstep)).astype("float32")
        self.next_dis = args.discount**args.nstep

    def _initial_setup(self, time_step):
        self.index = 0
        self.obs_shape = list(time_step.observation.shape)
        logger.debug(f"Observation shape: {self.obs_shape}")
        self.ims_channels = self.obs_shape[0] // self.args.frame_stack
        self.act_shape = time_step.action.shape

        # Arrays are big, so they are stored in ram and on disk. Tiny intersection is stored both in ram and on disk.
        self.ram_observations = np.empty(
            [self.ram_buffer_size + self.args.nstep, self.ims_channels, *self.obs_shape[1:]], dtype=np.uint8
        )

        # As the size is small, all these arrays are stored in ram.
        self.actions = np.empty([self.buffer_size, *self.act_shape], dtype=np.float32)
        self.rewards = np.empty([self.buffer_size], dtype=np.float32)
        self.discounts = np.empty([self.buffer_size], dtype=np.float32)
        # Rest is stored on disk, no need for an array as index is enough.

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
            self._save_observation(latest_obs)
            np.copyto(self.actions[self.index], time_step.action)
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
        ram_indices = indices[indices < self.ram_buffer_size]
        disk_indices = indices[indices >= self.ram_buffer_size]
        return self.gather_nstep_indices(indices, ram_indices, disk_indices)

    def _get_observations_from_disk(self, all_ranges: List[np.ndarray]) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        As some observations are stored on disk, fetch them. The observations are stored in a numpy array but
        can be accessed using numpy indexes therefore also a translation dict is returned, so that the same indexes
        as in ram can be used.
        :param all_ranges: list of numpy arrays with the ranges to fetch
        :return: tuple of numpy array with the observations and a translation dict
        """
        # Potential optimization: many workers and maybe more group in nstep.
        # Consider compression if IO is slow.
        unique_values = np.unique(all_ranges)
        observations_on_disk = np.empty(
            shape=[unique_values.shape[0], self.ims_channels, *self.obs_shape[1:]], dtype=np.uint8
        )
        translation_dict = {}
        for i, index in enumerate(unique_values):
            observation = self._get_observation_from_disk(index)
            observations_on_disk[i] = observation
            translation_dict[index] = i

        logger.debug(f"Fetched {observations_on_disk.shape} observations from disk.")

        return observations_on_disk, translation_dict

    def _get_observation_from_disk(self, index: int):
        return np.load(f"{self.temp_dir}/{index}.npy")

    def _save_observation(self, observation: np.ndarray):
        if self.index < self.ram_buffer_size:
            np.copyto(self.ram_observations[self.index], observation)
        else:
            self._save_observation_to_disk(self.index, observation)
        # Save tiny intersection in ram and on disk.
        if self.ram_observations.shape[0] > self.index >= (self.ram_buffer_size - self.args.frame_stack):
            np.copyto(self.ram_observations[self.index], observation)
            self._save_observation_to_disk(self.index, observation)

    def _save_observation_with_frame_stack(self, observation: np.ndarray):
        for i in range(self.args.frame_stack):
            self._save_observation(observation)
            self.index = (self.index + 1) % self.buffer_size

    def _save_observation_to_disk(self, index: int, observation: np.ndarray):
        logger.debug(f"Saving observation to disk: {index}")
        np.save(f"{self.temp_dir}/{index}.npy", observation)

    def _prepare_gather_indices(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples = indices.shape[0]
        all_gather_ranges = (
            np.stack(
                [np.arange(indices[i] - self.args.frame_stack, indices[i] + self.args.nstep) for i in range(n_samples)],
                axis=0,
            )
            % self.buffer_size
        )
        gather_ranges = all_gather_ranges[:, self.args.frame_stack :]  # bs x nstep
        obs_gather_ranges = all_gather_ranges[:, : self.args.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.args.frame_stack :]  # TODO check with Michał if not nstep?

        logger.debug("Shapes in _prepare_gather_indices")
        logger.debug(
            f"gather_ranges: {gather_ranges.shape}, obs_gather_ranges: {obs_gather_ranges.shape}, "
            f"nobs_gather_ranges: {nobs_gather_ranges.shape}"
        )
        return gather_ranges, obs_gather_ranges, nobs_gather_ranges

    def gather_nstep_indices(self, indices: np.ndarray, ram_indices: np.ndarray, disk_indices: np.ndarray):
        logger.debug(
            f"Gathering nstep indices for {indices.shape[0]} indices."
            f"and {ram_indices.shape[0]} ram indices and {disk_indices.shape[0]} disk indices."
        )
        gather_ranges, _, nobs_gather_ranges = self._prepare_gather_indices(indices)

        all_rewards = self.rewards[gather_ranges]
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        # RAM observations
        _, ram_obs_gather_ranges, ram_nobs_gather_ranges = self._prepare_gather_indices(ram_indices)
        ram_obs = np.reshape(self.ram_observations[ram_obs_gather_ranges], [-1, *self.obs_shape])
        ram_nobs = np.reshape(self.ram_observations[ram_nobs_gather_ranges], [-1, *self.obs_shape])
        logger.debug(f"ram_obs: {ram_obs.shape}, ram_nobs: {ram_nobs.shape}")

        # Disk observations
        if disk_indices.shape[0] > 0:
            _, disk_obs_gather_ranges, disk_nobs_gather_ranges = self._prepare_gather_indices(disk_indices)
            disk_observations, index_translation_dict = self._get_observations_from_disk(
                [disk_obs_gather_ranges, disk_nobs_gather_ranges]
            )
            logger.debug(f"disk_observations: {disk_observations.shape}")

            # Concatenate RAM and disk observations
            disk_obs_gather_ranges = np.vectorize(index_translation_dict.get)(disk_obs_gather_ranges)
            disk_nobs_gather_ranges = np.vectorize(index_translation_dict.get)(disk_nobs_gather_ranges)

            disk_obs = np.reshape(disk_observations[disk_obs_gather_ranges], [-1, *self.obs_shape])
            disk_nobs = np.reshape(disk_observations[disk_nobs_gather_ranges], [-1, *self.obs_shape])

            obs = np.concatenate([ram_obs, disk_obs], axis=0)
            nobs = np.concatenate([ram_nobs, disk_nobs], axis=0)
        else:
            logger.debug("No observations from disk, using only ram observations.")
            obs = ram_obs
            nobs = ram_nobs

        # Actions
        act = self.actions[indices]
        dis = np.expand_dims(self.next_dis * self.discounts[nobs_gather_ranges[:, -1]], axis=-1)

        ret = (obs, act, rew, dis, nobs)
        logger.debug(f"Return shapes {[r.shape for r in ret]}")
        return ret

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.index
