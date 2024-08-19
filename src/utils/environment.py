from typing import Callable

import gymnasium as gym


class Environment:
    """
    Utility class for environment creation and management.
    """

    @staticmethod
    def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str) -> Callable[[], gym.Env]:
        """
        Utility function for env creation.
        :param env_id: id of the environment
        :param seed: seed for randomness
        :param idx: idx
        :param capture_video: bool whether capture video or not
        :param run_name: name of the run for logging purposes
        :return: function which creates the environment
        """

        def thunk():
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            return env

        return thunk
