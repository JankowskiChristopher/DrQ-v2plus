import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf, open_dict

from agents.drqv2.buffers.buffer_factory import BufferFactory
from agents.drqv2.buffers.torch_ram_buffer import TorchRamBuffer
from agents.drqv2.drqv2 import DrQV2Agent
from agents.drqv2.utils.dmc import make
from agents.drqv2.utils.logger import Logger
from agents.drqv2.utils.utils import Timer, eval_mode, schedule
from agents.drqv2.utils.video import VideoRecorder
from utils.csv_writer import CSVWriter
from utils.recycler import calculate_dormant_neurons, reinitialize_dormant_neurons

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_agent(obs_spec: np.ndarray, action_spec: np.ndarray, cfg: DictConfig) -> DrQV2Agent:
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return DrQV2Agent(cfg)


class Workspace:
    def __init__(self, args: DictConfig):
        self.work_dir = Path.cwd()

        self.args = args

        # Used to be in a function but just in case seed here
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        torch.backends.cudnn.deterministic = self.args.torch_deterministic
        torch.backends.cudnn.benchmark = not self.args.torch_deterministic

        self.device = torch.device(args.device)
        self.logger = Logger(self.work_dir, use_tb=self.args.use_tb, args=self.args)
        self.train_env = make(self.args.task_name, self.args.frame_stack, self.args.action_repeat, self.args.seed)
        self.eval_env = make(self.args.task_name, self.args.frame_stack, self.args.action_repeat, self.args.seed)

        # Replay buffer and recorder
        buffer_factory = BufferFactory(self.args, self.train_env)
        self.replay_buffer, self.temp_directory = buffer_factory.get_buffer()

        self.video_recorder = VideoRecorder(self.work_dir if self.args.save_video else None)

        self.agent = make_agent(self.train_env.observation_spec(), self.train_env.action_spec(), self.args.agent)

        self.csv_writer = CSVWriter(f"/home/krzysztofj/distributional-sac/csv_drqv2/{args.wandb_group}/{args.seed}.csv",
                                    ["step", "reward", "seed"])
        self.timer = Timer()
        self._global_step = 0
        self._global_episode = 0

        # values to make logging metrics sparse
        self._metrics_log_frequency = args.metrics_log_frequency
        self._metrics_log_counter = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.temp_directory is not None:
            logger.info(f"Removing temp_directory: {self.temp_directory}")
            shutil.rmtree(self.temp_directory)

    @property
    def global_frame(self):
        return self._global_step * self.args.action_repeat

    def reset_dormant_neurons(self, dormant_masks: Dict[str, List[torch.Tensor]], critic_offsets: List[Optional[int]]):
        assert (dormant_masks is not None) and (critic_offsets is not None)
        reinitialize_dormant_neurons(self.agent.encoder, self.agent.encoder_opt, dormant_masks["encoder"])
        for i, (critic_modules, _) in enumerate(self.agent.critic.critic_modules):
            reinitialize_dormant_neurons(
                self.agent.critic,
                self.agent.critic_opt,
                dormant_masks[f"critic_{i}"],
                named_modules=critic_modules,
                offset=critic_offsets[i],
            )
        reinitialize_dormant_neurons(self.agent.actor, self.agent.actor_opt, dormant_masks["actor"])

    def check_dormant_neurons(self) -> Tuple[Dict[str, List[torch.Tensor]], List[Optional[int]]]:
        """
        Function checks the number of dormant neurons in the agent (encoder, actor, critic).
        It also updates the model and the optimizer with the new ones - currently not resetting, so not working,
        """
        dormant_masks = {}

        # hack to change batch size
        original_batch_size = self.replay_buffer.args.batch_size
        self.replay_buffer.args.batch_size = self.args.redo_batch_size
        tau = self.args.redo_tau

        observations, actions, _, _, _ = next(self.replay_buffer)

        # Check dormant neurons in the encoder and save output.
        encoder_outputs_dict = calculate_dormant_neurons((observations,), self.agent.encoder, tau)
        self.logger.log("redo/encoder_dormant_fraction", encoder_outputs_dict["dormant_fraction"], self.global_frame)
        self.logger.log("redo/encoder_zero_dormant", encoder_outputs_dict["zero_fraction"], self.global_frame)
        dormant_masks["encoder"] = encoder_outputs_dict["masks"]

        # Check critic.
        critics_masks = []
        offsets = [None]
        for i, (critic_modules, modules_names) in enumerate(self.agent.critic.critic_modules):
            critic_outputs_dict = calculate_dormant_neurons(
                (encoder_outputs_dict["model_outputs"], actions), self.agent.critic, tau, critic_modules, modules_names
            )
            self.logger.log(
                f"redo/critic_{i}_dormant_fraction", critic_outputs_dict["dormant_fraction"], self.global_frame
            )
            self.logger.log(f"redo/critic_{i}_zero_dormant", critic_outputs_dict["zero_fraction"], self.global_frame)
            critics_masks.append(critic_outputs_dict["masks"])
            if i < len(self.agent.critic.critic_modules) - 1:
                # masks have one layer too little but they include trunk, so it's ok.
                offsets.append(2 * len(critic_outputs_dict["masks"]))

        for i in range(len(self.agent.critic.critic_modules)):
            dormant_masks[f"critic_{i}"] = critics_masks[i]

        # Check actor
        actor_outputs_dict = calculate_dormant_neurons(
            (encoder_outputs_dict["model_outputs"], schedule(self.agent.stddev_schedule, self._global_step)),
            self.agent.actor,
            tau,
        )
        self.logger.log("redo/actor_dormant_fraction", actor_outputs_dict["dormant_fraction"], self.global_frame)
        self.logger.log("redo/actor_zero_dormant", actor_outputs_dict["zero_fraction"], self.global_frame)
        dormant_masks["actor"] = actor_outputs_dict["masks"]

        self.replay_buffer.args.batch_size = original_batch_size

        return dormant_masks, offsets

    def eval(self):
        step, episode, total_reward = 0, 0, 0

        while episode < self.args.num_eval_episodes:
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            stddev = schedule(self.agent.stddev_schedule, self._global_step)
            while not time_step.last():
                with torch.no_grad(), eval_mode(self.agent):
                    action = self.agent.act(time_step.observation, self._global_step, eval_mode=True, stddev=stddev)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f"{self.global_frame}.mp4")

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode", self._global_episode)
            log("step", self._global_step)

        # visualize replay buffer
        if isinstance(self.replay_buffer, TorchRamBuffer) and self._global_step > 0 and self.args.visualize_buffer:
            logger.info(
                f"Visualizing replay buffer. Saving to {self.work_dir / f'mean_observation_{self.global_frame}.png'}"
            )
            start_buffer_index = max(self._global_step - self.args.visualize_replay_buffer_samples, 0)
            end_buffer_index = len(self.replay_buffer)
            # TODO maybe add valid. Works now only for the GPU part of the buffer.
            observations = self.replay_buffer.gpu_observations[start_buffer_index:end_buffer_index] / 255.0
            observations = observations - self.replay_buffer.gpu_observations[0]
            mean_observation = torch.mean(torch.clip(observations, min=0, max=1), dim=0)
            plt.imshow(mean_observation.detach().cpu().numpy().transpose(1, 2, 0))
            plt.axis("off")
            plt.savefig(self.work_dir / f"mean_observation_{self.global_frame}.png")
            plt.close()

        self.csv_writer.add_row({"step": self.global_frame, "reward": total_reward / episode, "seed": self.args.seed})

    def train(self):
        # predicates
        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_buffer.add(time_step)
        metrics, dormant_masks, critic_offsets = None, None, None
        while self._global_step < self.args.num_train_frames // self.args.action_repeat:
            if time_step.last():
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.args.action_repeat

                    with self.logger.log_and_dump_ctx(self.global_frame, ty="train") as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("buffer_size", len(self.replay_buffer))
                        log("step", self._global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_buffer.add(time_step)
                # try to save snapshot
                if self.args.save_snapshot:
                    self.save_snapshot()

                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if self._global_step % (self.args.eval_every_frames // self.args.action_repeat) == 0:
                self.logger.log("eval_total_time", self.timer.total_time(), self.global_frame)
                self.eval()

            # sample action
            stddev = schedule(self.agent.stddev_schedule, self._global_step)
            with torch.no_grad(), eval_mode(self.agent):
                action = self.agent.act(time_step.observation, self._global_step, eval_mode=False, stddev=stddev)

            # try to update the agent
            if self._global_step >= self.args.num_seed_frames // self.args.action_repeat:
                metrics_list, grad_metrics_list = self.agent.update(self.replay_buffer, self._global_step)

                for metrics, grad_metrics in zip(metrics_list, grad_metrics_list, strict=True):
                    self.logger.log_metrics(metrics, self.global_frame, ty=None)
                    self.logger.log_metrics(grad_metrics, self.global_frame, ty=None)

                # Change later
                # if self._metrics_log_counter == 0:
                #     self._metrics_log_counter = self._metrics_log_frequency
                # else:
                #     self._metrics_log_counter -= 1

            # check dormant neurons and optionally reset them
            # TODO check what if not divisible?
            if self._global_step >= self.args.num_seed_frames // self.args.action_repeat:
                if self._global_step % self.args.redo_every_steps == 0:
                    dormant_masks, critic_offsets = self.check_dormant_neurons()

                # null in configs means no reset
                if (self.args.redo_reset_steps is not None) and (self._global_step % self.args.redo_reset_steps == 0):
                    self.reset_dormant_neurons(dormant_masks, critic_offsets)

                if (self.args.hard_reset_steps is not None) and (self._global_step % self.args.hard_reset_steps == 0):
                    self.agent.hard_reset()

            # Now we count global step not including replay ratio.
            self._global_step += 1

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_buffer.add(time_step)
            episode_step += 1

        self.csv_writer.write()  # flush remaining rows

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        keys_to_save = ["agent", "timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        with snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


def run(cfg: Optional[DictConfig] = None):
    GlobalHydra.instance().clear()
    with initialize(config_path="cfgs"):
        args = compose(config_name="config")

    with open_dict(args):
        args.merge_with(cfg)

    logger.info(f"Config arguments {OmegaConf.to_yaml(args)}")
    with Workspace(args) as workspace:
        logger.info(f"Starting DrQv2 with device: {workspace.device}")

        root_dir = Path.cwd()
        snapshot = root_dir / "snapshot.pt"
        if snapshot.exists():
            logger.info(f"resuming: {snapshot}")
            workspace.load_snapshot()

        workspace.train()


if __name__ == "__main__":
    run()
