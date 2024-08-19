# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import logging
import time
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from src.networks.actor import Actor
from src.networks.critic import SoftQNetwork
from src.utils.environment import Environment
from src.utils.exceptions import Exceptions
from src.utils.random import Random
from src.utils.writer import Writer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SAC:
    """
    Implementation of SAC algorithm.
    :ivar actor: actor network
    :ivar qf1: first critic network
    :ivar qf2: second critic network
    :ivar qf1_target: target critic network
    :ivar qf2_target: target critic network
    :ivar q_optimizer: optimizer for critic network
    :ivar actor_optimizer: optimizer for actor network
    :ivar rb: replay buffer
    :ivar alpha: entropy coefficient
    :ivar a_optimizer: optimizer for entropy coefficient
    :ivar device: device to run the algorithm on
    :ivar writer: tensorboard/wandb writer
    :ivar args: arguments from parser
    :ivar envs: environment(s)
    """

    def __init__(
        self,
        actor: Actor,
        critics: List[nn.Module],
        target_critics: List[nn.Module],
        rb: ReplayBuffer,
        args: DictConfig,
        writer: SummaryWriter,
        device,
        envs,
    ):
        self.actor = actor
        self.qf1, self.qf2 = critics
        self.qf1_target, self.qf2_target = target_critics

        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr)
        self.actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

        self.rb = rb

        # Automatic entropy tuning
        if args.autotune:
            # alpha must be scaled and the target is based on the action space dimension
            self.target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            # alpha is in range [0, inf] so we optimize log_alpha
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.alpha = args.alpha
            self.log_alpha = None
            self.a_optimizer = None

        self.device = device
        self.writer = writer
        self.args = args
        self.envs = envs

    def run(self) -> None:
        """
        Function responsible for running the SAC algorithm and monitoring the performance of training.
        """
        start_time = time.time()
        logger.info(f"Starting SAC training at {start_time} with device {self.device}")

        obs, _ = self.envs.reset(seed=self.args.seed)
        for global_step in range(self.args.total_timesteps):
            # For first n steps, actions are sampled.
            if global_step < self.args.learning_starts:
                actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            # After n steps actions are taken from the actor.
            else:
                actions, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            # terminations mean that the episode ended, truncations that the agent could have continued but the env
            # ended the execution of the agent.
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)

            # Record rewards for plotting purposes
            info = infos.get("final_info", None)
            info = info[0] if info is not None and len(info) == 1 else None
            if info is not None:
                logger.info(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]

            # original version had infos which gave mypy warnings
            self.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.args.learning_starts:
                # sample minibatch from replay buffer
                data = self.rb.sample(self.args.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
                    # Compute 2 target Q-values
                    qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
                    # Trick with minimum - turns out it is mean - std.
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                    # Rolling average
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.args.gamma * min_qf_next_target.view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                if global_step % self.args.policy_frequency == 0:  # TD 3 Delayed update support
                    # compensate for the delay by doing 'actor_update_interval' instead of 1
                    for _ in range(self.args.policy_frequency):
                        pi, log_pi, _ = self.actor.get_action(data.observations)
                        qf1_pi = self.qf1(data.observations, pi)
                        qf2_pi = self.qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        # Optimization of alpha coefficient
                        if self.args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action(data.observations)
                            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                            self.a_optimizer.zero_grad()
                            alpha_loss.backward()
                            self.a_optimizer.step()
                            self.alpha = self.log_alpha.exp().item()

                # update the target networks
                if global_step % self.args.target_network_frequency == 0:
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

                # logging
                if global_step % 100 == 0:
                    logger.info(f"SPS: {int(global_step / (time.time() - start_time))}")
                    self._write_logs_periodical(
                        {
                            "losses/qf1_values": qf1_a_values.mean().item(),
                            "losses/qf2_values": qf2_a_values.mean().item(),
                            "losses/qf1_loss": qf1_loss.item(),
                            "losses/qf2_loss": qf2_loss.item(),
                            "losses/qf_loss": qf_loss.item() / 2.0,
                            "losses/actor_loss": actor_loss.item(),
                            "losses/alpha": self.alpha,
                            "charts/SPS": int(global_step / (time.time() - start_time)),
                        },
                        global_step,
                    )
                    if self.args.autotune:
                        self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        self.envs.close()
        self.writer.close()

    def _write_logs_periodical(self, log_dict: Dict[str, Any], global_step: int) -> None:
        """
        Function responsible for writing logs to tensorboard periodically.
        :param log_dict: dictionary with keys as log names and values as log values
        :param global_step: step number
        """
        for key, value in log_dict.items():
            self.writer.add_scalar(key, value, global_step)


# @hydra.main(config_path="cfgs", config_name="config")
def run(cfg: Optional[DictConfig] = None):
    GlobalHydra.instance().clear()
    with initialize(config_path="cfgs"):
        args = compose(config_name="config")

    with open_dict(args):
        args.merge_with(cfg)

    Exceptions.stable_baselines3_version_exception()

    writer_obj = Writer(args)
    writer = writer_obj.writer
    run_name = writer_obj.run_name

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random_setter = Random(args)
    random_setter.set_all_seeds()

    # env setup
    envs = gym.vector.SyncVectorEnv([Environment.make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    envs.single_observation_space.dtype = np.float32

    actor = Actor(envs).to(device)
    critics = [SoftQNetwork(envs).to(device), SoftQNetwork(envs).to(device)]
    target_critics = [SoftQNetwork(envs).to(device), SoftQNetwork(envs).to(device)]

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    sac_agent = SAC(actor, critics, target_critics, rb, args, writer, device, envs)
    sac_agent.run()


if __name__ == "__main__":
    run()
