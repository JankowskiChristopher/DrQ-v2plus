from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from agents.drqv2.buffers.replay_buffer import AbstractReplayBuffer
from src.agents.drqv2.utils.utils import TruncatedNormal, schedule, soft_update_params, to_torch, weight_init


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim: int, action_shape: tuple, feature_dim: int, hidden_dim: int, use_ln: bool = False):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

        if use_ln:
            self.policy = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, action_shape[0]),
            )
        else:
            self.policy = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, action_shape[0]),
            )

        self.apply(weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std  # TODO: would be cool to have a learnable std

        dist = TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim: int, action_shape, feature_dim: int, hidden_dim: int, dropout_rate: float = 0.0, use_ln: bool = False):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

        if use_ln:
            self.Q1 = nn.Sequential(
                nn.Linear(feature_dim + action_shape[0], hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, 1),
            )
    
            self.Q2 = nn.Sequential(
                nn.Linear(feature_dim + action_shape[0], hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.Q1 = nn.Sequential(
                nn.Linear(feature_dim + action_shape[0], hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, 1),
            )
    
            self.Q2 = nn.Sequential(
                nn.Linear(feature_dim + action_shape[0], hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, 1),
            )

        self.apply(weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)

        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

    @property
    def critic_modules(self) -> List[Tuple[List[Tuple[str, nn.Module]], List[str]]]:
        extended_critic1 = list(self.trunk.named_modules()) + list(self.Q1.named_modules())
        extended_critic2 = list(self.trunk.named_modules()) + list(self.Q2.named_modules())

        trunk_names = [name for name, _ in self.named_modules() if name.startswith("trunk")]
        Q1_names = [name for name, _ in self.named_modules() if name.startswith("Q1")]
        Q2_names = [name for name, _ in self.named_modules() if name.startswith("Q2")]

        return [(extended_critic1, trunk_names + Q1_names), (extended_critic2, trunk_names + Q2_names)]


class DrQV2Agent:
    def __init__(self, args: DictConfig):
        self.args = args
        self.device: str = args.device
        self.critic_target_tau: float = args.critic_target_tau
        self.update_every_steps: int = args.update_every_steps
        self.use_tb: bool = args.use_tb
        self.num_expl_steps: int = args.num_expl_steps
        self.stddev_schedule: str = args.stddev_schedule
        self.stddev_clip: float = args.stddev_clip
        self.replay_ratio: int = args.replay_ratio
        self.use_ln : bool = args.use_ln

        # models
        self.encoder = Encoder(args.obs_shape).to(self.device)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=args.lr)
        self.encoder.train()

        self._init_actor()
        self._init_critic()

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.training = True

    def _init_actor(self):
        self.actor = Actor(
            self.encoder.repr_dim, self.args.action_shape, self.args.feature_dim, self.args.hidden_dim, use_ln=self.use_ln
        ).to(self.device)
        self.actor.train()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr)

    def _init_critic(self):
        self.critic = Critic(
            self.encoder.repr_dim,
            self.args.action_shape,
            self.args.feature_dim,
            self.args.hidden_dim,
            self.args.critic_dropout,
            use_ln=self.use_ln
        ).to(self.device)
        self.critic_target = Critic(
            self.encoder.repr_dim, self.args.action_shape, self.args.feature_dim, self.args.hidden_dim, use_ln=self.use_ln
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)
        self.critic.train()
        self.critic_target.train()

    def train(self, training: bool = True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs: np.ndarray, step: int, eval_mode: bool, stddev: float):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def shrink_and_perturb(self):
        """
        Performs soft reset of encoder, actor and critic.
        To be decided if fully connected layers should be hard reset.
        :return:
        """
        self._soft_reset_model(self.encoder, self.args.soft_reset_alpha)
        self._init_actor()
        self._init_critic()
        
    def hard_reset(self):
        self._init_actor()
        self._init_critic()

    def _soft_reset_model(self, net: nn.Module, alpha: float):
        for name, module in net.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                weight_shape = module.weight.data.shape
                module.weight.data = module.weight.data * alpha + (1 - alpha) * torch.randn(
                    weight_shape, device=self.device
                )
                bias_shape = module.bias.data.shape
                module.bias.data = module.bias.data * alpha + (1 - alpha) * torch.randn(bias_shape, device=self.device)

    def calculate_grad_statistics(
        self, module_name: str, agent_module: Union[nn.Module, Tuple[List[Tuple[str, nn.Module]], List[str]]]
    ) -> Dict[str, float]:
        grad_metrics = dict()
        grads_list = []
        agent_named_modules = agent_module[0] if isinstance(agent_module, tuple) else agent_module.named_modules()
        for name, module in agent_named_modules:
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                for _, param in module.named_parameters():
                    grads_list.append(param.grad.flatten())

        grads = torch.cat(grads_list)
        grad_metrics[f"{module_name}/grad_mean"] = grads.mean().item()
        grad_metrics[f"{module_name}/grad_std"] = grads.std().item()
        grad_metrics[f"{module_name}/grad_max"] = grads.max().item()
        grad_metrics[f"{module_name}/grad_min"] = grads.min().item()

        grad_metrics[f"{module_name}/grad_l1"] = torch.norm(grads, 1).item()
        grad_metrics[f"{module_name}/grad_l2"] = torch.norm(grads, 2).item()

        return grad_metrics

    def update_critic(self, obs, action, reward, discount, next_obs, step) -> Tuple[Dict[str, float], Dict[str, float]]:
        metrics = dict()
        grad_metrics = dict()

        with torch.no_grad():
            stddev = schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics["train/critic_target_q"] = target_Q.mean().item()
            metrics["train/critic_q1"] = Q1.mean().item()
            metrics["train/critic_q2"] = Q2.mean().item()
            metrics["train/critic_q1_std"] = Q1.std().item()
            metrics["train/critic_q2_std"] = Q2.std().item()
            metrics["train/critic_loss"] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.use_tb:
            critics_grad_metrics = [
                self.calculate_grad_statistics(f"critic_{i}", critic)
                for (i, critic) in enumerate(self.critic.critic_modules)
            ]
            for critic_grad_metrics in critics_grad_metrics:
                grad_metrics.update(critic_grad_metrics)
            grad_metrics.update(self.calculate_grad_statistics("encoder", self.encoder))

        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics, grad_metrics

    def update_actor(self, obs: torch.Tensor, step: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        metrics, grad_metrics = dict(), dict()

        stddev = schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.use_tb:
            grad_metrics.update(self.calculate_grad_statistics("actor", self.actor))

        self.actor_opt.step()

        if self.use_tb:
            metrics["train/actor_loss"] = actor_loss.item()
            metrics["train/actor_logprob"] = log_prob.mean().item()
            metrics["train/actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics, grad_metrics

    def update(
        self, replay_iter: AbstractReplayBuffer, step: int
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        metrics, grad_metrics = dict(), dict()

        if step % self.update_every_steps != 0:
            return [metrics], [grad_metrics]

        metrics_results, grad_metrics_results = [], []
        for i in range(self.replay_ratio):
            # Sample from the replay buffer
            if self.args.replay_buffer_name == "torch_ram_buffer":
                obs, action, reward, discount, next_obs = next(replay_iter)
            else:
                batch = next(replay_iter)
                obs, action, reward, discount, next_obs = to_torch(batch, self.device)

            # augment
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
            # encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

            # update critic and encoder
            critic_and_enc_metrics, grad_critic_and_enc_metrics = self.update_critic(
                obs, action, reward, discount, next_obs, step + i
            )
            metrics.update(critic_and_enc_metrics)
            grad_metrics.update(grad_critic_and_enc_metrics)

            # update actor
            actor_metrics, grad_actor_metrics = self.update_actor(obs.detach(), step + i)
            metrics.update(actor_metrics)
            grad_metrics.update(grad_actor_metrics)

            # update critic target
            soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

            metrics_results.append(metrics.copy())
            grad_metrics_results.append(grad_metrics.copy())

        return metrics_results, grad_metrics_results
