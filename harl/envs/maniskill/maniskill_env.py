import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import mani_skill.envs  # noqa: F401
from mani_skill.utils.common import flatten_state_dict
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def _t2n(x):
    return x.detach().cpu().numpy()


class ManiSkillEnv:
    def __init__(self, env_args):
        self.env_args = env_args
        self.n_envs = env_args["n_threads"]
        self.env = self.get_env(env_args)

        base_env = self.env.unwrapped
        self.agent_uids = list(base_env.agent.agents_dict.keys())
        self.n_agents = len(self.agent_uids)
        self.device = self.env.device

        obs, _ = self.env.reset()
        per_agent_obs = self._split_obs(obs)
        share_obs = self._flatten_share(obs)
        obs_dim = per_agent_obs.shape[-1]
        share_dim = share_obs.shape[-1]
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
            for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(share_dim,))
            for _ in range(self.n_agents)
        ]
        self.action_space = [
            base_env.agent.agents[i].single_action_space
            for i in range(self.n_agents)
        ]

    def get_env(self, env_args):
        env_kwargs = dict(
            obs_mode="state_dict",
            render_mode="rgb_array",
            sim_backend="physx_cuda",
            control_mode="pd_joint_delta_pos",
        )
        env = gym.make(env_args["task"], num_envs=self.n_envs, **env_kwargs)
        env = ManiSkillVectorEnv(
            env, self.n_envs, ignore_terminations=False, record_metrics=True
        )
        return env

    def _split_obs(self, obs):
        extra = obs["extra"]
        per_agent = []
        for uid in self.agent_uids:
            per_agent.append(
                flatten_state_dict(
                    {"agent": obs["agent"][uid], "extra": extra},
                    use_torch=True,
                    device=self.device,
                )
            )
        return torch.stack(per_agent, dim=1)

    def _flatten_share(self, obs):
        flat = flatten_state_dict(obs, use_torch=True, device=self.device)
        return flat.unsqueeze(1).expand(-1, self.n_agents, -1).contiguous()

    def _build_action(self, actions):
        actions = torch.as_tensor(actions, device=self.device)
        return {uid: actions[:, i] for i, uid in enumerate(self.agent_uids)}

    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(self._build_action(actions))
        per_agent_obs = self._split_obs(obs)
        share_obs = self._flatten_share(obs)
        done = torch.logical_or(term, trunc)
        rew = rew.unsqueeze(-1).unsqueeze(-1).expand(-1, self.n_agents, 1)
        done = done.unsqueeze(-1).expand(-1, self.n_agents)
        return (
            _t2n(per_agent_obs),
            _t2n(share_obs),
            _t2n(rew),
            _t2n(done),
            [[{} for _ in range(self.n_agents)] for _ in range(self.n_envs)],
            [None] * self.n_envs,
        )

    def reset(self):
        obs, _ = self.env.reset()
        per_agent_obs = self._split_obs(obs)
        share_obs = self._flatten_share(obs)
        return _t2n(per_agent_obs), _t2n(share_obs), [None] * self.n_envs

    def seed(self, seed):
        self._seed = seed

    def close(self):
        self.env.close()
