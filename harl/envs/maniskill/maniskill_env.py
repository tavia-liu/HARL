import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import mani_skill.envs  # noqa: F401
from mani_skill.utils.common import flatten_state_dict
from mani_skill.utils import gym_utils
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
            obs_mode=env_args.get("obs_mode", "state_dict"),
            render_mode="rgb_array",
            sim_backend="gpu",
            control_mode=env_args.get("control_mode", "pd_joint_delta_pos"),
            reward_mode=env_args.get("reward_mode", "normalized_dense"),
        )
        env = gym.make(env_args["task"], num_envs=self.n_envs, **env_kwargs)
        if env_args.get("record_video"):
            from mani_skill.utils.wrappers.record import RecordEpisode
            video_dir = env_args.get("video_dir", f"eval_videos/{env_args['task']}")
            env = RecordEpisode(
                env, output_dir=video_dir, save_trajectory=False,
                max_steps_per_video=gym_utils.find_max_episode_steps_value(env),
                video_fps=30,
            )
        partial_reset = env_args.get("partial_reset", True)
        env = ManiSkillVectorEnv(
            env, self.n_envs, ignore_terminations=not partial_reset, record_metrics=True,
            partial_reset=partial_reset,
        )
        return env

    def _split_obs(self, obs):
        extra = obs["extra"]
        per_agent = []
        for uid in self.agent_uids:
            flat = flatten_state_dict(
                {"agent": obs["agent"][uid], "extra": extra},
                use_torch=True,
                device=self.device,
            )
            per_agent.append(torch.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=-1.0))
        return torch.stack(per_agent, dim=1)

    def _flatten_share(self, obs):
        flat = flatten_state_dict(obs, use_torch=True, device=self.device)
        flat = torch.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=-1.0)
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
        infos = [[{} for _ in range(self.n_agents)] for _ in range(self.n_envs)]
        if "final_info" in info:
            mask = info["_final_info"]
            episode = info["final_info"]["episode"]
            for env_i in range(self.n_envs):
                if mask[env_i]:
                    ep = {k: float(v[env_i]) for k, v in episode.items()}
                    for agent_i in range(self.n_agents):
                        infos[env_i][agent_i]["episode"] = ep
        return (
            _t2n(per_agent_obs),
            _t2n(share_obs),
            _t2n(rew),
            _t2n(done),
            infos,
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
