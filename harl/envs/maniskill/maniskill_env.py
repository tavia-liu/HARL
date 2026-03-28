import importlib.util
import pathlib
import sys

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
import mani_skill.envs


def _t2n(x):
    """GPU's torch tensor to CPU's numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


class ManiSkillEnv:
    """
    HARL-compatible wrapper for ManiSkill3 multi-agent GPU vectorized envs.

    Key design decisions:
    - ManiSkill GPU envs auto-reset on done and return NEW episode obs,
      but terminated/truncated flags are NOT cleared after auto-reset.
      We track elapsed steps ourselves to produce correct one-shot done signals.
    - HARL expects done=True for exactly ONE step per episode boundary,
      then done=False for the new episode. This is critical for masks/GAE.

    Observation layout documented in the class docstring below.
    """

    def __init__(self, env_args):
        self.env_args = env_args
        self.env = gym.make(
            env_args["task"],
            num_envs=env_args["n_threads"],
            obs_mode="state",
            control_mode=env_args.get("control_mode", "pd_joint_delta_pos"),
            sim_backend="physx_cuda",
        )
        self.n_envs = env_args["n_threads"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- agent count ----------
        self.agent_names = list(self.env.action_space.spaces.keys())
        self.n_agents = len(self.agent_names)

        self.per_agent_act_dim = self.env.action_space.spaces[
            self.agent_names[0]
        ].shape[-1]

        self.total_obs_dim = self.env.observation_space.shape[-1]
        self.proprio_per_agent = env_args.get("proprio_per_agent", 18)
        self.proprio_total = self.proprio_per_agent * self.n_agents

        # ---------- obs split mode ----------
        self.extra_per_agent = env_args.get("extra_per_agent", 0)

        if self.extra_per_agent > 0:
            self.local_obs_dim = self.proprio_per_agent + self.extra_per_agent
        else:
            self.shared_dim = self.total_obs_dim - self.proprio_total
            self.local_obs_dim = self.proprio_per_agent + self.shared_dim

        self.observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(self.local_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(self.total_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.action_space = [
            Box(low=-1.0, high=1.0, shape=(self.per_agent_act_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        # ---------- track elapsed steps for correct done signals ----------
        self._max_episode_steps = env_args.get("episode_length", 100)
        self._elapsed = np.zeros(self.n_envs, dtype=np.int32)

    def step(self, actions):
        # HARL format: numpy (n_envs, n_agents, act_dim)
        # ManiSkill format: dict {agent_name: tensor(n_envs, act_dim)}
        action_dict = {}
        for i, name in enumerate(self.agent_names):
            action_dict[name] = torch.tensor(actions[:, i], dtype=torch.float32, device=self.device)

        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        local_obs, share_obs = self._split_obs(obs)

        reward_np = _t2n(reward)
        rewards = np.stack(
            [reward_np[:, np.newaxis] for _ in range(self.n_agents)], axis=1
        )

        # ============================================================
        # CRITICAL: Generate correct one-shot done signals.
        #
        # ManiSkill GPU env auto-resets on done but does NOT clear
        # terminated/truncated after reset. So after step 49 (done),
        # step 50 also shows terminated=True even though it's a new
        # episode. This breaks HARL's mask/GAE computation.
        #
        # Fix: track elapsed steps ourselves. Done fires once at
        # max_episode_steps, then resets to 0.
        # ============================================================
        self._elapsed += 1
        done_now = (self._elapsed >= self._max_episode_steps).astype(np.float32)

        # Check for early termination (success before max steps)
        terminated_np = _t2n(terminated).astype(bool)
        early_term = terminated_np & (self._elapsed < self._max_episode_steps)
        done_now = np.maximum(done_now, early_term.astype(np.float32))

        # Reset elapsed counter for envs that are done
        is_done = done_now > 0
        is_truncated = is_done & ~early_term  # done by time limit, not early termination

        dones = np.stack([done_now for _ in range(self.n_agents)], axis=1)

        # Reset elapsed for done envs
        self._elapsed[is_done] = 0

        # Build infos
        success_np = _t2n(info.get("success", np.zeros(self.n_envs, dtype=bool)))
        infos = [
            [
                {
                    "success": bool(success_np[env_i]),
                    "bad_transition": bool(is_truncated[env_i]),
                }
                for _ in range(self.n_agents)
            ]
            for env_i in range(self.n_envs)
        ]

        avail = [None] * self.n_envs
        return local_obs, share_obs, rewards, dones, infos, avail

    def reset(self):
        obs, _ = self.env.reset()
        local_obs, share_obs = self._split_obs(obs)
        avail = [None] * self.n_envs
        self._elapsed = np.zeros(self.n_envs, dtype=np.int32)
        return local_obs, share_obs, avail

    def seed(self, seed):
        pass

    def close(self):
        self.env.close()

    # ========== helper ==========

    def _split_obs(self, obs):
        shared = obs[:, self.proprio_total:]
        local_list = []
        for i in range(self.n_agents):
            start = i * self.proprio_per_agent
            end = start + self.proprio_per_agent
            local_list.append(torch.cat([obs[:, start:end], shared], dim=-1))

        local_obs = _t2n(torch.stack(local_list, dim=1))
        share_obs = _t2n(obs.unsqueeze(1).expand(-1, self.n_agents, -1))
        return local_obs, share_obs
