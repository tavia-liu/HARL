
import importlib.util
import pathlib
import sys

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
import mani_skill.envs

# mani_skill.envs is the conda-installed package.  Any custom tasks that live
# in the project's ManiSkill source tree (but were not copied into site-packages)
# must be force-loaded here so that their @register_env decorators run.
_PROJECT_TABLETOP = (
    pathlib.Path(__file__).resolve().parents[4]
    / "ManiSkill" / "mani_skill" / "envs" / "tasks" / "tabletop"
)
for _task_file in _PROJECT_TABLETOP.glob("*.py"):
    _mod_name = f"_project_tabletop.{_task_file.stem}"
    if _mod_name not in sys.modules and _task_file.stem != "__init__":
        try:
            _spec = importlib.util.spec_from_file_location(_mod_name, _task_file)
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
        except Exception:
            pass  # don't break if an unrelated task fails to load


def _t2n(x):
    """
    GPU's torch tensor to CPU's numpy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

class ManiSkillEnv:
    """
    HARL requires each environment to implement this interface:
        class Env:
        def __init__(self, args):
            self.env = ...
            self.n_agents = ...
            self.share_observation_space = ...
            self.observation_space = ...
            self.action_space = ...

        def step(self, actions):
            return obs, state, rewards, dones, info, available_actions

        def reset(self):
            return obs, state, available_actions

        def seed(self, seed):
            pass

        def render(self):
            pass

        def close(self):
            self.env.close()

    Observation layout
    ------------------
    Two modes are supported, controlled by env_args["extra_per_agent"]:

    Mode A — extra_per_agent == 0  (legacy, used by TwoRobotPickCube)
        Flat obs = [agent_0_proprio(P), ..., agent_{N-1}_proprio(P), shared(S)]
        local_obs_i  = proprio_i (P) + shared (S)          shape: P+S
        share_obs_i  = full flat obs                        shape: N*P+S

    Mode B — extra_per_agent > 0  (new, used by MultiPickBall)
        Flat obs = [agent_0_proprio(P), ..., agent_{N-1}_proprio(P),
                    agent_0_extra(E), ..., agent_{N-1}_extra(E)]
        local_obs_i  = proprio_i (P) + extra_i (E)         shape: P+E
        share_obs_i  = full flat obs                        shape: N*(P+E)
    """

    def __init__(self, env_args):

        self.env_args = env_args
        self.env = gym.make(
            env_args["task"],                # e.g. "MultiPickBall-v1"
            num_envs=env_args["n_threads"],  # e.g. 1024
            obs_mode="state",                # flat tensor
        )
        self.n_envs = env_args["n_threads"]

        # ---------- agent count (inferred from action space) ----------
        self.agent_names = list(self.env.action_space.spaces.keys())
        self.n_agents = len(self.agent_names)

        self.per_agent_act_dim = self.env.action_space.spaces[
            self.agent_names[0]
        ].shape[-1]  # e.g. 8 for Panda

        self.total_obs_dim = self.env.observation_space.shape[-1]
        self.proprio_per_agent = env_args.get("proprio_per_agent", 18)  # qpos(9)+qvel(9)
        self.proprio_total = self.proprio_per_agent * self.n_agents

        # ---------- obs split mode ----------
        self.extra_per_agent = env_args.get("extra_per_agent", 0)

        if self.extra_per_agent > 0:
            # Mode B: per-agent private extras
            self.local_obs_dim = self.proprio_per_agent + self.extra_per_agent
        else:
            # Mode A: shared extras (legacy)
            self.shared_dim = self.total_obs_dim - self.proprio_total
            self.local_obs_dim = self.proprio_per_agent + self.shared_dim

        # each agent's actor network input
        self.observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(self.local_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        # critic sees the full global state
        self.share_observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(self.total_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        # each agent's actor output
        self.action_space = [
            Box(low=-1.0, high=1.0, shape=(self.per_agent_act_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

    def step(self, actions):

        # HARL format: numpy (n_envs, n_agents, act_dim)
        # ManiSkill format: dict {agent_name: tensor(n_envs, act_dim)}
        action_dict = {}
        for i, name in enumerate(self.agent_names):
            action_dict[name] = torch.tensor(
                actions[:, i],
                dtype=torch.float32
            )

        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        local_obs, share_obs = self._split_obs(obs)

        reward_np = _t2n(reward)  # (n_envs,)
        rewards = np.stack(
            [reward_np[:, np.newaxis] for _ in range(self.n_agents)], axis=1
        )  # (n_envs, n_agents, 1)

        done_np = _t2n(terminated | truncated).astype(np.float32)  # (n_envs,)
        dones = np.stack(
            [done_np for _ in range(self.n_agents)], axis=1
        )  # (n_envs, n_agents)

        # Propagate "success" from ManiSkill info dict into per-env/per-agent infos
        # ManiSkill returns info["success"] as a boolean tensor of shape (n_envs,)
        success_np = _t2n(info.get("success", np.zeros(self.n_envs, dtype=bool)))
        infos = [
            [{"success": bool(success_np[env_i])} for _ in range(self.n_agents)]
            for env_i in range(self.n_envs)
        ]

        # continuous action space — no action mask needed
        avail = [None] * self.n_envs

        return local_obs, share_obs, rewards, dones, infos, avail

    def reset(self):
        obs, _ = self.env.reset()
        local_obs, share_obs = self._split_obs(obs)
        avail = [None] * self.n_envs
        return local_obs, share_obs, avail

    def seed(self, seed):
        pass

    def close(self):
        self.env.close()

    # ========== helper ==========

    def _split_obs(self, obs):
        """
        Split the flat obs tensor returned by ManiSkill into HARL format.

        Input:
            obs: torch tensor (n_envs, total_obs_dim) on GPU

        Output:
            local_obs:  numpy (n_envs, n_agents, local_obs_dim)
            share_obs:  numpy (n_envs, n_agents, total_obs_dim)
        """
        local_list = []

        if self.extra_per_agent > 0:
            # Mode B — per-agent private extras
            # Layout: [N*proprio | N*extra]
            for i in range(self.n_agents):
                p_start = i * self.proprio_per_agent
                p_end   = p_start + self.proprio_per_agent
                e_start = self.proprio_total + i * self.extra_per_agent
                e_end   = e_start + self.extra_per_agent
                agent_obs = torch.cat(
                    [obs[:, p_start:p_end], obs[:, e_start:e_end]], dim=-1
                )  # (n_envs, local_obs_dim)
                local_list.append(agent_obs)
        else:
            # Mode A — shared extras (legacy TwoRobotPickCube behaviour)
            # Layout: [N*proprio | shared]
            shared = obs[:, self.proprio_total:]
            for i in range(self.n_agents):
                p_start = i * self.proprio_per_agent
                p_end   = p_start + self.proprio_per_agent
                agent_local = torch.cat([obs[:, p_start:p_end], shared], dim=-1)
                local_list.append(agent_local)

        local_obs = _t2n(torch.stack(local_list, dim=1))   # (n_envs, n_agents, local_obs_dim)
        share_obs = _t2n(obs.unsqueeze(1).expand(-1, self.n_agents, -1))  # (n_envs, n_agents, total_obs_dim)

        return local_obs, share_obs
