
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
import mani_skill.envs


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
    """

    def __init__(self, env_args):

        self.env_args = env_args
        self.env = gym.make(
            env_args["task"],                # "TwoRobotPickCube-v1"
            num_envs=env_args["n_threads"],  # 1024
            obs_mode="state",                # flat tensor (1024, 66)
        )
        self.n_envs = env_args["n_threads"]
        self.n_agents = 2
        self.agent_names = list(self.env.action_space.spaces.keys())
        self.per_agent_act_dim = self.env.action_space.spaces[
            self.agent_names[0]
        ].shape[-1]  # 8
        self.total_obs_dim = self.env.observation_space.shape[-1]  # 66
        self.proprio_per_agent = 18   # qpos(9) + qvel(9)
        self.proprio_total = self.proprio_per_agent * self.n_agents  # 36
        self.shared_dim = self.total_obs_dim - self.proprio_total   # 30
        self.local_obs_dim = self.proprio_per_agent + self.shared_dim  # 48

        #  each agent's actor network input is 48 dimensions
        self.observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(self.local_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        # each agent's critic network input is 66 dimensions
        # critic look at global information, actor only look at local information.
        self.share_observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(self.total_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        # each agent's actor network output is 8 dimensions with range [-1, 1] 
        self.action_space = [
            Box(low=-1.0, high=1.0, shape=(self.per_agent_act_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

    def step(self, actions):

        #convert HARL action format to ManiSkill3 format 
        # HARL format: numpy (1024, 2, 8)
        # ManiSkill3 format: dict {"panda_wristcam-0": tensor(1024,8), "panda_wristcam-1": tensor(1024,8)}
        action_dict = {}
        for i, name in enumerate(self.agent_names):
            action_dict[name] = torch.tensor(
                actions[:, i], 
                dtype=torch.float32
            )  # (1024, 8)

        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        local_obs, share_obs = self._split_obs(obs)
        # local_obs: numpy (1024, 2, 48)
        # share_obs: numpy (1024, 2, 66)
        
        reward_np = _t2n(reward)  # (1024,)
        rewards = np.stack(
            [reward_np[:, np.newaxis] for _ in range(self.n_agents)], axis=1
        )  # (1024, 2, 1)
        
        done_np = _t2n(terminated | truncated).astype(np.float32)  # (1024,)
        dones = np.stack(
            [done_np for _ in range(self.n_agents)], axis=1
        )  # (1024, 2)


        infos = [[{} for _ in range(self.n_agents)] for _ in range(self.n_envs)]

        # continuous action space does not need action mask, return None
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

    # ========== helper function==========

    def _split_obs(self, obs):
        """
        Split the flat obs returned by ManiSkill3 into the format required by HARL.
        Input:
            obs: torch tensor (n_envs, 66) on GPU
        
        Output:
            local_obs: numpy (n_envs, n_agents, 48)
                agent_0 take obs[0:18] + obs[36:66]
                agent_1 take obs[18:36] + obs[36:66] 
            share_obs: numpy (n_envs, n_agents, 66)
       
        """
        shared = obs[:, self.proprio_total:] 

        local_list = []
        for i in range(self.n_agents):
            start = i * self.proprio_per_agent
            end = start + self.proprio_per_agent
            agent_proprio = obs[:, start:end] 

            agent_local = torch.cat([agent_proprio, shared], dim=-1)  # (1024, 48)
            local_list.append(agent_local)

        # stack to (n_envs, n_agents, 48) and then convert to numpy
        local_obs = _t2n(torch.stack(local_list, dim=1))

        # share_obs: each agent's critic look at the whole obs
        # (n_envs, 66) -> copy n_agents times -> (n_envs, n_agents, 66)
        share_obs = _t2n(obs.unsqueeze(1).expand(-1, self.n_agents, -1))

        return local_obs, share_obs
