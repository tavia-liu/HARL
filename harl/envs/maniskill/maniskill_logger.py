"""
ManiSkill3 logger for HARL.
"""

import time
import numpy as np
from harl.common.base_logger import BaseLogger


class ManiSkillLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["task"]

    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()
        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, "
            "total num timesteps {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)

        print(
            "Average step reward is {}.".format(
                critic_train_info["average_step_rewards"]
            )
        )

        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(self.done_episodes_rewards)
            print(
                "Some episodes done, average episode reward is {}.\n".format(
                    aver_episode_rewards
                )
            )
            self.writter.add_scalars(
                "train_episode_rewards",
                {"aver_rewards": aver_episode_rewards},
                self.total_num_steps,
            )
            self.log_file.write(
                ",".join(map(str, [self.total_num_steps, aver_episode_rewards]))
                + "\n"
            )
            self.log_file.flush()
            self.done_episodes_rewards = []

    def eval_init(self):
        super().eval_init()
        # Track per-episode success flags across eval threads
        self.eval_episode_success = []
        self.one_episode_success = [[] for _ in range(self.algo_args["eval"]["n_eval_rollout_threads"])]

    def eval_per_step(self, eval_data):
        super().eval_per_step(eval_data)
        # eval_infos shape: (n_eval_threads, n_agents) — each element is a dict
        # ManiSkill puts "success" in info at env level (same for all agents),
        # so read from agent 0 of each thread.
        eval_infos = eval_data[4]
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            info = eval_infos[eval_i][0]  # agent 0's info dict
            if "success" in info:
                self.one_episode_success[eval_i].append(float(info["success"]))

    def eval_thread_done(self, tid):
        super().eval_thread_done(tid)
        # Episode is successful if success was achieved at any step
        if self.one_episode_success[tid]:
            self.eval_episode_success.append(
                float(any(self.one_episode_success[tid]))
            )
        self.one_episode_success[tid] = []

    def eval_log(self, eval_episode):
        """Log eval rewards and success rate."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_avg_rew = np.mean(self.eval_episode_rewards)

        # Success rate
        success_rate = (
            np.mean(self.eval_episode_success) if self.eval_episode_success else 0.0
        )

        # TensorBoard
        self.log_env({
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
        })
        self.writter.add_scalars(
            "eval_success_rate",
            {"eval_success_rate": success_rate},
            self.total_num_steps,
        )

        print(
            "Evaluation average episode reward is {:.4f}, success rate is {:.2%}.\n".format(
                eval_avg_rew, success_rate
            )
        )

        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_rew, success_rate]))
            + "\n"
        )
        self.log_file.flush()
