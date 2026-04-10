import time
import numpy as np
import wandb
from harl.common.base_logger import BaseLogger

class WandbWriter:
    """Replacement for SummaryWriter that logs to wandb."""

    def add_scalars(self, main_tag, tag_scalar_dict, global_step):
        for k, v in tag_scalar_dict.items():
            wandb.log({k: v}, step=int(global_step))

    def export_scalars_to_json(self, path):
        pass

    def close(self):
        wandb.finish()

class ManiSkillLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super().__init__(args, algo_args, env_args, num_agents, writter, run_dir)
        wandb.init(
            project="maniskill_baselines",
            name=f"{args['exp_name']}_seed{algo_args['seed']['seed']}",
            config={"args": args, "algo_args": algo_args, "env_args": env_args},
        )
        self.writter = WandbWriter()
    def get_task_name(self):
        return self.env_args["task"]
    def eval_init(self):
        super().eval_init()
        self.eval_episode_metrics = []

    def eval_thread_done(self, tid):
        super().eval_thread_done(tid)
        # Read episode metrics from ManiSkillVectorEnv's final_info
        ep = self.eval_infos[tid][0].get("episode", {})
        if ep:
            self.eval_episode_metrics.append(ep)

    def eval_log(self, eval_episode):
        super().eval_log(eval_episode)
        if self.eval_episode_metrics:
            for k in self.eval_episode_metrics[0]:
                vals = [ep[k] for ep in self.eval_episode_metrics]
                mean_val = np.mean(vals)
                self.writter.add_scalars(
                    f"eval/{k}", {f"eval/{k}": mean_val}, self.total_num_steps,
                )

    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        """Log information for each episode."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()
        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
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
                ",".join(map(str, [self.total_num_steps, aver_episode_rewards])) + "\n"
            )
            self.log_file.flush()
            self.done_episodes_rewards = []
