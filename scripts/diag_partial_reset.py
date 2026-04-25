"""
Diagnose partial_reset effects by loading a ppo_fast checkpoint and rolling
it out under both partial_reset=True and partial_reset=False.

Usage:
    python diag_partial_reset.py \
        --checkpoint /path/to/runs/.../final_ckpt.pt \
        --env-id TwoRobotStackCube-v1 \
        --num-envs 16 \
        --num-steps 500

Does NOT modify ppo_fast.py. Imports the Agent class from it.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

# Make ppo_fast importable without copying.
PPO_DIR = Path("/engrfs/project/chongjie/buzhao/robot/ManiSkill/examples/baselines/ppo")
sys.path.insert(0, str(PPO_DIR))
from ppo_fast import Agent  # noqa: E402

import mani_skill.envs  # noqa: F401,E402
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper  # noqa: E402
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # noqa: E402


def build_env(env_id: str, num_envs: int, partial_reset: bool, device: str):
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="gpu",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
    )
    envs = gym.make(env_id, num_envs=num_envs, **env_kwargs)
    if isinstance(envs.single_action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(
        envs, num_envs,
        ignore_terminations=not partial_reset,
        record_metrics=True,
    )
    return envs


@torch.no_grad()
def rollout(agent: Agent, envs, num_steps: int, device: str, label: str):
    obs, _ = envs.reset(seed=0)
    term_total = 0
    trunc_total = 0
    episode_returns = []
    episode_lens = []
    episode_success_once = []
    episode_success_at_end = []
    # per-env running trackers
    n = envs.num_envs
    cur_return = torch.zeros(n, device=device)
    cur_len = torch.zeros(n, device=device, dtype=torch.long)
    cur_succ_once = torch.zeros(n, device=device, dtype=torch.bool)

    for t in range(num_steps):
        action = agent.actor_mean(obs)
        obs, rew, term, trunc, info = envs.step(action)
        term_total += int(term.sum())
        trunc_total += int(trunc.sum())
        cur_return += rew
        cur_len += 1

        # success flag if the underlying env exposes it
        succ = None
        if isinstance(info, dict) and "success" in info:
            succ = info["success"]
            cur_succ_once |= succ

        done = term | trunc
        if done.any():
            idxs = torch.nonzero(done, as_tuple=False).flatten().tolist()
            for i in idxs:
                episode_returns.append(float(cur_return[i].item()))
                episode_lens.append(int(cur_len[i].item()))
                episode_success_once.append(bool(cur_succ_once[i].item()))
                if succ is not None:
                    episode_success_at_end.append(bool(succ[i].item()))
                cur_return[i] = 0.0
                cur_len[i] = 0
                cur_succ_once[i] = False

    def mean(xs):
        return float(np.mean(xs)) if len(xs) else float("nan")

    print(f"\n=== {label} ===")
    print(f"  steps collected      : {num_steps} x {n} envs = {num_steps*n}")
    print(f"  terminations (total) : {term_total}")
    print(f"  truncations  (total) : {trunc_total}")
    print(f"  episodes completed   : {len(episode_returns)}")
    print(f"  mean episode return  : {mean(episode_returns):.3f}")
    print(f"  mean episode length  : {mean(episode_lens):.2f}")
    print(f"  success_once rate    : {mean(episode_success_once):.3f}")
    if episode_success_at_end:
        print(f"  success_at_end rate  : {mean(episode_success_at_end):.3f}")
    print(f"  term-only (term & !trunc): "
          f"~ term_total - (both overlap) = {term_total} (trunc = {trunc_total})")
    return {
        "term": term_total, "trunc": trunc_total,
        "episodes": len(episode_returns),
        "return": mean(episode_returns),
        "len": mean(episode_lens),
        "success_once": mean(episode_success_once),
        "success_at_end": mean(episode_success_at_end) if episode_success_at_end else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--env-id", required=True, type=str)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=500,
                   help="Steps per rollout. Use >> horizon to see multiple resets.")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Build a probe env (partial_reset=True) just to read obs/act dims; use True so defaults match training.
    probe = build_env(args.env_id, args.num_envs, partial_reset=True, device=device)
    n_obs = math.prod(probe.single_observation_space.shape)
    n_act = math.prod(probe.single_action_space.shape)
    probe.close()

    print(f"env={args.env_id}  n_obs={n_obs}  n_act={n_act}  device={device}")
    print(f"loading checkpoint: {args.checkpoint}")

    agent = Agent(n_obs, n_act, device=device)
    state = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(state)
    agent.eval()

    results = {}
    for label, partial_reset in [("partial_reset=True", True), ("partial_reset=False", False)]:
        envs = build_env(args.env_id, args.num_envs, partial_reset=partial_reset, device=device)
        results[label] = rollout(agent, envs, args.num_steps, device, label)
        envs.close()

    print("\n=== SUMMARY (side-by-side) ===")
    keys = ["term", "trunc", "episodes", "return", "len", "success_once", "success_at_end"]
    hdr = f"{'metric':<18} | {'partial=True':>14} | {'partial=False':>14}"
    print(hdr)
    print("-" * len(hdr))
    for k in keys:
        a = results["partial_reset=True"][k]
        b = results["partial_reset=False"][k]
        print(f"{k:<18} | {a:>14.3f} | {b:>14.3f}")

    print("\nInterpretation:")
    t_a = results["partial_reset=True"]["term"]
    t_b = results["partial_reset=False"]["term"]
    if t_a == 0 and t_b == 0:
        print("  terminations never fire in either mode -> partial_reset is INERT for this task/policy.")
    elif t_a > 0 and t_b == 0:
        print("  terminations fire with partial_reset=True but are suppressed with False (expected; ignore_terminations=True).")
        print("  Flag is wired correctly; differences in training curves would be real if term rate is meaningful.")
    else:
        print("  unexpected: terminations observed in partial_reset=False mode. Check wrapper plumbing.")


if __name__ == "__main__":
    main()
