"""Evaluate a trained PPO checkpoint over deterministic combat episodes."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from combat_env import CombatEnv, load_stats
from train_ppo import ActorCritic


def evaluate(stats_path: str, checkpoint_path: str, episodes: int, seed: int) -> None:
    stats = load_stats(stats_path)
    env = CombatEnv(stats, seed=seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ActorCritic(int(checkpoint["observation_size"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    wins = 0
    durations: list[int] = []
    damages: list[float] = []
    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        while True:
            with torch.no_grad():
                logits, _ = model(torch.as_tensor(observation).unsqueeze(0))
                logits = logits[0].masked_fill(
                    ~torch.as_tensor(env.action_mask(), dtype=torch.bool), -1e9
                )
                action = int(torch.argmax(logits).item())
            observation, _, done, info = env.step(action)
            if done:
                wins += int(info["won"])
                durations.append(int(info["time"]))
                damages.append(float(info["total_damage"]))
                break
    print(
        f"episodes={episodes} wins={wins} win_rate={wins / episodes:.3f} "
        f"average_time={np.mean(durations):.1f} average_damage={np.mean(damages):.1f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", default="data/combat_stats.json")
    parser.add_argument("--checkpoint", default="RL/checkpoints/diluc_ppo.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.stats, args.checkpoint, args.episodes, args.seed)
