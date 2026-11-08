"""Deep Q-Network (Dueling Double DQN) training pipeline for Wushin combat."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from combat_env import ACTION_COUNT, CHARACTER_KEYS, CombatEnv, load_stats


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return F.silu(self.norm(x + residual))


class DuelingQNetwork(nn.Module):
    """Dueling Q-Network: separates state-value V(s) and advantage A(s, a)."""

    def __init__(self, observation_size: int, action_count: int = ACTION_COUNT, hidden_dim: int = 128):
        super().__init__()
        self.input_norm = nn.LayerNorm(observation_size)
        self.input_layer = nn.Linear(observation_size, hidden_dim)

        # Feature trunk
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)

        # Value stream V(s)
        self.value_fc = nn.Linear(hidden_dim, 64)
        self.value_out = nn.Linear(64, 1)

        # Advantage stream A(s, a)
        self.adv_fc = nn.Linear(hidden_dim, 64)
        self.adv_out = nn.Linear(64, action_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        features = F.silu(self.input_layer(x))
        features = self.res1(features)
        features = self.res2(features)

        value = self.value_out(F.silu(self.value_fc(features)))
        advantages = self.adv_out(F.silu(self.adv_fc(features)))

        # Dueling aggregation: Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer: collections.deque = collections.deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        self.buffer.append((obs, action, reward, next_obs, done, next_mask))

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones, next_masks = zip(*batch)

        return (
            torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.long, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(next_obs), dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(next_masks), dtype=torch.bool, device=device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def _select_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for training, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write_progress(progress_file: str, message: str, percent: int, **extra: Any) -> None:
    if not progress_file:
        return
    payload = {"message": message, "percent": percent}
    payload.update(extra)
    path = Path(progress_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + "\n")


def _evaluate_model(
    env: CombatEnv,
    model: DuelingQNetwork,
    episodes: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    if episodes <= 0:
        return {}
    wins = 0
    durations: list[int] = []
    damages: list[float] = []
    incoming_damages: list[float] = []
    buffed_damages: list[float] = []
    survivals: list[float] = []
    parries: list[int] = []
    dodges: list[int] = []
    switches: list[int] = []
    buffs: list[int] = []
    heals: list[int] = []
    hp_healed: list[float] = []
    field_ratios: list[dict[str, float]] = []
    total_moves = {k: np.zeros(4, dtype=np.int32) for k in CHARACTER_KEYS}

    for episode in range(episodes):
        start_char = episode % len(CHARACTER_KEYS)
        observation = env.reset(seed=seed + episode, start_active=start_char)
        episode_incoming = 0.0
        episode_buffed_dmg = 0.0
        char_ticks = {k: 0 for k in CHARACTER_KEYS}

        while True:
            char_ticks[CHARACTER_KEYS[env.active]] += 1
            obs_t = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.as_tensor(env.action_mask(), dtype=torch.bool, device=device)
            with torch.no_grad():
                q_vals = model(obs_t)
                action = int(torch.argmax(q_vals[0].masked_fill(~mask, -1e9)).item())

            observation, _, done, info = env.step(action)
            episode_incoming += float(info.get("incoming_damage", 0.0))
            if len(env.active_buffs) > 0:
                episode_buffed_dmg += float(info.get("damage", 0.0))

            if done:
                wins += int(info["won"])
                durations.append(int(info["time"]))
                damages.append(float(info["total_damage"]))
                incoming_damages.append(episode_incoming)
                buffed_damages.append(episode_buffed_dmg)
                alive_ratio = float(np.sum(env.hp > 0)) / float(len(CHARACTER_KEYS))
                survivals.append(alive_ratio)
                parries.append(int(info["parry_count"]))
                dodges.append(int(info["dodge_count"]))
                switches.append(int(info["switch_count"]))
                buffs.append(int(info["buffs_triggered"]))
                heals.append(int(info.get("heals_triggered", 0)))
                hp_healed.append(float(info.get("hp_healed_total", 0.0)))
                total_t = max(1, sum(char_ticks.values()))
                field_ratios.append({k: char_ticks[k] / total_t for k in CHARACTER_KEYS})
                for k in CHARACTER_KEYS:
                    total_moves[k] += np.asarray(info["moves_used"][k], dtype=np.int32)
                break

    move_dist = {k: [int(x) for x in total_moves[k]] for k in CHARACTER_KEYS}
    presence = {k: float(np.mean([fr[k] for fr in field_ratios])) * 100.0 for k in CHARACTER_KEYS}

    return {
        "win_rate": wins / episodes,
        "average_time": float(np.mean(durations)),
        "average_damage": float(np.mean(damages)),
        "average_incoming_damage": float(np.mean(incoming_damages)),
        "average_buffed_damage": float(np.mean(buffed_damages)),
        "average_survival_rate": float(np.mean(survivals)) * 100.0,
        "average_parries": float(np.mean(parries)),
        "average_dodges": float(np.mean(dodges)),
        "average_switches": float(np.mean(switches)),
        "average_buffs": float(np.mean(buffs)),
        "average_heals": float(np.mean(heals)),
        "average_hp_healed": float(np.mean(hp_healed)),
        "field_presence": presence,
        "move_distribution": move_dist,
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    try:
        device = _select_device(args)
    except RuntimeError as error:
        message = str(error)
        _write_progress(args.progress, message, 0, error=message)
        print(message)
        return

    env = CombatEnv(load_stats(args.stats), seed=args.seed)
    online_model = DuelingQNetwork(env.observation_size).to(device)
    target_model = DuelingQNetwork(env.observation_size).to(device)
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(online_model.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(capacity=args.buffer_size)

    device_label = f"{device} ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else str(device)
    _write_progress(args.progress, "starting train (DQN)...", 0, device=device_label)
    print(f"DQN Training started on {device_label}. Observation size: {env.observation_size}, Actions: {ACTION_COUNT}")
    if args.early_stop_streak > 0:
        print(f"Early stopping active: will stop if policy wins for {args.early_stop_streak} consecutive updates (win rate >= {args.early_stop_win_rate*100:.0f}%).")

    # Warmup replay buffer with random steps
    print(f"Warming up replay buffer with {args.warmup_steps} steps...")
    obs = env.reset(seed=args.seed)
    for _ in range(args.warmup_steps):
        mask = env.action_mask()
        valid = np.where(mask > 0.5)[0]
        act = int(np.random.choice(valid)) if len(valid) > 0 else 0
        next_obs, rew, done, _ = env.step(act)
        next_mask = env.action_mask()
        replay_buffer.push(obs, act, rew, next_obs, done, next_mask)
        obs = env.reset() if done else next_obs

    consecutive_winning_updates = 0
    consecutive_eval_wins = 0
    early_stopped = False

    obs = env.reset(seed=args.seed + 1)
    episode_wins_in_update = 0
    episodes_in_update = 0

    for update in range(1, args.updates + 1):
        progress_frac = float(update - 1) / max(1, args.updates - 1)
        current_epsilon = max(args.epsilon_end, args.epsilon_start * (1.0 - progress_frac))

        episode_wins_in_update = 0
        episodes_in_update = 0
        losses: list[float] = []

        # Collect interaction steps per update
        for _ in range(args.steps_per_update):
            mask = env.action_mask()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            if np.random.rand() < current_epsilon:
                valid = np.where(mask > 0.5)[0]
                action = int(np.random.choice(valid)) if len(valid) > 0 else 0
            else:
                with torch.no_grad():
                    q_vals = online_model(obs_t)
                    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
                    action = int(torch.argmax(q_vals[0].masked_fill(~mask_t, -1e9)).item())

            next_obs, rew, done, info = env.step(action)
            next_mask = env.action_mask()
            replay_buffer.push(obs, action, rew, next_obs, done, next_mask)

            if done:
                episodes_in_update += 1
                if bool(info.get("won", False)):
                    episode_wins_in_update += 1
                obs = env.reset()
            else:
                obs = next_obs

            # Train mini-batch
            if len(replay_buffer) >= args.batch_size:
                b_obs, b_actions, b_rewards, b_next_obs, b_dones, b_next_masks = replay_buffer.sample(args.batch_size, device)

                # Current Q(s, a)
                current_q = online_model(b_obs).gather(1, b_actions.unsqueeze(1)).squeeze(1)

                # Double DQN target: action chosen by online_model, evaluated by target_model
                with torch.no_grad():
                    next_online_q = online_model(b_next_obs)
                    next_online_q = next_online_q.masked_fill(~b_next_masks, -1e9)
                    best_next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)

                    next_target_q = target_model(b_next_obs).gather(1, best_next_actions).squeeze(1)
                    target_q = b_rewards + (1.0 - b_dones) * args.gamma * next_target_q

                loss = F.smooth_l1_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_model.parameters(), args.max_grad_norm)
                optimizer.step()
                losses.append(float(loss.item()))

                # Soft target update
                for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
                    target_param.data.copy_(args.tau * online_param.data + (1.0 - args.tau) * target_param.data)

        # Check update streak
        if episodes_in_update > 0:
            up_win_rate = episode_wins_in_update / episodes_in_update
            if up_win_rate >= args.early_stop_win_rate:
                consecutive_winning_updates += 1
            else:
                consecutive_winning_updates = 0

        avg_loss = float(np.mean(losses)) if losses else 0.0

        if update % args.log_every == 0 or update == args.updates:
            metrics = _evaluate_model(env, online_model, args.eval_episodes, device, args.seed + update * args.eval_episodes)
            env.reset(seed=args.seed + update)
            if metrics["win_rate"] >= args.early_stop_win_rate:
                consecutive_eval_wins += 1
            else:
                consecutive_eval_wins = 0

            metrics.update({
                "update": update,
                "reward": float(metrics.get("average_damage", 0.0) / 1000.0),
                "loss": avg_loss,
                "policy_loss": avg_loss,
                "value_loss": avg_loss,
                "entropy": 0.0,
                "epsilon": float(current_epsilon),
                "device": device_label,
                "consecutive_wins": consecutive_winning_updates,
            })
            moves_str = " | ".join(f"{k}:{v}" for k, v in metrics.get('move_distribution', {}).items())
            streak_str = f"Streak: {consecutive_winning_updates}/{args.early_stop_streak}" if args.early_stop_streak > 0 else ""
            print(
                f"[DQN Update {update:03d}/{args.updates}] "
                f"Win: {metrics['win_rate']*100:.1f}% | "
                f"{streak_str} | "
                f"Dmg: {metrics['average_damage']:.1f} | "
                f"Heals: {metrics['average_heals']:.1f} ({metrics['average_hp_healed']:.0f} HP) | "
                f"Switches: {metrics['average_switches']:.1f} | "
                f"Buffs: {metrics['average_buffs']:.1f} | "
                f"Parries: {metrics['average_parries']:.1f} | "
                f"Dodges: {metrics['average_dodges']:.1f} | "
                f"Loss: {avg_loss:.4f} | Eps: {current_epsilon:.3f}\n"
                f"   Moves -> {moves_str}",
                flush=True
            )
        else:
            metrics = {
                "update": update,
                "reward": 0.0,
                "loss": avg_loss,
                "epsilon": float(current_epsilon),
                "device": device_label,
                "consecutive_wins": consecutive_winning_updates,
            }

        status_msg = (
            f"training DQN {update * 100 // args.updates}% (streak: {consecutive_winning_updates}/{args.early_stop_streak})"
            if args.early_stop_streak > 0
            else f"training DQN {update * 100 // args.updates}%"
        )
        _write_progress(args.progress, status_msg, update * 100 // args.updates, **metrics)

        # Early stopping trigger
        if args.early_stop_streak > 0:
            eval_target = max(3, min(args.early_stop_streak, 10))
            if consecutive_winning_updates >= args.early_stop_streak or consecutive_eval_wins >= eval_target:
                final_eval = _evaluate_model(env, online_model, args.eval_episodes, device, args.seed + 99999)
                if final_eval["win_rate"] >= args.early_stop_win_rate:
                    reason = (
                        f"{consecutive_winning_updates} updates in a row"
                        if consecutive_winning_updates >= args.early_stop_streak
                        else f"{consecutive_eval_wins} evaluation checkpoints in a row"
                    )
                    print(
                        f"\n=======================================================\n"
                        f"[DQN EARLY STOPPING TRIGGERED] Mastered combat! Policy won for {reason}\n"
                        f"Final Evaluation Win Rate: {final_eval['win_rate']*100:.1f}%\n"
                        f"Stopping early at update {update}/{args.updates} to save compute.\n"
                        f"======================================================="
                    )
                    early_stopped = True
                    metrics.update(final_eval)
                    metrics["update"] = update
                    _write_progress(args.progress, f"mastered! early stopped after {reason}", 100, **metrics)
                    break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algo": "dqn",
            "model": online_model.state_dict(),
            "observation_size": env.observation_size,
            "action_count": ACTION_COUNT,
            "stats_path": str(args.stats),
        },
        output,
    )
    print(f"Saved DQN policy checkpoint to {output}")
    if not early_stopped:
        _write_progress(args.progress, "done..", 100, device=device_label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", default="data/combat_stats.json")
    parser.add_argument("--output", default="RL/checkpoints/diluc_dqn.pt")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--steps-per-update", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--epsilon-start", type=float, default=0.50)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--early-stop-streak", type=int, default=25)
    parser.add_argument("--early-stop-win-rate", type=float, default=0.75)
    parser.add_argument("--progress", default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
