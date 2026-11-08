"""Q-Learning / Advantage Actor-Critic (A2C) training pipeline for Wushin combat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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


class QActorCritic(nn.Module):
    """Deep Q / Actor-Critic architecture for multi-character combat."""

    def __init__(self, observation_size: int, action_count: int = ACTION_COUNT, hidden_dim: int = 128):
        super().__init__()
        self.input_norm = nn.LayerNorm(observation_size)
        self.shared_input = nn.Linear(observation_size, hidden_dim)
        self.shared_res = ResidualBlock(hidden_dim)

        # Policy / Action trunk
        self.actor_fc = nn.Linear(hidden_dim, hidden_dim)
        self.actor_res = ResidualBlock(hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_count)

        # Value / Q trunk
        self.critic_fc = nn.Linear(hidden_dim, hidden_dim)
        self.critic_res = ResidualBlock(hidden_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(observation)
        features = F.silu(self.shared_input(x))
        features = self.shared_res(features)

        actor_feats = self.actor_res(F.silu(self.actor_fc(features)))
        logits = self.actor_head(actor_feats)

        critic_feats = self.critic_res(F.silu(self.critic_fc(features)))
        value = self.critic_head(critic_feats)
        return logits, value


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


def masked_distribution(logits: torch.Tensor, mask: np.ndarray | None) -> Categorical:
    if mask is not None:
        tensor_mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(~tensor_mask, -1e9)
    return Categorical(logits=logits)


def collect_rollout(
    env: CombatEnv,
    model: QActorCritic,
    steps: int,
    device: torch.device,
    epsilon: float = 0.0,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    values: list[torch.Tensor] = []
    dones: list[bool] = []
    action_masks: list[np.ndarray] = []
    observation = env._observation()
    episodes_completed = 0
    episodes_won = 0

    for _ in range(steps):
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        mask = env.action_mask()
        with torch.no_grad():
            logits, value = model(obs_t)
            dist = masked_distribution(logits[0], mask)

        if epsilon > 0.0 and np.random.rand() < epsilon:
            valid = np.where(mask > 0.5)[0]
            action = torch.tensor(int(np.random.choice(valid)) if len(valid) > 0 else 0, device=device)
        else:
            action = dist.sample()

        next_obs, reward, done, info = env.step(int(action.item()))
        observations.append(observation)
        actions.append(int(action.item()))
        log_probs.append(dist.log_prob(action).detach())
        rewards.append(reward)
        values.append(value[0].detach())
        dones.append(done)
        action_masks.append(mask)

        if done:
            episodes_completed += 1
            if bool(info.get("won", False)):
                episodes_won += 1
            observation = env.reset()
        else:
            observation = next_obs

    with torch.no_grad():
        _, next_value = model(torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0))

    return {
        "observations": torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=device),
        "actions": torch.as_tensor(actions, dtype=torch.long, device=device),
        "log_probs": torch.stack(log_probs),
        "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=device),
        "values": torch.stack(values),
        "dones": torch.as_tensor(dones, dtype=torch.float32, device=device),
        "action_masks": torch.as_tensor(np.asarray(action_masks), dtype=torch.bool, device=device),
        "next_value": next_value[0],
    }, {"episodes": episodes_completed, "wins": episodes_won}


def calculate_returns(rollout: dict[str, torch.Tensor], gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = rollout["rewards"]
    values = rollout["values"]
    dones = rollout["dones"]
    next_value = rollout["next_value"]

    returns = torch.zeros_like(rewards)
    running_return = next_value
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return * (1.0 - dones[t])
        returns[t] = running_return

    advantages = returns - values.squeeze(-1)
    return advantages, returns


def _evaluate_model(
    env: CombatEnv,
    model: QActorCritic,
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
                logits, _ = model(obs_t)
                action = int(torch.argmax(logits[0].masked_fill(~mask, -1e9)).item())

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

    try:
        device = _select_device(args)
    except RuntimeError as error:
        message = str(error)
        _write_progress(args.progress, message, 0, error=message)
        print(message)
        return

    env = CombatEnv(load_stats(args.stats), seed=args.seed)
    model = QActorCritic(env.observation_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    device_label = f"{device} ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else str(device)
    _write_progress(args.progress, "starting train (Q-Learning / A2C)...", 0, device=device_label)
    print(f"Q-Learning/A2C Training started on {device_label}. Observation size: {env.observation_size}, Actions: {ACTION_COUNT}")
    if args.early_stop_streak > 0:
        print(f"Early stopping active: will stop if policy wins for {args.early_stop_streak} consecutive updates (win rate >= {args.early_stop_win_rate*100:.0f}%).")

    consecutive_winning_updates = 0
    consecutive_eval_wins = 0
    early_stopped = False

    for update in range(1, args.updates + 1):
        progress_frac = float(update - 1) / max(1, args.updates - 1)
        current_epsilon = max(args.epsilon_end, args.epsilon_start * (1.0 - progress_frac))
        current_entropy_coef = max(0.005, args.entropy_coefficient * (1.0 - 0.7 * progress_frac))

        rollout, rollout_stats = collect_rollout(env, model, args.rollout_steps, device, epsilon=current_epsilon)
        advantages, returns = calculate_returns(rollout, args.gamma)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if rollout_stats["episodes"] > 0:
            up_win_rate = rollout_stats["wins"] / rollout_stats["episodes"]
            if up_win_rate >= args.early_stop_win_rate:
                consecutive_winning_updates += 1
            else:
                consecutive_winning_updates = 0

        logits, values = model(rollout["observations"])
        logits = logits.masked_fill(~rollout["action_masks"], -1e9)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(rollout["actions"])
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = 0.5 * (returns.detach() - values.squeeze(-1)).pow(2).mean()
        loss = policy_loss + args.value_coefficient * value_loss - current_entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if update % args.log_every == 0 or update == args.updates:
            metrics = _evaluate_model(env, model, args.eval_episodes, device, args.seed + update * args.eval_episodes)
            env.reset(seed=args.seed + update)
            if metrics["win_rate"] >= args.early_stop_win_rate:
                consecutive_eval_wins += 1
            else:
                consecutive_eval_wins = 0

            metrics.update({
                "update": update,
                "reward": float(rollout["rewards"].mean().item()),
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "epsilon": float(current_epsilon),
                "device": device_label,
                "consecutive_wins": consecutive_winning_updates,
            })
            moves_str = " | ".join(f"{k}:{v}" for k, v in metrics.get('move_distribution', {}).items())
            streak_str = f"Streak: {consecutive_winning_updates}/{args.early_stop_streak}" if args.early_stop_streak > 0 else ""
            print(
                f"[Q-Learning Update {update:03d}/{args.updates}] "
                f"Reward: {metrics['reward']:+.3f} | "
                f"Win: {metrics['win_rate']*100:.1f}% | "
                f"{streak_str} | "
                f"Dmg: {metrics['average_damage']:.1f} | "
                f"Heals: {metrics['average_heals']:.1f} ({metrics['average_hp_healed']:.0f} HP) | "
                f"Switches: {metrics['average_switches']:.1f} | "
                f"Buffs: {metrics['average_buffs']:.1f} | "
                f"Parries: {metrics['average_parries']:.1f} | "
                f"Dodges: {metrics['average_dodges']:.1f}\n"
                f"   Moves -> {moves_str}",
                flush=True
            )
        else:
            metrics = {
                "update": update,
                "reward": float(rollout["rewards"].mean().item()),
                "epsilon": float(current_epsilon),
                "device": device_label,
                "consecutive_wins": consecutive_winning_updates,
            }

        status_msg = (
            f"training Q-Learning {update * 100 // args.updates}% (streak: {consecutive_winning_updates}/{args.early_stop_streak})"
            if args.early_stop_streak > 0
            else f"training Q-Learning {update * 100 // args.updates}%"
        )
        _write_progress(args.progress, status_msg, update * 100 // args.updates, **metrics)

        # Early stopping trigger
        if args.early_stop_streak > 0:
            eval_target = max(3, min(args.early_stop_streak, 10))
            if consecutive_winning_updates >= args.early_stop_streak or consecutive_eval_wins >= eval_target:
                final_eval = _evaluate_model(env, model, args.eval_episodes, device, args.seed + 99999)
                if final_eval["win_rate"] >= args.early_stop_win_rate:
                    reason = (
                        f"{consecutive_winning_updates} updates in a row"
                        if consecutive_winning_updates >= args.early_stop_streak
                        else f"{consecutive_eval_wins} evaluation checkpoints in a row"
                    )
                    print(
                        f"\n=======================================================\n"
                        f"[Q-LEARNING EARLY STOPPING TRIGGERED] Mastered combat! Policy won for {reason}\n"
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
            "algo": "qlearning",
            "model": model.state_dict(),
            "observation_size": env.observation_size,
            "action_count": ACTION_COUNT,
            "stats_path": str(args.stats),
        },
        output,
    )
    print(f"Saved Q-Learning policy checkpoint to {output}")
    if not early_stopped:
        _write_progress(args.progress, "done..", 100, device=device_label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", default="data/combat_stats.json")
    parser.add_argument("--output", default="RL/checkpoints/diluc_qlearning.pt")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.03)
    parser.add_argument("--epsilon-start", type=float, default=0.35)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
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
