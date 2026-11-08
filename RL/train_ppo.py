from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from combat_env import ACTION_COUNT, CHARACTER_KEYS, CombatEnv, load_stats


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act2 = nn.SiLU()
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=float(np.sqrt(2)))
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=float(np.sqrt(2)))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.act1(self.ln1(self.fc1(x)))
        res = self.ln2(self.fc2(res))
        return self.act2(x + res)


class ActorCritic(nn.Module):
    """Deep Residual Actor-Critic network with decoupled Policy and Value trunks."""

    def __init__(self, observation_size: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_norm = nn.LayerNorm(observation_size)

        # Decoupled Policy (Actor) trunk
        self.actor_in = nn.Sequential(
            nn.Linear(observation_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.actor_res1 = ResidualBlock(hidden_dim)
        self.actor_res2 = ResidualBlock(hidden_dim)
        self.policy = nn.Linear(hidden_dim, ACTION_COUNT)

        # Decoupled Value (Critic) trunk
        self.critic_in = nn.Sequential(
            nn.Linear(observation_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.critic_res1 = ResidualBlock(hidden_dim)
        self.critic_res2 = ResidualBlock(hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

        self._init_heads()

    def _init_heads(self):
        nn.init.orthogonal_(self.actor_in[0].weight, gain=float(np.sqrt(2)))
        nn.init.zeros_(self.actor_in[0].bias)
        nn.init.orthogonal_(self.critic_in[0].weight, gain=float(np.sqrt(2)))
        nn.init.zeros_(self.critic_in[0].bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm_obs = self.obs_norm(observations)
        a = self.actor_in(norm_obs)
        a = self.actor_res1(a)
        a = self.actor_res2(a)
        logits = self.policy(a)

        c = self.critic_in(norm_obs)
        c = self.critic_res1(c)
        c = self.critic_res2(c)
        val = self.value(c).squeeze(-1)
        return logits, val


def masked_distribution(logits: torch.Tensor, mask: np.ndarray | None) -> Categorical:
    if mask is not None:
        tensor_mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(~tensor_mask, -1e9)
    return Categorical(logits=logits)


def collect_rollout(
    env: CombatEnv,
    model: ActorCritic,
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
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        action_mask = env.action_mask()
        with torch.no_grad():
            logits, value = model(observation_tensor)
            distribution = masked_distribution(logits[0], action_mask)

        # Epsilon-greedy exploration over valid masked actions
        if epsilon > 0.0 and np.random.rand() < epsilon:
            valid_actions = np.where(action_mask > 0.5)[0]
            if len(valid_actions) > 0:
                selected_action = int(np.random.choice(valid_actions))
                action = torch.tensor(selected_action, device=device)
            else:
                action = distribution.sample()
        else:
            action = distribution.sample()

        next_observation, reward, done, info = env.step(int(action.item()))
        observations.append(observation)
        actions.append(int(action.item()))
        log_probs.append(distribution.log_prob(action).detach())
        rewards.append(reward)
        values.append(value[0].detach())
        dones.append(done)
        action_masks.append(action_mask)
        if done:
            episodes_completed += 1
            if bool(info.get("won", False)):
                episodes_won += 1
            observation = env.reset()
        else:
            observation = next_observation

    with torch.no_grad():
        _, next_value = model(torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0))

    return {
        "observations": torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=device),
        "actions": torch.as_tensor(actions, dtype=torch.long, device=device),
        "old_log_probs": torch.stack(log_probs),
        "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=device),
        "values": torch.stack(values),
        "dones": torch.as_tensor(dones, dtype=torch.float32, device=device),
        "action_masks": torch.as_tensor(np.asarray(action_masks), dtype=torch.bool, device=device),
        "next_value": next_value[0],
    }, {"episodes": episodes_completed, "wins": episodes_won}


def calculate_gae(rollout: dict[str, torch.Tensor], gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rollout["rewards"])
    last_gae = torch.zeros((), device=advantages.device)
    values = torch.cat((rollout["values"], rollout["next_value"].reshape(1)))
    for index in reversed(range(len(advantages))):
        non_terminal = 1.0 - rollout["dones"][index]
        delta = rollout["rewards"][index] + gamma * values[index + 1] * non_terminal - values[index]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[index] = last_gae
    return advantages, advantages + rollout["values"]


def _write_progress(path: str, message: str, percent: int, **metrics: object) -> None:
    if not path:
        return
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as progress_file:
        progress_file.write(
            json.dumps(
                {"timestamp": time.time(), "message": message, "percent": percent, **metrics}
            )
            + "\n"
        )


def _select_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return torch.device("cuda")
    if args.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _evaluate_model(
    env: CombatEnv,
    model: ActorCritic,
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

    # Evaluate across starting characters evenly
    for episode in range(episodes):
        start_char = episode % len(CHARACTER_KEYS)
        observation = env.reset(seed=seed + episode, start_active=start_char)
        episode_incoming = 0.0
        episode_buffed_dmg = 0.0
        char_ticks = {k: 0 for k in CHARACTER_KEYS}

        while True:
            char_ticks[CHARACTER_KEYS[env.active]] += 1
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(observation_tensor)
                mask = torch.as_tensor(env.action_mask(), dtype=torch.bool, device=device)
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
    presence = {
        k: float(np.mean([fr[k] for fr in field_ratios])) * 100.0 for k in CHARACTER_KEYS
    }

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
    model = ActorCritic(env.observation_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    device_label = f"{device} ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else str(device)
    _write_progress(args.progress, "starting train...", 0, device=device_label)
    print(f"Training started on {device_label}. Observation size: {env.observation_size}, Actions: {ACTION_COUNT}")
    if args.early_stop_streak > 0:
        print(f"Early stopping active: will stop if policy wins for {args.early_stop_streak} consecutive updates (win rate >= {args.early_stop_win_rate*100:.0f}%).")

    consecutive_winning_updates = 0
    consecutive_eval_wins = 0
    early_stopped = False

    for update in range(1, args.updates + 1):
        # Exploration curriculum: decay epsilon from args.epsilon_start down to args.epsilon_end
        progress_frac = float(update - 1) / max(1, args.updates - 1)
        current_epsilon = max(args.epsilon_end, args.epsilon_start * (1.0 - progress_frac))
        current_entropy_coef = max(0.005, args.entropy_coefficient * (1.0 - 0.7 * progress_frac))

        rollout, rollout_stats = collect_rollout(env, model, args.rollout_steps, device, epsilon=current_epsilon)
        advantages, returns = calculate_gae(rollout, args.gamma, args.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Check rollout episode win rate for update streak
        if rollout_stats["episodes"] > 0:
            up_win_rate = rollout_stats["wins"] / rollout_stats["episodes"]
            if up_win_rate >= args.early_stop_win_rate:
                consecutive_winning_updates += 1
            else:
                consecutive_winning_updates = 0

        for _ in range(args.epochs):
            logits, values = model(rollout["observations"])
            logits = logits.masked_fill(~rollout["action_masks"], -1e9)
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(rollout["actions"])
            ratio = (new_log_probs - rollout["old_log_probs"]).exp()
            clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
            policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
            value_loss = 0.5 * (returns - values).pow(2).mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + args.value_coefficient * value_loss - current_entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        if update % args.log_every == 0 or update == args.updates:
            metrics = _evaluate_model(
                env,
                model,
                args.eval_episodes,
                device,
                args.seed + update * args.eval_episodes,
            )
            env.reset(seed=args.seed + update)
            if metrics["win_rate"] >= args.early_stop_win_rate:
                consecutive_eval_wins += 1
            else:
                consecutive_eval_wins = 0

            metrics.update(
                {
                    "update": update,
                    "reward": float(rollout["rewards"].mean().item()),
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "epsilon": float(current_epsilon),
                    "device": device_label,
                    "consecutive_wins": consecutive_winning_updates,
                }
            )
            moves_str = " | ".join(f"{k}:{v}" for k, v in metrics.get('move_distribution', {}).items())
            streak_str = f"Streak: {consecutive_winning_updates}/{args.early_stop_streak}" if args.early_stop_streak > 0 else ""
            print(
                f"[Update {update:03d}/{args.updates}] "
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
            f"training {update * 100 // args.updates}% (streak: {consecutive_winning_updates}/{args.early_stop_streak})"
            if args.early_stop_streak > 0
            else f"training {update * 100 // args.updates}%"
        )
        _write_progress(
            args.progress,
            status_msg,
            update * 100 // args.updates,
            **metrics,
        )

        # Early stopping trigger: stop if policy won for N updates in a row
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
                        f"[EARLY STOPPING TRIGGERED] Mastered combat! Policy won for {reason}\n"
                        f"Final Evaluation Win Rate: {final_eval['win_rate']*100:.1f}%\n"
                        f"Stopping early at update {update}/{args.updates} to save time.\n"
                        f"======================================================="
                    )
                    early_stopped = True
                    metrics.update(final_eval)
                    metrics["update"] = update
                    _write_progress(
                        args.progress,
                        f"mastered! early stopped after {reason}",
                        100,
                        **metrics,
                    )
                    break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algo": "ppo",
            "model": model.state_dict(),
            "observation_size": env.observation_size,
            "action_count": ACTION_COUNT,
            "stats_path": str(args.stats),
        },
        output,
    )
    print(f"Saved policy checkpoint to {output}")
    if not early_stopped:
        _write_progress(args.progress, "done..", 100, device=device_label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", default="data/combat_stats.json")
    parser.add_argument("--output", default="RL/checkpoints/diluc_ppo.pt")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.03)
    parser.add_argument("--epsilon-start", type=float, default=0.35)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--early-stop-streak", type=int, default=25, help="Stop training if agent wins for N consecutive updates in a row (default: 25, 0 to disable)")
    parser.add_argument("--early-stop-win-rate", type=float, default=0.75, help="Win rate threshold (0.0 to 1.0) to qualify as a winning update/evaluation (default: 0.75)")
    parser.add_argument("--progress", default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
