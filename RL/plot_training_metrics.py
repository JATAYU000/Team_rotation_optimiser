"""Plot and visualize multi-character combat PPO training metrics and graphs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def resolve_log_path(raw_path: str) -> Path:
    if raw_path.startswith("user://"):
        relative = raw_path[len("user://") :]
        app_data = os.environ.get("APPDATA", "")
        candidates = [
            Path(app_data) / "Godot" / "app_userdata" / "Wushin" / relative,
            Path(app_data) / "Godot" / "app_userdata" / "wushin" / relative,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    return Path(raw_path)


def load_metrics(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Training progress log not found: {log_path}")

    entries: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "update" in data and "win_rate" in data:
                    entries.append(data)
            except json.JSONDecodeError:
                continue
    return entries


def plot_dashboard(entries: list[dict], save_path: str | None = None) -> None:
    if not entries:
        print("No evaluation update entries found in log.")
        return

    updates = [e["update"] for e in entries]
    win_rates = [e["win_rate"] * 100.0 for e in entries]
    rewards = [e.get("reward", 0.0) for e in entries]
    losses = [e.get("loss", 0.0) for e in entries]
    val_losses = [e.get("value_loss", 0.0) for e in entries]
    avg_damages = [e.get("average_damage", 0.0) for e in entries]
    avg_buffed_damages = [e.get("average_buffed_damage", 0.0) for e in entries]
    avg_incoming = [e.get("average_incoming_damage", 0.0) for e in entries]
    parries = [e.get("average_parries", 0.0) for e in entries]
    dodges = [e.get("average_dodges", 0.0) for e in entries]
    switches = [e.get("average_switches", 0.0) for e in entries]
    buffs = [e.get("average_buffs", 0.0) for e in entries]
    heals = [e.get("average_heals", 0.0) for e in entries]
    hp_healed = [e.get("average_hp_healed", 0.0) for e in entries]
    durations = [e.get("average_time", 0.0) for e in entries]
    survivals = [entries[i].get("average_survival_rate", 100.0 if win_rates[i] > 0 else 33.3) for i in range(len(entries))]
    entropies = [e.get("entropy", 0.0) for e in entries]
    epsilons = [e.get("epsilon", 0.0) for e in entries]

    # Character presence percentages
    presence_knight = [e.get("field_presence", {}).get("Knight", 33.3) for e in entries]
    presence_mage = [e.get("field_presence", {}).get("Mage", 33.3) for e in entries]
    presence_rogue = [e.get("field_presence", {}).get("Rogue", 33.3) for e in entries]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(
        "Wushin PPO Multi-Character Combat Training & Synergy Dashboard",
        fontsize=18,
        fontweight="bold",
        color="#f0d588",
        y=0.995,
    )

    # 1. Win Rate & Reward
    ax1 = axes[0, 0]
    ax1.set_title("Win Rate & Policy Reward", fontsize=12, color="#f0d588")
    ax1.plot(updates, win_rates, color="#34d399", linewidth=2.4, label="Win Rate (%)")
    ax1.set_xlabel("PPO Update")
    ax1.set_ylabel("Win Rate (%)", color="#34d399")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(updates, rewards, color="#60a5fa", linewidth=1.6, linestyle="--", label="Avg Reward")
    ax1_twin.set_ylabel("Mean Reward", color="#60a5fa")

    # 2. Damage Dealt (Total vs Buff Window Burst) & Incoming Damage
    ax2 = axes[0, 1]
    ax2.set_title("Boss Damage Output (Total vs Buffed Burst Window)", fontsize=12, color="#f0d588")
    ax2.plot(updates, avg_damages, color="#fbbf24", linewidth=2.2, label="Total Boss Damage")
    if any(b > 0 for b in avg_buffed_damages):
        ax2.plot(updates, avg_buffed_damages, color="#f59e0b", linewidth=1.8, linestyle=":", label="Buffed Burst Dmg")
    ax2.plot(updates, avg_incoming, color="#f87171", linewidth=1.8, label="Team Incoming Damage")
    ax2.axhline(y=5000, color="#ef4444", linestyle="--", alpha=0.5, label="Boss Max HP (5000)")
    ax2.set_xlabel("PPO Update")
    ax2.set_ylabel("HP Points")
    ax2.legend(loc="upper left", framealpha=0.6)
    ax2.grid(True, linestyle="--", alpha=0.3)

    # 3. Tactical Defense: Parries & Dodges
    ax3 = axes[1, 0]
    ax3.set_title("Tactical Defense: Parries & Perfect Dodges per Episode", fontsize=12, color="#f0d588")
    ax3.plot(updates, parries, color="#fb923c", linewidth=2.2, marker="o", markersize=4, label="Parry Stuns")
    ax3.plot(updates, dodges, color="#38bdf8", linewidth=1.8, marker="s", markersize=4, label="Perfect Dodges")
    ax3.set_xlabel("PPO Update")
    ax3.set_ylabel("Count per Episode")
    ax3.legend(loc="upper left", framealpha=0.6)
    ax3.grid(True, linestyle="--", alpha=0.3)

    # 4. Character Field Presence (Diluc vs Lisa vs Lynae Rotation Share)
    ax4 = axes[1, 1]
    ax4.set_title("Character Field Presence (% Time Active)", fontsize=12, color="#f0d588")
    ax4.plot(updates, presence_knight, color="#e11d48", linewidth=2.2, marker="o", markersize=3, label="Diluc (Knight / DPS)")
    ax4.plot(updates, presence_mage, color="#8b5cf6", linewidth=2.0, marker="^", markersize=3, label="Lisa (Mage / Buffer & Heal)")
    ax4.plot(updates, presence_rogue, color="#10b981", linewidth=2.0, marker="s", markersize=3, label="Lynae (Rogue / Buffer)")
    ax4.set_xlabel("PPO Update")
    ax4.set_ylabel("Field Time (%)")
    ax4.set_ylim(-5, 105)
    ax4.legend(loc="upper right", framealpha=0.6)
    ax4.grid(True, linestyle="--", alpha=0.3)

    # 5. Multi-Character Synergy: Switches, Buffs & Heals Deployed
    ax5 = axes[2, 0]
    ax5.set_title("Synergy: Switches, Buffs & Heals Deployed", fontsize=12, color="#f0d588")
    ax5.plot(updates, switches, color="#a78bfa", linewidth=2.2, marker="^", markersize=4, label="Switches")
    ax5.plot(updates, buffs, color="#ec4899", linewidth=2.0, marker="d", markersize=4, label="Party Buffs")
    if any(h > 0 for h in heals):
        ax5.plot(updates, heals, color="#34d399", linewidth=2.0, marker="o", markersize=4, label="Team Heals")
    ax5.set_xlabel("PPO Update")
    ax5.set_ylabel("Count per Episode")
    ax5.legend(loc="upper left", framealpha=0.6)
    ax5.grid(True, linestyle="--", alpha=0.3)

    if any(hp > 0 for hp in hp_healed):
        ax5_twin = ax5.twinx()
        ax5_twin.plot(updates, hp_healed, color="#10b981", linewidth=1.5, linestyle=":", label="HP Restored")
        ax5_twin.set_ylabel("HP Restored / Ep", color="#10b981")

    # 6. Team Survival Rate & Battle Duration
    ax6 = axes[2, 1]
    ax6.set_title("Team Survival Rate & Battle Duration", fontsize=12, color="#f0d588")
    ax6.plot(updates, survivals, color="#10b981", linewidth=2.0, label="Team Survival Rate (%)")
    ax6.set_xlabel("PPO Update")
    ax6.set_ylabel("Survival Rate (%)", color="#10b981")
    ax6.set_ylim(-5, 105)
    ax6.grid(True, linestyle="--", alpha=0.3)
    ax6_twin = ax6.twinx()
    ax6_twin.plot(updates, durations, color="#38bdf8", linewidth=1.5, linestyle="--", label="Avg Battle Duration (s)")
    ax6_twin.set_ylabel("Seconds", color="#38bdf8")

    # 7. Move Usage Distribution (Last Evaluation)
    ax7 = axes[3, 0]
    ax7.set_title("Ability Distribution Across Characters (Latest Eval)", fontsize=12, color="#f0d588")
    last_entry = entries[-1]
    move_dist = last_entry.get("move_distribution", {})
    if move_dist:
        moves = ["Basic", "Heavy", "Skill", "Ult"]
        x = np.arange(len(moves))
        width = 0.25
        chars = ["Knight", "Rogue", "Mage"]
        colors = ["#e11d48", "#10b981", "#8b5cf6"]
        labels = ["Diluc (Knight)", "Lynae (Rogue)", "Lisa (Mage)"]
        for idx, char_key in enumerate(chars):
            counts = move_dist.get(char_key, [0, 0, 0, 0])
            ax7.bar(x + idx * width - width, counts, width, label=labels[idx], color=colors[idx], alpha=0.85)
        ax7.set_xticks(x)
        ax7.set_xticklabels(moves)
        ax7.set_ylabel("Total Uses in Eval")
        ax7.legend(loc="upper right", framealpha=0.6)
    ax7.grid(True, linestyle="--", alpha=0.3)

    # 8. Policy Loss, Value Loss & Exploration Entropy
    ax8 = axes[3, 1]
    ax8.set_title("Optimization: Loss & Exploration Entropy", fontsize=12, color="#f0d588")
    ax8.plot(updates, losses, color="#f43f5e", linewidth=1.8, label="Total Loss")
    if any(vl > 0 for vl in val_losses):
        ax8.plot(updates, val_losses, color="#fb7185", linewidth=1.2, linestyle=":", label="Value Loss")
    ax8.set_xlabel("PPO Update")
    ax8.set_ylabel("Loss", color="#f43f5e")
    ax8.legend(loc="upper left", framealpha=0.6)
    ax8.grid(True, linestyle="--", alpha=0.3)
    ax8_twin = ax8.twinx()
    ax8_twin.plot(updates, entropies, color="#2dd4bf", linewidth=1.5, linestyle=":", label="Entropy")
    ax8_twin.plot(updates, epsilons, color="#94a3b8", linewidth=1.2, linestyle="--", label="Epsilon (Explore)")
    ax8_twin.set_ylabel("Entropy / Epsilon", color="#2dd4bf")

    plt.tight_layout()
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        print(f"Dashboard saved to: {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot and evaluate training metrics from PPO progress log.")
    parser.add_argument(
        "--log",
        default="user://rl_training_progress.jsonl",
        help="Path to progress log file (supports user:// prefix)",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional output PNG file path. If omitted, opens the interactive GUI window.",
    )
    args = parser.parse_args()
    log_file = resolve_log_path(args.log)
    print(f"Reading training metrics from: {log_file}")
    entries = load_metrics(log_file)
    print(f"Loaded {len(entries)} evaluation updates.")
    plot_dashboard(entries, save_path=args.save if args.save else None)


if __name__ == "__main__":
    main()
