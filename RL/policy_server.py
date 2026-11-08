"""Local UDP inference server used by the Godot visual simulation.
Supports PPO, DQN, and Q-Learning policy checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket

import torch

from combat_env import ACTION_COUNT
from train_ppo import ActorCritic
from train_dqn import DuelingQNetwork
from train_qlearning import QActorCritic

ACTION_NAMES = [
    "Wait",
    "Basic Attack",
    "Heavy Attack",
    "Skill",
    "Ultimate",
    "Switch to Diluc (Knight)",
    "Switch to Lisa (Mage)",
    "Switch to Lynae (Rogue)",
    "Dodge",
]


def load_policy_model(checkpoint_path: str) -> tuple[torch.nn.Module, str, int]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    algo = checkpoint.get("algo", "ppo").lower()
    obs_size = int(checkpoint["observation_size"])

    if algo == "dqn":
        model: torch.nn.Module = DuelingQNetwork(obs_size)
    elif algo == "qlearning":
        model = QActorCritic(obs_size)
    else:
        algo = "ppo"
        model = ActorCritic(obs_size)

    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, algo, obs_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="RL/checkpoints/diluc_ppo.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    model, algo, obs_size = load_policy_model(args.checkpoint)

    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind((args.host, args.port))
    print(
        f"Policy server listening on {args.host}:{args.port} | Model: {algo.upper()} | Observation size: {obs_size}",
        flush=True,
    )

    while True:
        packet, address = server.recvfrom(65535)
        try:
            request = json.loads(packet.decode("utf-8"))
            obs_list = request["observation"]
            if len(obs_list) != obs_size:
                print(f"[Warning] Mismatched observation size: Godot sent {len(obs_list)}, model expects {obs_size}", flush=True)

            observation = torch.as_tensor(obs_list, dtype=torch.float32).unsqueeze(0)
            mask = torch.as_tensor(request.get("action_mask", [1] * ACTION_COUNT), dtype=torch.bool)

            with torch.no_grad():
                if algo == "dqn":
                    q_vals = model(observation)
                    action = int(torch.argmax(q_vals[0].masked_fill(~mask, -1e9)).item())
                else:
                    logits, _ = model(observation)
                    action = int(torch.argmax(logits[0].masked_fill(~mask, -1e9)).item())

            action_name = ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else f"Action {action}"
            print(f"[{algo.upper()} Server] Req #{request.get('id', 0)} -> {action_name}", flush=True)
            response = {"id": request.get("id", 0), "action": action}
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            print(f"[PolicyServer Error] {error}", flush=True)
            response = {"id": 0, "error": str(error)}
        server.sendto(json.dumps(response).encode("utf-8"), address)


if __name__ == "__main__":
    main()
