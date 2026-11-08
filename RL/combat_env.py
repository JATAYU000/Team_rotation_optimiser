"""Combat simulation environment matching Wushin game mechanics."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np


def load_stats(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


ACTION_WAIT = 0
ACTION_BASIC = 1
ACTION_HEAVY = 2
ACTION_SKILL = 3
ACTION_ULTIMATE = 4
ACTION_SWITCH_KNIGHT = 5
ACTION_SWITCH_MAGE = 6
ACTION_SWITCH_ROGUE = 7
ACTION_DODGE = 8
ACTION_COUNT = 9

CHARACTER_KEYS = ("Knight", "Mage", "Rogue")
MOVE_NAMES = ("basic", "heavy", "skill", "ultimate")
BOSS_MOVES = ("Up", "Summon", "Taunt")
MAX_PATTERN_SLOTS = 16

PARRY_TOKEN_THRESHOLD = 2.0
BOSS_STUN_TICKS = 4
PLAYER_FREEZE_TICKS = 1
DODGE_MAX_CHARGES = 3
DODGE_REGEN_TICKS = 8

COOLDOWN_ACTION_PENALTY = 0.05
INVALID_ACTION_PENALTY = 0.02
IDLE_WHILE_READY_PENALTY = 0.002

BUFF_REGEX = re.compile(
    r"(?i)(crit_?rate|crate|cr|crit_?dmg|crit_?damage|cdmg|cd|attack|atk|a|defense|def|d|heal|hp|h)\s*[:=+]?\s*([+-]?\d+(?:\.\d+)?)\s*%?"
)


def _parse_buff_spec(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    if isinstance(raw, dict):
        stat_idx = int(raw.get("stat", 0))
        amt = float(raw.get("amount", 0.0))
        mapping = {0: "hp", 1: "atk", 2: "def", 3: "crit_rate", 4: "crit_damage"}
        return [{"stat": mapping.get(stat_idx, "atk"), "amount": amt}]
    if isinstance(raw, str):
        buffs: list[dict[str, Any]] = []
        for match in BUFF_REGEX.finditer(raw):
            token = match.group(1).lower()
            val = float(match.group(2))
            stat = "atk"
            if token in ("hp", "h", "heal"):
                stat = "hp"
            elif token in ("crit_rate", "crate", "cr"):
                stat = "crit_rate"
            elif token in ("crit_dmg", "crit_damage", "cdmg", "cd"):
                stat = "crit_damage"
            elif token in ("defense", "def", "d"):
                stat = "def"
            buffs.append({"stat": stat, "amount": val})
        return buffs
    return []


def _tokens(section: dict[str, Any], key: str) -> list[str]:
    patterns = section.get("attack_patterns", {}) or section.get("attacks", {})
    raw = patterns.get(key, [])
    if isinstance(raw, list):
        return [str(x) for x in raw]
    tokens: list[str] = []
    buffer = ""
    for char in str(raw):
        if char.isdigit() or char == ".":
            buffer += char
        else:
            if buffer:
                tokens.append(buffer)
                buffer = ""
            if char != " ":
                tokens.append(char)
    if buffer:
        tokens.append(buffer)
    return tokens


class CombatEnv:
    """Discrete one-second combat simulator with character rotation, queuing, and tactical synergy.

    Actions:
      0: Wait / Observe
      1-4: Basic, Heavy, Skill, Ultimate (buffers if currently in animation)
      5-7: Switch to Knight (Diluc), Mage (Lisa), or Rogue (Lynae)
      8: Dodge incoming boss hit
    """

    def __init__(self, stats: dict[str, Any], seed: int | None = None, start_active: int = -1):
        self.stats = stats
        self.rng = np.random.default_rng(seed)
        self.start_active = start_active
        self.observation_size = 0
        self.reset()
        self.observation_size = int(self._observation().size)

    def reset(self, seed: int | None = None, start_active: int = -1) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        boss = self.stats["boss"]
        self.boss_max_hp = float(boss["max_hp"])
        self.boss_hp = self.boss_max_hp
        self.max_time = int(float(boss.get("battle_time", 300)))
        self.time = 0

        # Uniform character start exploration
        if start_active >= 0:
            self.active = start_active % len(CHARACTER_KEYS)
        elif self.start_active >= 0:
            self.active = self.start_active % len(CHARACTER_KEYS)
        else:
            self.active = int(self.rng.integers(0, len(CHARACTER_KEYS)))

        self.hp = np.array(
            [float(self.stats["characters"][key]["max_hp"]) for key in CHARACTER_KEYS],
            dtype=np.float32,
        )
        # Individual cooldowns per character (3 characters x 4 moves = 12 total)
        self.all_cooldowns = np.zeros((len(CHARACTER_KEYS), 4), dtype=np.float32)

        self.attack_tokens: list[str] = []
        self.attack_elapsed = 0
        self.attack_second = -1
        self.attack_name = ""
        self.queued_attack: int | None = None

        self.boss_move = -1
        self.boss_tokens: list[str] = []
        self.boss_elapsed = 0
        self.boss_second = -1

        # Dodge state
        self.dodge_charges = DODGE_MAX_CHARGES
        self.dodge_regen_counter = 0
        self.dodge_queued = False

        # Parry, stun, and freeze states
        self.boss_stunned_ticks = 0
        self.player_frozen_ticks = 0

        self.last_damage = 0.0
        self.last_incoming_damage = 0.0
        self.total_damage = 0.0
        self.active_buffs: list[dict[str, Any]] = []
        self.ticks_since_last_switch = 10
        self.moves_since_last_switch = 0

        # Episode tracking metrics
        self.parry_count = 0
        self.dodge_count = 0
        self.switch_count = 0
        self.moves_used = {k: [0, 0, 0, 0] for k in CHARACTER_KEYS}
        self.buffs_triggered = 0
        self.heals_triggered = 0
        self.hp_healed_total = 0.0
        self.buffered_action_reward = 0.0

        self.done = False
        return self._observation()

    @property
    def cooldowns(self) -> np.ndarray:
        return self.all_cooldowns[self.active]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished episode.")
        action = int(action)
        if action < 0 or action >= ACTION_COUNT:
            raise ValueError(f"Action must be in [0, {ACTION_COUNT - 1}].")

        reward = 0.0
        self.last_damage = 0.0
        self.last_incoming_damage = 0.0

        prev_alive = (self.hp > 0).copy()

        # 1. Apply action
        action_reward = self._apply_action(action)
        reward += action_reward

        # 2. Check boss initiating attack (only when not stunned and idle)
        if self.time % 5 == 0 and self.boss_move < 0 and self.boss_stunned_ticks <= 0:
            self._start_boss_attack()

        # 3. Advance attacks, parry checks, damage calculations
        parried, dodged = self._advance_tick_resolution()
        reward += self.buffered_action_reward
        self.buffered_action_reward = 0.0

        if parried:
            self.parry_count += 1
            reward += 3.5  # Big tactical reward for parrying boss hit!
        elif dodged:
            self.dodge_count += 1
            reward += 2.0  # Tactical reward for active dodge

        # Check teammate death penalty
        now_alive = (self.hp > 0)
        deaths = int(np.sum(prev_alive & ~now_alive))
        if deaths > 0:
            reward -= 25.0 * deaths  # Heavy penalty for letting a teammate die

        # 4. Advance cooldowns across all characters (bench characters cool down too!)
        self.all_cooldowns = np.maximum(0.0, self.all_cooldowns - 1.0)

        # 5. Dodge recharge
        if self.dodge_charges < DODGE_MAX_CHARGES:
            self.dodge_regen_counter += 1
            if self.dodge_regen_counter >= DODGE_REGEN_TICKS:
                self.dodge_charges += 1
                self.dodge_regen_counter = 0

        self.time += 1
        self.ticks_since_last_switch += 1

        # Reward shaping: damage dealt vs incoming damage
        if self.last_damage > 0:
            dmg_reward = 30.0 * self.last_damage / max(1.0, self.boss_max_hp)
            # Hypercarry burst window amplification: when Diluc attacks under party buffs, amplify heavily!
            if self.active == 0 and len(self.active_buffs) > 0:
                dmg_reward += 5.0 * (self.last_damage / max(1.0, self.boss_max_hp)) * len(self.active_buffs)
            elif len(self.active_buffs) > 0:
                dmg_reward += 0.5 * len(self.active_buffs)
            reward += dmg_reward

        reward -= 5.0 * self.last_incoming_damage / max(1.0, self._team_max_hp())
        reward -= 0.005

        # Direct penalty for teammate death
        now_alive = (self.hp > 0)
        deaths = int(np.sum(prev_alive & ~now_alive))
        if deaths > 0:
            reward -= 35.0 * deaths

        if self.boss_hp <= 0:
            self.done = True
            reward += 150.0
        elif self.hp.sum() <= 0:
            self.done = True
            reward -= 100.0
        elif self.time >= self.max_time:
            self.done = True
            reward -= 100.0  # Defeat by timeout: boss survived, heavy penalty!

        info = {
            "time": self.time,
            "boss_hp": self.boss_hp,
            "active_character": CHARACTER_KEYS[self.active],
            "damage": self.last_damage,
            "incoming_damage": self.last_incoming_damage,
            "boss_attack": BOSS_MOVES[self.boss_move] if self.boss_move >= 0 else None,
            "boss_stunned_ticks": self.boss_stunned_ticks,
            "player_frozen_ticks": self.player_frozen_ticks,
            "parried": parried,
            "dodged": dodged,
            "parry_count": self.parry_count,
            "dodge_count": self.dodge_count,
            "switch_count": self.switch_count,
            "buffs_triggered": self.buffs_triggered,
            "heals_triggered": self.heals_triggered,
            "hp_healed_total": self.hp_healed_total,
            "active_buffs_count": len(self.active_buffs),
            "moves_used": self.moves_used,
            "total_damage": self.total_damage,
            "won": self.boss_hp <= 0,
        }
        return self._observation(), float(reward), self.done, info

    def action_mask(self) -> np.ndarray:
        mask = np.ones(ACTION_COUNT, dtype=np.float32)

        # Attacks 1-4: can be played or buffered as long as player is not frozen and cooldown <= 0
        if self.player_frozen_ticks > 0:
            mask[ACTION_BASIC : ACTION_ULTIMATE + 1] = 0.0
        else:
            for idx, cd in enumerate(self.all_cooldowns[self.active]):
                if cd > 0:
                    mask[ACTION_BASIC + idx] = 0.0

        # Switches 5-7: can switch anytime character is alive, not currently active, not frozen,
        # character has been on-field for at least 3 seconds (swap cooldown), and not mid-attack
        can_switch = (
            self.player_frozen_ticks <= 0
            and self.ticks_since_last_switch >= 3
            and not self.attack_tokens
        )
        for index, hp in enumerate(self.hp):
            if hp <= 0 or not can_switch:
                mask[ACTION_SWITCH_KNIGHT + index] = 0.0
        if can_switch:
            mask[ACTION_SWITCH_KNIGHT + self.active] = 0.0

        # Dodge 8: requires charge and not frozen
        if self.dodge_charges <= 0 or self.player_frozen_ticks > 0:
            mask[ACTION_DODGE] = 0.0

        return mask

    def _apply_action(self, action: int) -> float:
        if action == ACTION_WAIT:
            ready_moves = (
                not self.attack_tokens
                and self.player_frozen_ticks <= 0
                and any(cd <= 0.0 for cd in self.all_cooldowns[self.active])
            )
            return -0.20 if ready_moves else 0.0

        if ACTION_BASIC <= action <= ACTION_ULTIMATE:
            move_index = action - ACTION_BASIC
            move_name = MOVE_NAMES[move_index]
            active_key = CHARACTER_KEYS[self.active]
            char_data = self.stats["characters"][active_key]

            if self.player_frozen_ticks > 0:
                return -INVALID_ACTION_PENALTY

            # If already performing an attack animation: BUFFER the move without penalty!
            # Mirroring Godot's queue_or_play(anim): only latest valid attack input is queued.
            if self.attack_tokens:
                if self.all_cooldowns[self.active][move_index] > 0:
                    return 0.0  # Move on cooldown; ignore without penalty
                self.queued_attack = move_index
                return 0.05  # Positive feedback for buffering upcoming ability

            # Start attack immediately
            if self.all_cooldowns[self.active][move_index] > 0:
                cooldown_limit = float(char_data["moves"][move_name]["cooldown"])
                cooldown_ratio = self.all_cooldowns[self.active][move_index] / max(1.0, cooldown_limit)
                return -COOLDOWN_ACTION_PENALTY * (1.0 + cooldown_ratio)

            pattern_name = {
                "basic": "Basic (2s)",
                "heavy": "Heavy (4s)",
                "skill": "Skill (5s)",
                "ultimate": "Ultimate (8s)",
            }[move_name]
            self.attack_tokens = _tokens(char_data, pattern_name)
            self.attack_elapsed = 0
            self.attack_second = -1
            self.attack_name = move_name
            self.all_cooldowns[self.active][move_index] = float(char_data["moves"][move_name]["cooldown"])
            self.moves_used[active_key][move_index] += 1
            self.moves_since_last_switch += 1

            action_reward = 0.20
            if move_name == "heavy":
                action_reward += 0.40  # Encourage heavy attack damage rhythm and parry setup
            elif move_name == "skill":
                action_reward += 0.80
            elif move_name == "ultimate":
                action_reward += 1.20

            # Diluc hypercarry burst execution reward:
            if self.active == 0 and (self._get_team_buff("crit_rate") > 0 or self._get_team_buff("atk") > 0 or len(self.active_buffs) > 0):
                action_reward += 1.0
            # Encourage supports to pass field to Diluc rather than spamming low DPS basics during buff window:
            elif self.active != 0 and self.hp[0] > 0 and len(self.active_buffs) > 0 and move_name == "basic":
                action_reward -= 0.15

            if move_name in ("skill", "ultimate"):
                buff_count, buff_reward = self._apply_move_buffs(move_name)
                action_reward += buff_reward
            return action_reward

        if ACTION_SWITCH_KNIGHT <= action <= ACTION_SWITCH_ROGUE:
            target = action - ACTION_SWITCH_KNIGHT
            if target == self.active or self.hp[target] <= 0 or self.player_frozen_ticks > 0:
                return -INVALID_ACTION_PENALTY

            # Switching cancels current attack animation (character exits and new character drops in)
            self.attack_tokens = []
            self.attack_elapsed = 0
            self.attack_name = ""
            self.queued_attack = None

            # Rotation synergy bonus: only awarded if character actually fought/acted on field!
            switch_bonus = 0.0
            if self.ticks_since_last_switch >= 3 and self.moves_since_last_switch > 0:
                # 1. Hypercarry Rotation: Switching to Diluc (target == 0) while party buffs are active!
                has_dps_buffs = (self._get_team_buff("crit_rate") > 0 or self._get_team_buff("atk") > 0 or len(self.active_buffs) > 0)
                if target == 0 and has_dps_buffs:
                    switch_bonus += 3.0  # Massive synergy: Rotated to DPS during active Buff Window!

                # 2. Tactical Healer Rotation: Switching to Lisa (Mage, target == 1) when team needs heal and Skill is ready!
                if target == 1 and self.all_cooldowns[1][2] <= 0:
                    alive_r = [self.hp[i] / float(self.stats["characters"][k]["max_hp"]) for i, k in enumerate(CHARACTER_KEYS) if self.hp[i] > 0]
                    t_min = min(alive_r) if alive_r else 1.0
                    if t_min < 0.65:
                        switch_bonus += 3.5 * (1.0 - t_min)  # Emergency tactical swap to healer

                # 3. Rotating back to Support when Diluc (self.active == 0) has moves on cooldown:
                if self.active == 0 and any(cd > 0 for cd in self.all_cooldowns[0][1:]):
                    target_has_ready_moves = any(cd <= 0 for cd in self.all_cooldowns[target][2:])
                    if target_has_ready_moves:
                        switch_bonus += 1.5

                # 4. Emergency preservation bonus: encourage switching away from critically damaged character
                active_max_hp = float(self.stats["characters"][CHARACTER_KEYS[self.active]]["max_hp"])
                if self.hp[self.active] < 0.4 * active_max_hp and self.hp[target] > 0.6 * float(self.stats["characters"][CHARACTER_KEYS[target]]["max_hp"]):
                    switch_bonus += 2.5

                target_ready = sum(1 for cd in self.all_cooldowns[target] if cd <= 0.0)
                current_ready = sum(1 for cd in self.all_cooldowns[self.active] if cd <= 0.0)
                if target_ready > current_ready:
                    switch_bonus += 0.35
            elif self.moves_since_last_switch == 0:
                # Discourage dry switch-farming without using abilities
                switch_bonus = -0.50

            self.active = target
            self.switch_count += 1
            self.ticks_since_last_switch = 0
            self.moves_since_last_switch = 0
            return switch_bonus

        if action == ACTION_DODGE:
            if self.dodge_charges > 0 and self.player_frozen_ticks <= 0:
                self.dodge_queued = True
                self.dodge_charges -= 1
                self.dodge_regen_counter = 0
                return 0.05
            return -INVALID_ACTION_PENALTY

        return 0.0

    def _start_boss_attack(self) -> None:
        self.boss_move = int(self.rng.integers(0, len(BOSS_MOVES)))
        pattern_name = {
            "Up": "Atk 1 UP (9s)",
            "Summon": "Atk 2 Summon (10s)",
            "Taunt": "Atk 3 Taunt (9s)",
        }[BOSS_MOVES[self.boss_move]]
        self.boss_tokens = _tokens(self.stats["boss"], pattern_name)
        self.boss_elapsed = 0
        self.boss_second = -1

    def _advance_tick_resolution(self) -> tuple[bool, bool]:
        # Sample player token for this tick
        pending_player_token = 0.0
        if self.attack_tokens and self.attack_elapsed < len(self.attack_tokens):
            token_str = self.attack_tokens[self.attack_elapsed].strip()
            if token_str and token_str.upper() not in ("X", "_"):
                pending_player_token = float(token_str)

        # Sample boss token for this tick
        pending_boss_token = 0.0
        if self.boss_tokens and self.boss_elapsed < len(self.boss_tokens) and self.boss_stunned_ticks <= 0:
            token_str = self.boss_tokens[self.boss_elapsed].strip()
            if token_str and token_str.upper() not in ("X", "_"):
                pending_boss_token = float(token_str)

        parried = False
        dodged = False

        # Parry check
        if (
            pending_player_token > PARRY_TOKEN_THRESHOLD
            and pending_boss_token > 0.0
            and self.player_frozen_ticks <= 0
        ):
            parried = True
            pending_boss_token = 0.0
            self.boss_stunned_ticks = BOSS_STUN_TICKS
            self.boss_tokens = []
            self.boss_move = -1
            self.boss_elapsed = 0
            self.attack_tokens = []
            self.attack_elapsed = 0
            self.attack_name = ""
            self.player_frozen_ticks = PLAYER_FREEZE_TICKS

        # Dodge check: only triggers when player proactively queued dodge
        if not parried and pending_boss_token > 0.0 and self.dodge_queued:
            dodged = True
            pending_boss_token = 0.0

        # Clear dodge queue at tick boundary
        self.dodge_queued = False

        # Calculate and apply damages at tick boundary
        if pending_player_token > 0.0:
            character = self.stats["characters"][CHARACTER_KEYS[self.active]]
            base_atk = float(character.get("attack", 0)) + self._get_team_buff("atk")
            attack_val = pending_player_token * max(0.0, base_atk)
            crit_rate = float(character.get("crit_rate", 0)) + (self._get_team_buff("crit_rate") / 100.0)
            if self.rng.random() <= crit_rate:
                crit_dmg = float(character.get("crit_damage", 1)) + (self._get_team_buff("crit_damage") / 100.0)
                attack_val *= max(1.0, crit_dmg)
            boss_def = float(self.stats["boss"].get("defense", 0))
            dmg = max(0.0, attack_val * (1.0 - boss_def / 100.0))
            self.last_damage += dmg
            self.total_damage += dmg
            self.boss_hp = max(0.0, self.boss_hp - dmg)

        if pending_boss_token > 0.0:
            boss_data = self.stats["boss"]
            player_def = float(self.stats["characters"][CHARACTER_KEYS[self.active]].get("defense", 0)) + self._get_team_buff("def")
            boss_dmg = pending_boss_token * float(boss_data.get("attack", 0)) * (1.0 - min(90.0, player_def) / 100.0)
            dealt = min(float(self.hp[self.active]), max(0.0, boss_dmg))
            self.hp[self.active] -= dealt
            self.last_incoming_damage += dealt
            # Auto switch if active died
            if self.hp[self.active] <= 0:
                self.attack_tokens = []
                self.attack_elapsed = 0
                self.attack_name = ""
                self.queued_attack = None
                for next_idx in range(len(CHARACTER_KEYS)):
                    if self.hp[next_idx] > 0:
                        self.active = next_idx
                        break

        # Advance attack elapsed counters if not cancelled by parry
        if self.attack_tokens:
            self.attack_second = self.attack_elapsed
            self.attack_elapsed += 1
            if self.attack_elapsed >= len(self.attack_tokens):
                self.attack_tokens = []
                self.attack_elapsed = 0
                self.attack_name = ""
                # Start queued attack immediately if buffered and ready!
                if self.queued_attack is not None:
                    queued_idx = self.queued_attack
                    self.queued_attack = None
                    if self.all_cooldowns[self.active][queued_idx] <= 0:
                        queued_move = MOVE_NAMES[queued_idx]
                        active_key = CHARACTER_KEYS[self.active]
                        char_data = self.stats["characters"][active_key]
                        pattern_name = {
                            "basic": "Basic (2s)",
                            "heavy": "Heavy (4s)",
                            "skill": "Skill (5s)",
                            "ultimate": "Ultimate (8s)",
                        }[queued_move]
                        self.attack_tokens = _tokens(char_data, pattern_name)
                        self.attack_elapsed = 0
                        self.attack_second = -1
                        self.attack_name = queued_move
                        self.all_cooldowns[self.active][queued_idx] = float(char_data["moves"][queued_move]["cooldown"])
                        self.moves_used[active_key][queued_idx] += 1
                        self.moves_since_last_switch += 1
                        if queued_move in ("skill", "ultimate"):
                            _, buff_reward = self._apply_move_buffs(queued_move)
                            self.buffered_action_reward += buff_reward

        if self.boss_tokens and self.boss_stunned_ticks <= 0:
            self.boss_second = self.boss_elapsed
            self.boss_elapsed += 1
            if self.boss_elapsed >= len(self.boss_tokens):
                self.boss_tokens = []
                self.boss_move = -1
                self.boss_elapsed = 0

        # Countdown boss stun & player freeze
        if self.boss_stunned_ticks > 0:
            self.boss_stunned_ticks -= 1
        if self.player_frozen_ticks > 0:
            self.player_frozen_ticks -= 1

        # Countdown active buffs
        remaining_buffs: list[dict[str, Any]] = []
        for buff in self.active_buffs:
            buff["duration"] -= 1
            if buff["duration"] > 0:
                remaining_buffs.append(buff)
        self.active_buffs = remaining_buffs

        return parried, dodged

    def _apply_move_buffs(self, move_name: str) -> tuple[int, float]:
        active_key = CHARACTER_KEYS[self.active]
        char_data = self.stats["characters"][active_key]
        display_name = char_data.get("display_name", active_key)
        pattern_name = {
            "basic": "Basic (2s)",
            "heavy": "Heavy (4s)",
            "skill": "Skill (5s)",
            "ultimate": "Ultimate (8s)",
        }[move_name]
        buff_key = f"{display_name}:{pattern_name}"
        raw_buff = (
            self.stats.get("attack_buffs", {}).get(buff_key)
            or char_data.get("moves", {}).get(move_name, {}).get("buff", "")
        )
        specs = _parse_buff_spec(raw_buff)
        duration = 30 if move_name == "ultimate" else 6
        applied = 0
        extra_reward = 0.0

        for spec in specs:
            stat = spec["stat"]
            amt = float(spec["amount"])
            if stat == "hp":
                # Team-wide heal: restores missing HP to all living teammates up to max_hp
                alive_ratios_before: list[float] = []
                total_healed = 0.0
                for i, k in enumerate(CHARACTER_KEYS):
                    if self.hp[i] > 0:
                        char_max = float(self.stats["characters"][k]["max_hp"])
                        alive_ratios_before.append(self.hp[i] / char_max)
                        heal_possible = min(amt, max(0.0, char_max - self.hp[i]))
                        self.hp[i] = min(char_max, self.hp[i] + amt)
                        total_healed += heal_possible

                min_ratio = min(alive_ratios_before) if alive_ratios_before else 1.0
                self.heals_triggered += 1
                self.hp_healed_total += total_healed
                applied += 1

                # Heal reward scaling:
                # Measures how much of the nominal heal (amt) was actually absorbed (e.g. 140/200 vs 200/200)
                eff_ratio = min(2.5, total_healed / max(1.0, amt))
                if total_healed > 0.0:
                    urgency_mult = 1.0 + 3.0 * max(0.0, 1.0 - min_ratio)
                    heal_reward = 4.0 * eff_ratio * urgency_mult
                    extra_reward += heal_reward
                else:
                    # Overheal penalty: wasted healing cooldown when party is completely unharmed (>= 95% HP)
                    if min_ratio >= 0.95:
                        extra_reward -= 0.25
            else:
                self.active_buffs.append({"stat": stat, "amount": amt, "duration": duration})
                applied += 1
                self.buffs_triggered += 1
                extra_reward += 2.5  # Positive incentive for deploying party buffs

        return applied, extra_reward

    def _get_team_buff(self, stat_name: str) -> float:
        total = 0.0
        for buff in self.active_buffs:
            if buff["stat"] == stat_name:
                total += buff["amount"]
        return total

    def _team_max_hp(self) -> float:
        return sum(float(self.stats["characters"][k]["max_hp"]) for k in CHARACTER_KEYS)

    def _observation(self) -> np.ndarray:
        boss = self.stats["boss"]
        values: list[float] = [
            self.boss_hp / max(1.0, self.boss_max_hp),
            max(0.0, 1.0 - self.time / max(1, self.max_time)),
            self.time / max(1, self.max_time),
        ]
        # 1. 3 Character HP ratios
        hp_ratios: list[float] = []
        for i, k in enumerate(CHARACTER_KEYS):
            r = self.hp[i] / max(1.0, float(self.stats["characters"][k]["max_hp"]))
            hp_ratios.append(r)
            values.append(r)

        # 2. 3 Active character one-hot
        for i in range(3):
            values.append(float(self.active == i))

        # 3. 15 Post-buff effective stats (Knight, Mage, Rogue x 5 stats)
        buff_atk = self._get_team_buff("atk")
        buff_def = self._get_team_buff("def")
        buff_cr = self._get_team_buff("crit_rate")
        buff_cd = self._get_team_buff("crit_damage")

        for k in CHARACTER_KEYS:
            c_data = self.stats["characters"][k]
            eff_atk = (float(c_data.get("attack", 0.0)) + buff_atk) / 100.0
            eff_def = (float(c_data.get("defense", 0.0)) + buff_def) / 100.0
            eff_cr = min(1.0, float(c_data.get("crit_rate", 0.0)) + (buff_cr / 100.0))
            eff_cd = (float(c_data.get("crit_damage", 1.0)) + (buff_cd / 100.0)) / 3.0
            dps_pot = (eff_atk * 100.0 * (1.0 + eff_cr * (eff_cd * 3.0))) / 150.0
            values.extend([eff_atk, eff_def, eff_cr, eff_cd, dps_pot])

        # 4. 5 Team tactical & burst indicators
        alive_ratios = [r for r in hp_ratios if r > 0.0]
        min_hp_ratio = min(alive_ratios) if alive_ratios else 0.0
        values.append(min_hp_ratio)
        values.append(1.0 if min_hp_ratio < 0.5 else 0.0)

        max_buff_dur = max((b["duration"] for b in self.active_buffs), default=0)
        values.append(min(1.0, float(max_buff_dur) / 30.0))

        is_dps_active = float(self.active == 0)
        values.append(is_dps_active)

        has_dps_buffs = (buff_cr > 0.0 or buff_atk > 0.0)
        dps_ready_to_burst = float(is_dps_active == 0.0 and has_dps_buffs and self.hp[0] > 0.0)
        values.append(dps_ready_to_burst)

        # 5. 12 Cooldowns across all characters (Knight, Mage, Rogue x 4 moves)
        for char_idx in range(len(CHARACTER_KEYS)):
            for cd in self.all_cooldowns[char_idx]:
                values.append(min(1.0, float(cd) / 32.0))

        # 6. 10 Combat state & boss
        values.extend([
            self.last_damage / max(1.0, self.boss_max_hp),
            self.last_incoming_damage / max(1.0, self._team_max_hp()),
            float(bool(self.attack_tokens)),
            min(1.0, self.attack_elapsed / MAX_PATTERN_SLOTS),
        ])
        values.extend(float(self.boss_move == index) for index in range(3))
        values.append(min(1.0, self.boss_elapsed / MAX_PATTERN_SLOTS))
        values.append(float(self.boss_move >= 0))
        values.append(min(1.0, self.boss_stunned_ticks / 4.0))

        # 7. 6 Defense & buffs
        values.append(min(1.0, self.player_frozen_ticks / 1.0))
        values.append(min(1.0, self.dodge_charges / float(DODGE_MAX_CHARGES)))
        values.append(min(2.0, buff_atk / 50.0))
        values.append(min(2.0, buff_def / 50.0))
        values.append(min(1.0, buff_cr / 100.0))
        values.append(min(1.0, buff_cd / 100.0))

        # 8. 1 Boss immediate threat flag
        boss_threat = 0.0
        if self.boss_tokens and self.boss_stunned_ticks <= 0:
            for check_idx in (self.boss_elapsed, self.boss_elapsed + 1):
                if check_idx < len(self.boss_tokens):
                    tok = self.boss_tokens[check_idx].strip()
                    if tok and tok.upper() not in ("X", "_", ""):
                        boss_threat = 1.0
                        break
        values.append(boss_threat)

        # 9. 32 Look-ahead padded attack tokens
        values.extend(self._padded_tokens(self.attack_tokens, self.attack_elapsed - 1))
        values.extend(self._padded_tokens(self.boss_tokens, self.boss_elapsed - 1))
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _padded_tokens(tokens: list[str], current: int) -> list[float]:
        result: list[float] = []
        for index in range(MAX_PATTERN_SLOTS):
            if index >= len(tokens):
                result.append(0.0)
            elif index <= current:
                result.append(0.0)
            else:
                token = tokens[index].strip()
                result.append(0.0 if token.upper() in ("X", "_", "") else min(1.0, float(token) / 10.0))
        return result
