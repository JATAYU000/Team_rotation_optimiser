extends Node

const STATS_PATH        := "res://data/combat_stats.json"
const SAVED_STATS_PATH  := "user://combat_stats.json"
const TITLE_SCENE       := "res://SimulatorConfig.tscn"

# ── Parry / Dodge / Buff constants ─────────────────────────────────────────────
const PARRY_TOKEN_THRESHOLD := 2.0   # player outgoing multiplier > this triggers parry
const BOSS_STUN_TICKS       := 4     # ticks the boss cannot attack after a parry
const PLAYER_FREEZE_TICKS   := 1     # ticks the player cannot act after a parry
const DODGE_MAX_CHARGES     := 3
const DODGE_REGEN_TICKS     := 8     # ticks per charge regeneration
const DODGE_STEP_BACK       := 1.5   # units the character steps back on dodge
const SKILL_BUFF_DURATION   := 6     # ticks for skill buffs
const ULT_BUFF_DURATION     := 30    # ticks for ultimate buffs

# ── Combat state ─────────────────────────────────────────────────────────────
var stats: Dictionary = {}
var character_hp: Dictionary = {}
var boss_hp: float
var battle_time_left: float
var battle_finished := false

# ── Tick-based attack tracking ───────────────────────────────────────────────
var last_player_state := "Idle"
var last_boss_state   := "Idle"
var switch_timer      := 0.0
var character_cooldowns := {
	"Knight": {"basic": 0.0, "heavy": 0.0, "skill": 0.0, "ultimate": 0.0},
	"Mage":   {"basic": 0.0, "heavy": 0.0, "skill": 0.0, "ultimate": 0.0},
	"Rogue":  {"basic": 0.0, "heavy": 0.0, "skill": 0.0, "ultimate": 0.0}
}
var move_cooldowns: Dictionary:
	get:
		var c_key := _character_key(player_manager.current_char) if player_manager != null else "Knight"
		return character_cooldowns.get(c_key, character_cooldowns["Knight"])

var player_attack_name   := ""
var player_attack_tokens: Array[String] = []
var player_attack_tick   := 0        # index of the next token to resolve

var boss_attack_name     := ""
var boss_attack_tokens: Array[String] = []
var boss_attack_tick     := 0

# Pending values resolved at tick boundary
var _pending_player_token := 0.0
var _pending_boss_token   := 0.0
var last_damage           := 0.0
var last_incoming_damage  := 0.0
var last_player_hit_crit  := false

# ── Dodge state ───────────────────────────────────────────────────────────────
var dodge_charges       := DODGE_MAX_CHARGES
var dodge_regen_counter := 0
var dodge_queued        := false     # set true when Shift is pressed this frame

# ── Parry / stun / freeze state ───────────────────────────────────────────────
var boss_stunned_ticks  := 0
var player_frozen_ticks := 0

# ── Buff state ────────────────────────────────────────────────────────────────
var attack_buffs: Dictionary = {}
var active_buffs: Array[Dictionary] = [] # entries: { "type": String, "amount": float, "duration": int, "label": String }

# ── Scene references ──────────────────────────────────────────────────────────
@onready var player_manager = $Player
@onready var boss           = $Skeleton_Warrior

# ── HUD nodes ─────────────────────────────────────────────────────────────────
var boss_bar: ProgressBar
var boss_label: Label
var timer_label: Label
var player_bar: ProgressBar
var player_label: Label
var move_buttons: Dictionary = {}
var outcome_panel: ColorRect
var dodge_charge_label: Label
var dodge_regen_bar: ProgressBar
var active_buff_label: Label
var hud_layer: CanvasLayer
var float_label_layer: CanvasLayer

# ── RL state ──────────────────────────────────────────────────────────────────
var rl_simulation     := false
var rl_udp: PacketPeerUDP
var rl_request_id     := 0
var rl_waiting        := false
var rl_request_timer  := 0.0
var rl_decision_timer := 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

func _exit_tree() -> void:
	if rl_udp != null:
		rl_udp.close()
		rl_udp = null


func _ready() -> void:
	stats = _load_stats()
	attack_buffs = stats.get("attack_buffs", {}) as Dictionary
	var boss_stats: Dictionary = stats.get("boss", {})
	boss_hp          = float(boss_stats.get("max_hp", 5000))
	battle_time_left = float(boss_stats.get("battle_time", 300.0))
	for character_name in ["Knight", "Mage", "Rogue"]:
		var character_stats: Dictionary = stats.get("characters", {}).get(character_name, {})
		character_hp[character_name] = float(character_stats.get("max_hp", 1000))

	for c in [player_manager.char_1, player_manager.char_2, player_manager.char_3]:
		if c != null and c.has_signal("attack_started"):
			c.attack_started.connect(_on_character_attack_started.bind(c))

	rl_simulation = get_tree().root.get_meta("rl_simulation", false)
	if rl_simulation:
		get_tree().root.set_meta("rl_simulation", false)
		rl_udp = PacketPeerUDP.new()
		rl_udp.bind(0)
		rl_udp.connect_to_host("127.0.0.1", 8765)

	_register_dodge_key()
	_build_hud()
	_start_tick_timer()


func _register_dodge_key() -> void:
	# Bind Shift to the "dodge" action if no key is already mapped.
	if not InputMap.has_action("dodge"):
		InputMap.add_action("dodge")
	if InputMap.action_get_events("dodge").is_empty():
		var ev := InputEventKey.new()
		ev.physical_keycode = KEY_SHIFT
		InputMap.action_add_event("dodge", ev)


func _start_tick_timer() -> void:
	var timer := Timer.new()
	timer.name       = "TickTimer"
	timer.wait_time  = 1.0
	timer.one_shot   = false
	timer.timeout.connect(_resolve_tick)
	add_child(timer)
	timer.start()


# ═══════════════════════════════════════════════════════════════════════════════
# PER-FRAME (_process)  —  input capture, RL polling, boss animation monitoring
# ═══════════════════════════════════════════════════════════════════════════════

func _process(delta: float) -> void:
	if battle_finished:
		return

	# Advance the visible battle timer each frame.
	battle_time_left = maxf(0.0, battle_time_left - delta)

	if rl_simulation:
		_process_rl_agent(delta)

	# Detect boss animation state transitions to load attack token arrays.
	_monitor_boss_attack_state()

	_update_hud()


func _unhandled_input(event: InputEvent) -> void:
	if battle_finished or player_frozen_ticks > 0:
		return
	if event.is_action_pressed("dodge") and dodge_charges > 0:
		dodge_queued = true


# ═══════════════════════════════════════════════════════════════════════════════
# TICK RESOLVER — fires exactly once per second via the TickTimer
# ═══════════════════════════════════════════════════════════════════════════════

func _resolve_tick() -> void:
	if battle_finished:
		return

	last_damage          = 0.0
	last_incoming_damage = 0.0
	_pending_player_token = 0.0
	_pending_boss_token   = 0.0

	# 1. Decrement ability cooldowns across all characters and switch timer.
	switch_timer = maxf(0.0, switch_timer - 1.0)
	for c_key in character_cooldowns:
		for move_name in character_cooldowns[c_key]:
			character_cooldowns[c_key][move_name] = maxf(0.0, character_cooldowns[c_key][move_name] - 1.0)

	# 2. Sample the player's outgoing token for this tick.
	if player_attack_tokens.size() > 0 and player_attack_tick < player_attack_tokens.size():
		var tok := player_attack_tokens[player_attack_tick].strip_edges()
		if not tok.is_empty() and tok.to_upper() != "X" and tok != "_":
			_pending_player_token = tok.to_float()

	# 3. Sample the boss's outgoing token for this tick.
	if boss_attack_tokens.size() > 0 and boss_attack_tick < boss_attack_tokens.size() \
			and boss_stunned_ticks <= 0:
		var tok := boss_attack_tokens[boss_attack_tick].strip_edges()
		if not tok.is_empty() and tok.to_upper() != "X" and tok != "_":
			_pending_boss_token = tok.to_float()

	# 4. PARRY check — player token > threshold while boss deals damage this tick.
	var parried := false
	if _pending_player_token > PARRY_TOKEN_THRESHOLD \
			and _pending_boss_token > 0.0 \
			and player_frozen_ticks <= 0:
		parried = true
		_pending_boss_token = 0.0          # negate incoming damage
		_trigger_boss_stun()
		# Trigger parry animation and stop attack token progression
		player_attack_name = ""
		player_attack_tokens.clear()
		player_attack_tick = 0
		var active = player_manager.current_char
		if active != null and active.has_method("play_parry"):
			active.play_parry()
		elif active != null and active.has_method("force_idle"):
			active.force_idle()
		player_frozen_ticks = PLAYER_FREEZE_TICKS
		_set_player_frozen(true)
		_spawn_center_label("Parried!", Color(1.0, 0.55, 0.1))

	# 5. DODGE check — only runs if parry did not already negate incoming damage.
	if not parried and dodge_queued:
		if _pending_boss_token > 0.0:
			_pending_boss_token = 0.0      # negate incoming damage
			_spawn_center_label("Perfect Dodge!", Color(0.3, 0.9, 1.0))
		# Consume charge and step back regardless of whether a hit was dodged.
		dodge_charges       = maxi(0, dodge_charges - 1)
		dodge_regen_counter = 0
		_tween_dodge_step_back(player_manager.current_char)
	dodge_queued = false

	# 6. Compute final damage values from the sampled tokens.
	var player_damage := 0.0
	if _pending_player_token > 0.0:
		player_damage = _calculate_player_damage(_pending_player_token)

	var boss_damage := 0.0
	if _pending_boss_token > 0.0:
		boss_damage = _calculate_boss_damage(_pending_boss_token)

	# 7. Apply damages (both happen at the same tick boundary).
	if player_damage > 0.0:
		_apply_player_hit_tick(player_damage)

	if boss_damage > 0.0:
		_apply_boss_damage_tick(boss_damage)

	# 8. Spawn floating damage labels.
	if player_damage > 0.0:
		if last_player_hit_crit:
			_spawn_world_label("CRIT -%d!" % roundi(player_damage), boss, Color(1.0, 0.28, 0.12), true)
		else:
			_spawn_world_label("-%d" % roundi(player_damage), boss, Color(0.95, 0.85, 0.3), false)
	if boss_damage > 0.0:
		_spawn_world_label("-%d" % roundi(boss_damage), player_manager.current_char, Color(0.9, 0.25, 0.25))

	# 9. Advance player attack tick index.
	if player_attack_tokens.size() > 0:
		player_attack_tick += 1
		if player_attack_tick >= player_attack_tokens.size():
			player_attack_name = ""
			player_attack_tokens.clear()
			player_attack_tick = 0

	# 10. Advance boss attack tick index (skip while boss is stunned).
	if boss_attack_tokens.size() > 0 and boss_stunned_ticks <= 0:
		boss_attack_tick += 1
		if boss_attack_tick >= boss_attack_tokens.size():
			boss_attack_name = ""
			boss_attack_tokens.clear()
			boss_attack_tick = 0

	# 11. Countdown boss stun and player freeze.
	if boss_stunned_ticks > 0:
		boss_stunned_ticks -= 1

	if player_frozen_ticks > 0:
		player_frozen_ticks -= 1
		if player_frozen_ticks <= 0:
			_set_player_frozen(false)

	# 12. Regenerate one dodge charge every DODGE_REGEN_TICKS ticks.
	if dodge_charges < DODGE_MAX_CHARGES:
		dodge_regen_counter += 1
		if dodge_regen_counter >= DODGE_REGEN_TICKS:
			dodge_charges       += 1
			dodge_regen_counter  = 0

	# 13. Tick active buffs and remove expired ones.
	var surviving_buffs: Array[Dictionary] = []
	for b in active_buffs:
		b["duration"] = int(b.get("duration", 0)) - 1
		if int(b["duration"]) > 0:
			surviving_buffs.append(b)
	active_buffs = surviving_buffs

	# 14. Timeout check (time_left is decremented in _process, checked here).
	if not battle_finished and battle_time_left <= 0.0:
		_finish_battle(false)

	# 15. Keep HUD immediately updated with latest tick numbers.
	_update_hud()


# ═══════════════════════════════════════════════════════════════════════════════
# DAMAGE CALCULATION & BUFFS
# ═══════════════════════════════════════════════════════════════════════════════

func _get_team_buff(stat_name: String) -> float:
	var total := 0.0
	for b in active_buffs:
		if b.get("type", "") == stat_name:
			total += float(b.get("amount", 0.0))
	return total


func _calculate_player_damage(multiplier: float) -> float:
	last_player_hit_crit = false
	var active = player_manager.current_char
	if active == null:
		return 0.0
	var character_key  := _character_key(active)
	var char_stats: Dictionary = stats.get("characters", {}).get(character_key, {})
	var base_attack    := float(char_stats.get("attack", 0.0)) + _get_team_buff("atk")
	var damage         := multiplier * maxf(0.0, base_attack)
	var crit_rate      := float(char_stats.get("crit_rate", 0.0)) + (_get_team_buff("crit_rate") / 100.0)
	var crit_damage    := float(char_stats.get("crit_damage", 1.0)) + (_get_team_buff("crit_damage") / 100.0)
	if randf() <= crit_rate:
		last_player_hit_crit = true
		damage *= maxf(1.0, crit_damage)
	var defense_pct := float(stats.get("boss", {}).get("defense", 0.0))
	return maxf(0.0, damage * (1.0 - defense_pct / 100.0))


func _calculate_boss_damage(multiplier: float) -> float:
	var active = player_manager.current_char
	if active == null or not player_manager.is_character_alive(active):
		return 0.0
	var character_key := _character_key(active)
	var defense_pct   := float(stats.get("characters", {}).get(character_key, {}).get("defense", 0)) + _get_team_buff("def")
	var boss_atk      := float(stats.get("boss", {}).get("attack", 0.0))
	return maxf(0.0, multiplier * boss_atk * (1.0 - defense_pct / 100.0))


func _parse_buff_string(raw: Variant) -> Array[Dictionary]:
	var result: Array[Dictionary] = []
	if raw == null:
		return result
	# Support legacy object format { "stat": 4.0, "amount": ... }
	if raw is Dictionary:
		var stat_val = raw.get("stat")
		var amt_val = raw.get("amount")
		if stat_val != null and amt_val != null:
			var stat_map = {1: "atk", 2: "def", 3: "heal", 4: "crit_rate", 5: "crit_damage"}
			var st_name = stat_map.get(int(stat_val), "")
			if st_name != "" and str(amt_val).strip_edges() != "":
				result.append({"type": st_name, "amount": float(amt_val)})
		return result

	var text := str(raw).strip_edges()
	if text.is_empty():
		return result

	# Regex matching abbreviations: H/HP/HEAL, CR/CRATE, CD/CDMG, A/ATK, D/DEF
	var regex := RegEx.new()
	regex.compile("(?i)(crit_?rate|crate|cr|crit_?dmg|crit_?damage|cdmg|cd|attack|atk|a|defense|def|d|heal|hp|h)\\s*[:=+]?\\s*([+-]?\\d+(?:\\.\\d+)?)\\s*%?")
	for m in regex.search_all(text):
		var tag := m.get_string(1).to_lower().replace("_", "")
		var val := m.get_string(2).to_float()
		var buff_type := ""
		if tag in ["h", "hp", "heal"]:
			buff_type = "heal"
		elif tag in ["cr", "crate", "critrate"]:
			buff_type = "crit_rate"
		elif tag in ["cd", "cdmg", "critdmg", "critdamage"]:
			buff_type = "crit_damage"
		elif tag in ["a", "atk", "attack"]:
			buff_type = "atk"
		elif tag in ["d", "def", "defense"]:
			buff_type = "def"
		if buff_type != "":
			result.append({"type": buff_type, "amount": val})
	return result


func _apply_move_buffs(active: Node, move_name: String) -> void:
	if active == null or (move_name != "skill" and move_name != "ultimate"):
		return
	var character_key := _character_key(active)
	var char_stats: Dictionary = stats.get("characters", {}).get(character_key, {})
	var display_name: String = char_stats.get("display_name", character_key).to_upper()
	var pattern_name: String = "Skill (5s)" if move_name == "skill" else "Ultimate (8s)"
	var buff_key := display_name + ":" + pattern_name
	var raw_buff = attack_buffs.get(buff_key, null)
	if raw_buff == null:
		# Fallback check using character_key if display_name differed
		buff_key = character_key.to_upper() + ":" + pattern_name
		raw_buff = attack_buffs.get(buff_key, null)
	if raw_buff == null:
		return

	var parsed_list := _parse_buff_string(raw_buff)
	if parsed_list.is_empty():
		return

	var duration := SKILL_BUFF_DURATION if move_name == "skill" else ULT_BUFF_DURATION
	var applied_summaries: Array[String] = []

	for b in parsed_list:
		var b_type: String = b["type"]
		var b_amt: float = b["amount"]
		if b_type == "heal":
			# HP +x to all char (heals lost HP, does not exceed max_hp)
			for char_k in ["Knight", "Mage", "Rogue"]:
				var cur: float = float(character_hp.get(char_k, 0.0))
				var mx: float = float(stats.get("characters", {}).get(char_k, {}).get("max_hp", 1000.0))
				if cur > 0.0:
					character_hp[char_k] = minf(mx, cur + b_amt)
			_spawn_world_label("+%d HP" % roundi(b_amt), active, Color(0.2, 1.0, 0.4))
			applied_summaries.append("+%d HP (Heal)" % roundi(b_amt))
		else:
			var display_tag := ""
			match b_type:
				"atk":
					display_tag = "+%d ATK" % roundi(b_amt)
				"def":
					display_tag = "+%d%% DEF" % roundi(b_amt)
				"crit_rate":
					display_tag = "+%d%% CR" % roundi(b_amt)
				"crit_damage":
					display_tag = "+%d%% CD" % roundi(b_amt)
			active_buffs.append({
				"type": b_type,
				"amount": b_amt,
				"duration": duration,
				"label": display_tag
			})
			applied_summaries.append("%s (%ds)" % [display_tag, duration])

	if not applied_summaries.is_empty():
		_spawn_center_label("Buff: " + ", ".join(applied_summaries), Color(0.9, 0.8, 0.2))
	_update_hud()


func _apply_player_hit_tick(damage: float) -> void:
	boss_hp     = maxf(0.0, boss_hp - damage)
	last_damage += damage
	_update_hud()
	if boss_hp <= 0.0:
		_finish_battle(true)


func _apply_boss_damage_tick(damage: float) -> void:
	var active = player_manager.current_char
	if active == null or not player_manager.is_character_alive(active):
		return
	var character_key := _character_key(active)
	var current_hp    := float(character_hp.get(character_key, 0.0))
	var dealt         := minf(current_hp, damage)
	character_hp[character_key] = maxf(0.0, current_hp - dealt)
	last_incoming_damage       += dealt
	_update_hud()
	if character_hp[character_key] <= 0.0:
		player_attack_name = ""
		player_attack_tokens.clear()
		player_attack_tick = 0
		player_manager.kill_character(active)
		_after_character_death()    # fire-and-forget coroutine


func _after_character_death() -> void:
	await get_tree().create_timer(0.8).timeout
	if not player_manager.switch_to_next_alive():
		_finish_battle(false)


# ═══════════════════════════════════════════════════════════════════════════════
# PARRY & DODGE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

func _trigger_boss_stun() -> void:
	# Interrupt the current boss attack token sequence.
	boss_attack_name = ""
	boss_attack_tokens.clear()
	boss_attack_tick    = 0
	boss_stunned_ticks  = BOSS_STUN_TICKS
	# Travel to "Hit"; the pre-built Hit → stun → Idle transition chain fires.
	if boss == null:
		return
	var tree = boss.get_node_or_null("AnimationTree")
	if tree == null:
		return
	var playback = tree.get("parameters/StateMachine/playback")
	if playback != null:
		playback.travel("Hit")


func _tween_dodge_step_back(char_node: Node3D) -> void:
	if char_node == null:
		return
	var origin := char_node.global_position
	# Move strictly in the direction opposite to the boss.
	var away_dir := Vector3.ZERO
	if boss != null:
		away_dir = char_node.global_position - boss.global_position
		away_dir.y = 0.0
	if away_dir.length_squared() > 0.001:
		away_dir = away_dir.normalized()
	else:
		away_dir = Vector3(0, 0, 1)
	var back   := origin + away_dir * DODGE_STEP_BACK
	var tween  := create_tween()
	tween.tween_property(char_node, "global_position", back,   0.3)
	tween.tween_property(char_node, "global_position", origin, 0.4)


func _set_player_frozen(frozen: bool) -> void:
	var active = player_manager.current_char
	if active != null and active.has_method("set_frozen"):
		active.set_frozen(frozen)


# ═══════════════════════════════════════════════════════════════════════════════
# BOSS ATTACK MONITORING  (called every frame, loads tokens only)
# ═══════════════════════════════════════════════════════════════════════════════

func _monitor_boss_attack_state() -> void:
	if boss == null:
		return
	var tree = boss.get_node_or_null("AnimationTree")
	if tree == null:
		return
	var state_machine = tree.get("parameters/StateMachine/playback")
	if state_machine == null:
		return
	var state: String = state_machine.get_current_node()
	# Load tokens when the boss transitions into a new attack animation.
	if state != last_boss_state and state in ["Summon", "Taunt", "Up"]:
		if boss_stunned_ticks <= 0 and boss_attack_tokens.is_empty():
			_start_boss_attack(state)
	last_boss_state = state


func _start_boss_attack(move_name: String) -> void:
	boss_attack_name   = move_name
	boss_attack_tokens = _get_pattern_tokens(stats.get("boss", {}), move_name)
	boss_attack_tick   = 0


# ═══════════════════════════════════════════════════════════════════════════════
# PLAYER ATTACK SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════

func _start_player_attack(move_name: String, active: Node) -> void:
	var character_key  := _character_key(active)
	var char_stats: Dictionary = stats.get("characters", {}).get(character_key, {})
	var move: Dictionary = char_stats.get("moves", {}).get(move_name, {})
	character_cooldowns[character_key][move_name] = float(move.get("cooldown", 0.0))
	player_attack_name   = move_name
	player_attack_tokens = _get_pattern_tokens(char_stats, move_name)
	player_attack_tick   = 0
	_apply_move_buffs(active, move_name)


func _get_move_name_for_character(character_node: Node, animation_name: String) -> String:
	if character_node == player_manager.char_1:
		# Diluc / Knight
		return {
			"Diag": "basic",
			"Load": "heavy",
			"Raise": "skill",
			"Chop 2": "ultimate"
		}.get(animation_name, "")
	elif character_node == player_manager.char_2:
		# Lisa / Mage
		return {
			"Shoot": "basic",
			"Aim": "heavy",
			"Yo": "skill",
			"Long": "ultimate"
		}.get(animation_name, "")
	elif character_node == player_manager.char_3:
		# Lynae / Rogue
		return {
			"Aim": "basic",
			"Reload": "heavy",
			"Spin": "skill",
			"Look": "ultimate"
		}.get(animation_name, "")
	return ""


func _get_animation_for_move(character_node: Node, move_name: String) -> String:
	if character_node == player_manager.char_1:
		return {"basic": "Diag", "heavy": "Load", "skill": "Raise", "ultimate": "Chop 2"}.get(move_name, "")
	elif character_node == player_manager.char_2:
		return {"basic": "Shoot", "heavy": "Aim", "skill": "Yo", "ultimate": "Long"}.get(move_name, "")
	elif character_node == player_manager.char_3:
		return {"basic": "Aim", "heavy": "Reload", "skill": "Spin", "ultimate": "Look"}.get(move_name, "")
	return ""


func _on_character_attack_started(animation_name: String, character_node: Node) -> void:
	if battle_finished or not player_manager.is_character_alive(character_node):
		return
	if character_node != player_manager.current_char:
		return
	var move_name := _get_move_name_for_character(character_node, animation_name)
	if move_name != "":
		_start_player_attack(move_name, character_node)


func _on_player_attack_started(animation_name: String) -> void:
	_on_character_attack_started(animation_name, player_manager.char_1)


# ═══════════════════════════════════════════════════════════════════════════════
# FLOATING DAMAGE LABELS
# ═══════════════════════════════════════════════════════════════════════════════

func _spawn_world_label(text: String, world_node: Node3D, color: Color, large: bool = false) -> void:
	if world_node == null or float_label_layer == null:
		return
	var camera := get_viewport().get_camera_3d()
	if camera == null:
		return
	# Offset upward in world space so the label appears above the model.
	var world_pos   := world_node.global_position + Vector3(0, 1.8, 0)
	var screen_pos  := camera.unproject_position(world_pos)
	_create_floating_label(text, screen_pos, color, large)


func _spawn_center_label(text: String, color: Color) -> void:
	if float_label_layer == null:
		return
	var vp_size    := get_viewport().get_visible_rect().size
	var screen_pos := vp_size * 0.5
	_create_floating_label(text, screen_pos, color, true)


func _create_floating_label(text: String, screen_pos: Vector2, color: Color, large: bool) -> void:
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", 30 if large else 20)
	label.add_theme_color_override("font_color", color)
	label.add_theme_color_override("font_outline_color", Color(0.0, 0.0, 0.0, 0.85))
	label.add_theme_constant_override("outline_size", 4)
	# Centre the label horizontally on the target point.
	label.position = screen_pos + Vector2(-60, 0)
	label.z_index  = 10
	float_label_layer.add_child(label)

	var rise   := -60.0 if large else -50.0
	var target := label.position + Vector2(0, rise)
	var tween  := create_tween()
	tween.tween_property(label, "position", target, 0.9)
	tween.parallel().tween_property(label, "modulate",
			Color(color.r, color.g, color.b, 0.0), 0.9)
	tween.tween_callback(label.queue_free)


# ═══════════════════════════════════════════════════════════════════════════════
# RL AGENT (unchanged logic, variable names updated)
# ═══════════════════════════════════════════════════════════════════════════════

func _process_rl_agent(delta: float) -> void:
	if rl_udp == null:
		return
	rl_request_timer  += delta
	rl_decision_timer += delta
	while rl_udp.get_available_packet_count() > 0:
		var response_text := rl_udp.get_packet().get_string_from_utf8()
		var response: Variant = JSON.parse_string(response_text)
		if response is Dictionary and int(response.get("id", -1)) == rl_request_id:
			rl_waiting       = false
			rl_request_timer = 0.0
			var action: int  = int(response.get("action", 0))
			print("[RL Agent] Received Action %d from policy server" % action)
			_apply_rl_action(action)
	if not rl_waiting and rl_decision_timer >= 1.0:
		rl_decision_timer = 0.0
		rl_request_id    += 1
		rl_request_timer  = 0.0
		var request := {
			"id":          rl_request_id,
			"observation": _rl_observation(),
			"action_mask": _rl_action_mask()
		}
		rl_udp.put_packet(JSON.stringify(request).to_utf8_buffer())
		rl_waiting = true
	elif rl_waiting and rl_request_timer >= 2.0:
		rl_waiting       = false
		rl_request_timer = 0.0


func _apply_rl_action(action: int) -> void:
	if action == 0:
		return
	if action >= 1 and action <= 4:
		var move_names := ["basic", "heavy", "skill", "ultimate"]
		_on_move_pressed(move_names[action - 1])
		return
	if action >= 5 and action <= 7 and not player_manager.is_switching:
		var targets := [player_manager.char_1, player_manager.char_2, player_manager.char_3]
		var target   = targets[action - 5]
		if player_manager.is_character_alive(target) and target != player_manager.current_char and switch_timer <= 0.0:
			player_attack_name = ""
			player_attack_tokens.clear()
			player_attack_tick = 0
			switch_timer = 3.0
			player_manager.swap_to(target)
		return
	if action == 8:
		if dodge_charges > 0 and player_frozen_ticks <= 0:
			dodge_queued = true
		return


func _rl_action_mask() -> Array[float]:
	var mask: Array[float] = []
	mask.append(1.0) # 0: Wait
	var active = player_manager.current_char
	var active_key := _character_key(active) if active != null else "Knight"
	var active_cds: Dictionary = character_cooldowns.get(active_key, {})
	for move_name in ["basic", "heavy", "skill", "ultimate"]:
		# Allow playing or buffering if cooldown <= 0 and player not frozen
		mask.append(1.0 if active != null \
				and player_manager.is_character_alive(active) \
				and not player_manager.is_switching \
				and player_frozen_ticks <= 0 \
				and float(active_cds.get(move_name, 0.0)) <= 0.0 else 0.0)
	for character in [player_manager.char_1, player_manager.char_2, player_manager.char_3]:
		# Allow switching anytime character is alive, not current, not mid-swap, and swap cooldown <= 0
		mask.append(1.0 if player_manager.is_character_alive(character) \
				and character != player_manager.current_char \
				and not player_manager.is_switching \
				and player_frozen_ticks <= 0 \
				and switch_timer <= 0.0 else 0.0)
	# 8: Dodge
	mask.append(1.0 if dodge_charges > 0 and player_frozen_ticks <= 0 else 0.0)
	return mask


func _rl_observation() -> Array[float]:
	var characters := [player_manager.char_1, player_manager.char_2, player_manager.char_3]
	var max_time   := maxf(1.0, float(stats.get("boss", {}).get("battle_time", 300.0)))
	var boss_max   := float(stats.get("boss", {}).get("max_hp", 1.0))
	var values: Array[float] = [
		boss_hp / maxf(1.0, boss_max),
		battle_time_left / max_time,
		maxf(0.0, 1.0 - battle_time_left / max_time)
	]
	# 1. 3 character HP ratios
	var hp_ratios: Array[float] = []
	for index in range(3):
		var key: String = _character_key(characters[index])
		var max_hp: float = float(stats.get("characters", {}).get(key, {}).get("max_hp", 1.0))
		var r: float = float(character_hp.get(key, 0.0)) / maxf(1.0, max_hp)
		hp_ratios.append(r)
		values.append(r)

	# 2. 3 active character one-hot
	for index in range(3):
		values.append(1.0 if player_manager.current_char == characters[index] else 0.0)

	# 3. 15 post-buff effective stats (Knight, Mage, Rogue x 5 stats)
	var buff_atk: float = _get_team_buff("atk")
	var buff_def: float = _get_team_buff("def")
	var buff_cr: float  = _get_team_buff("crit_rate")
	var buff_cd: float  = _get_team_buff("crit_damage")

	for c_key in ["Knight", "Mage", "Rogue"]:
		var c_data: Dictionary = stats.get("characters", {}).get(c_key, {})
		var eff_atk: float = (float(c_data.get("attack", 0.0)) + buff_atk) / 100.0
		var eff_def: float = (float(c_data.get("defense", 0.0)) + buff_def) / 100.0
		var eff_cr: float  = minf(1.0, float(c_data.get("crit_rate", 0.0)) + (buff_cr / 100.0))
		var eff_cd: float  = (float(c_data.get("crit_damage", 1.0)) + (buff_cd / 100.0)) / 3.0
		var dps_pot: float = (eff_atk * 100.0 * (1.0 + eff_cr * (eff_cd * 3.0))) / 150.0
		values.append(eff_atk)
		values.append(eff_def)
		values.append(eff_cr)
		values.append(eff_cd)
		values.append(dps_pot)

	# 4. 5 team tactical & burst indicators
	var min_hp_ratio: float = 1.0
	for r: float in hp_ratios:
		if r > 0.0:
			min_hp_ratio = minf(min_hp_ratio, r)
	values.append(min_hp_ratio)
	values.append(1.0 if min_hp_ratio < 0.5 else 0.0)

	var max_buff_dur: float = 0.0
	for b in active_buffs:
		max_buff_dur = maxf(max_buff_dur, float(b.get("duration", 0)))
	values.append(minf(1.0, max_buff_dur / 30.0))

	var is_dps_active: float = 1.0 if player_manager.current_char == player_manager.char_1 else 0.0
	values.append(is_dps_active)

	var has_dps_buffs: bool = (buff_cr > 0.0 or buff_atk > 0.0)
	var diluc_alive: bool = float(character_hp.get("Knight", 0.0)) > 0.0
	var dps_ready_to_burst: float = 1.0 if (is_dps_active == 0.0 and has_dps_buffs and diluc_alive) else 0.0
	values.append(dps_ready_to_burst)

	# 5. 12 cooldowns across all characters (Knight, Mage, Rogue x 4 moves)
	for c_key in ["Knight", "Mage", "Rogue"]:
		var cds: Dictionary = character_cooldowns.get(c_key, {})
		for move_name in ["basic", "heavy", "skill", "ultimate"]:
			values.append(minf(1.0, float(cds.get(move_name, 0.0)) / 32.0))

	# 6. 10 combat state & boss
	values.append(last_damage           / maxf(1.0, boss_max))
	values.append(last_incoming_damage  / maxf(1.0, _team_max_hp()))
	values.append(1.0 if player_attack_name != "" else 0.0)
	values.append(minf(1.0, float(player_attack_tick) / 16.0))
	var boss_index := ["Up", "Summon", "Taunt"].find(boss_attack_name)
	for index in range(3):
		values.append(1.0 if boss_index == index else 0.0)
	values.append(minf(1.0, float(boss_attack_tick) / 16.0))
	values.append(1.0 if boss_attack_name != "" else 0.0)
	values.append(minf(1.0, float(boss_stunned_ticks) / 4.0))

	# 7. 6 defense & buffs
	values.append(minf(1.0, float(player_frozen_ticks) / 1.0))
	values.append(minf(1.0, float(dodge_charges) / float(DODGE_MAX_CHARGES)))
	values.append(minf(2.0, buff_atk / 50.0))
	values.append(minf(2.0, buff_def / 50.0))
	values.append(minf(1.0, buff_cr / 100.0))
	values.append(minf(1.0, buff_cd / 100.0))

	# 8. 1 Boss immediate threat flag
	var boss_threat: float = 0.0
	if boss_attack_tokens.size() > 0 and boss_stunned_ticks <= 0:
		for check_idx in [boss_attack_tick, boss_attack_tick + 1]:
			if check_idx < boss_attack_tokens.size():
				var tok: String = boss_attack_tokens[check_idx].strip_edges()
				if not tok.is_empty() and tok.to_upper() not in ["X", "_"]:
					boss_threat = 1.0
					break
	values.append(boss_threat)

	# 9. 32 Padded token look-ahead (tokens already resolved = index < player_attack_tick)
	values.append_array(_rl_padded_tokens(player_attack_tokens, player_attack_tick - 1))
	values.append_array(_rl_padded_tokens(boss_attack_tokens,   boss_attack_tick   - 1))
	return values


func _rl_padded_tokens(tokens: Array[String], current: int) -> Array[float]:
	var values: Array[float] = []
	for index in range(16):
		if index >= tokens.size() or index <= current:
			values.append(0.0)
		else:
			var token := tokens[index].strip_edges()
			values.append(0.0 if token.is_empty() or token.to_upper() in ["X", "_"] \
					else minf(1.0, token.to_float() / 10.0))
	return values


# ═══════════════════════════════════════════════════════════════════════════════
# STATS & PATTERN HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

func _load_stats() -> Dictionary:
	var stats_path := SAVED_STATS_PATH if FileAccess.file_exists(SAVED_STATS_PATH) else STATS_PATH
	if not FileAccess.file_exists(stats_path):
		push_error("Combat stats file is missing: " + stats_path)
		return {}
	var file: FileAccess = FileAccess.open(stats_path, FileAccess.READ)
	if file == null:
		push_error("Unable to load combat stats from " + stats_path)
		return {}
	var parsed: Variant = JSON.parse_string(file.get_as_text())
	if not parsed is Dictionary:
		push_error("Combat stats must contain a JSON object.")
		return {}
	return parsed


func _get_pattern_tokens(owner_stats: Dictionary, move_name: String) -> Array[String]:
	var patterns: Dictionary = owner_stats.get("attack_patterns", {}) as Dictionary
	var aliases: Dictionary = {
		"basic":    "Basic (2s)",
		"heavy":    "Heavy (4s)",
		"skill":    "Skill (5s)",
		"ultimate": "Ultimate (8s)",
		"Up":       "Atk 1 UP (9s)",
		"Summon":   "Atk 2 Summon (10s)",
		"Taunt":    "Atk 3 Taunt (9s)"
	}
	var pattern_key: String = str(aliases.get(move_name, move_name))
	var pattern: Variant    = patterns.get(pattern_key, ["X"])
	var tokens: Array[String] = []
	if pattern is Array:
		for token in pattern:
			tokens.append(str(token))
	else:
		var pattern_text := str(pattern)
		if pattern_text.contains(" "):
			for token in pattern_text.split(" ", false):
				tokens.append(str(token))
		else:
			for index in range(pattern_text.length()):
				tokens.append(pattern_text[index])
	return tokens


func _character_key(character: Node) -> String:
	if character == player_manager.char_1:
		return "Knight"
	if character == player_manager.char_2:
		return "Mage"
	return "Rogue"


func _team_max_hp() -> float:
	var total := 0.0
	for character_name in ["Knight", "Mage", "Rogue"]:
		total += float(stats.get("characters", {}).get(character_name, {}).get("max_hp", 0.0))
	return total


# ═══════════════════════════════════════════════════════════════════════════════
# HUD
# ═══════════════════════════════════════════════════════════════════════════════

func _build_hud() -> void:
	# Primary HUD layer
	hud_layer       = CanvasLayer.new()
	hud_layer.name  = "CombatHUD"
	add_child(hud_layer)

	# Separate layer for floating damage numbers (renders on top)
	float_label_layer       = CanvasLayer.new()
	float_label_layer.name  = "FloatLabels"
	float_label_layer.layer = 10
	add_child(float_label_layer)

	var root := Control.new()
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_PASS
	hud_layer.add_child(root)

	# ── Top: boss HP + timer ──────────────────────────────────────────────────
	var top := VBoxContainer.new()
	top.set_anchors_preset(Control.PRESET_TOP_WIDE)
	top.offset_left            = 24
	top.offset_top             = 18
	top.offset_right           = -24
	top.custom_minimum_size.y  = 56
	top.mouse_filter           = Control.MOUSE_FILTER_STOP
	root.add_child(top)

	var boss_row := HBoxContainer.new()
	boss_label                    = Label.new()
	boss_label.custom_minimum_size.x = 180
	boss_row.add_child(boss_label)
	timer_label                              = Label.new()
	timer_label.size_flags_horizontal        = Control.SIZE_EXPAND_FILL
	timer_label.horizontal_alignment         = HORIZONTAL_ALIGNMENT_RIGHT
	boss_row.add_child(timer_label)
	top.add_child(boss_row)

	boss_bar                 = ProgressBar.new()
	boss_bar.show_percentage = false
	top.add_child(boss_bar)

	# ── Bottom-right: player HP + move buttons + dodge + buffs ───────────────
	var bottom := VBoxContainer.new()
	bottom.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	bottom.position = Vector2(-440, -198)
	bottom.size     = Vector2(415, 188)
	bottom.mouse_filter = Control.MOUSE_FILTER_STOP
	root.add_child(bottom)

	player_label = Label.new()
	bottom.add_child(player_label)

	player_bar                 = ProgressBar.new()
	player_bar.show_percentage = false
	bottom.add_child(player_bar)

	# Move buttons row
	var moves := HBoxContainer.new()
	moves.alignment = BoxContainer.ALIGNMENT_END
	bottom.add_child(moves)
	for move_name in ["basic", "heavy", "skill", "ultimate"]:
		var button := Button.new()
		button.text                = move_name.to_upper()
		button.custom_minimum_size = Vector2(90, 38)
		button.focus_mode          = Control.FOCUS_NONE
		button.pressed.connect(_on_move_pressed.bind(move_name))
		moves.add_child(button)
		move_buttons[move_name] = button

	# Dodge row
	var dodge_row := HBoxContainer.new()
	dodge_row.alignment = BoxContainer.ALIGNMENT_END
	dodge_row.add_theme_constant_override("separation", 6)
	bottom.add_child(dodge_row)

	var dodge_title := Label.new()
	dodge_title.text = "DODGE"
	dodge_title.add_theme_font_size_override("font_size", 10)
	dodge_row.add_child(dodge_title)

	dodge_charge_label = Label.new()
	dodge_charge_label.text = "[%d]" % dodge_charges
	dodge_charge_label.add_theme_font_size_override("font_size", 14)
	dodge_charge_label.add_theme_color_override("font_color", Color(0.3, 0.9, 1.0))
	dodge_row.add_child(dodge_charge_label)

	dodge_regen_bar                       = ProgressBar.new()
	dodge_regen_bar.max_value             = DODGE_REGEN_TICKS
	dodge_regen_bar.value                 = 0
	dodge_regen_bar.show_percentage       = false
	dodge_regen_bar.custom_minimum_size   = Vector2(80, 8)
	dodge_regen_bar.size_flags_vertical   = Control.SIZE_SHRINK_CENTER
	dodge_row.add_child(dodge_regen_bar)

	# Active Buffs display
	active_buff_label = Label.new()
	active_buff_label.add_theme_font_size_override("font_size", 10)
	active_buff_label.add_theme_color_override("font_color", Color(0.95, 0.85, 0.3))
	active_buff_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	bottom.add_child(active_buff_label)

	_update_hud()


func _update_hud() -> void:
	if boss_bar == null:
		return
	var boss_max := float(stats.get("boss", {}).get("max_hp", 5000))
	boss_bar.max_value = boss_max
	boss_bar.value     = boss_hp
	boss_label.text    = "THE SKELLY  %d / %d" % [boss_hp, boss_max]

	var total_secs := maxi(0, floori(battle_time_left))
	timer_label.text = "TIME  %02d:%02d" % [floori(float(total_secs) / 60.0), total_secs % 60]

	var active = player_manager.current_char
	if active == null:
		return
	var key    := _character_key(active)
	var max_hp := float(stats.get("characters", {}).get(key, {}).get("max_hp", 1000))
	player_bar.max_value = max_hp
	player_bar.value     = character_hp.get(key, 0.0)
	player_label.text    = "%s  %d / %d" % [
		stats.get("characters", {}).get(key, {}).get("display_name", key),
		character_hp.get(key, 0.0),
		max_hp
	]

	for move_name in move_buttons:
		var cd: float = move_cooldowns[move_name]
		move_buttons[move_name].disabled = cd > 0.0 or battle_finished or player_frozen_ticks > 0
		move_buttons[move_name].text = "%s%s" % [
			move_name.to_upper(),
			(" %.0f" % cd) if cd > 0.0 else ""
		]

	# Dodge UI
	if dodge_charge_label != null:
		dodge_charge_label.text     = "[%d]" % dodge_charges
		dodge_charge_label.modulate = Color(0.4, 0.4, 0.4) if dodge_charges == 0 else Color.WHITE
	if dodge_regen_bar != null:
		dodge_regen_bar.visible = dodge_charges < DODGE_MAX_CHARGES
		dodge_regen_bar.value   = dodge_regen_counter

	# Buffs UI
	if active_buff_label != null:
		if active_buffs.is_empty():
			active_buff_label.text = ""
		else:
			var buff_strs: Array[String] = []
			for b in active_buffs:
				buff_strs.append("%s (%ds)" % [b.get("label", b.get("type", "")), b.get("duration", 0)])
			active_buff_label.text = "BUFFS: " + " | ".join(buff_strs)


func _on_move_pressed(move_name: String) -> void:
	if battle_finished or move_cooldowns[move_name] > 0.0 or player_frozen_ticks > 0:
		return
	var active = player_manager.current_char
	if active != null and active.has_method("queue_or_play"):
		var anim: String = _get_animation_for_move(active, move_name)
		if anim != "":
			active.queue_or_play(anim)


func _finish_battle(victory: bool) -> void:
	if battle_finished:
		return
	battle_finished = true
	_update_hud()
	if victory and boss != null:
		var tree = boss.get_node_or_null("AnimationTree")
		if tree != null:
			var playback = tree.get("parameters/StateMachine/playback")
			if playback != null:
				playback.travel("Dead")
	outcome_panel = ColorRect.new()
	outcome_panel.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	outcome_panel.color        = Color(0.02, 0.02, 0.03, 0.88)
	outcome_panel.mouse_filter = Control.MOUSE_FILTER_STOP
	var result := Label.new()
	result.text = "VICTORY" if victory else "DEFEAT"
	result.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
	result.position  -= Vector2(140, 45)
	result.size       = Vector2(280, 90)
	result.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	result.vertical_alignment   = VERTICAL_ALIGNMENT_CENTER
	result.add_theme_font_size_override("font_size", 42)
	outcome_panel.add_child(result)
	var return_button := Button.new()
	return_button.text = "RETURN TO TITLE"
	return_button.set_anchors_preset(Control.PRESET_CENTER)
	return_button.position   = Vector2(-110, 55)
	return_button.size       = Vector2(220, 44)
	return_button.focus_mode = Control.FOCUS_ALL
	return_button.pressed.connect(_on_return_to_title)
	outcome_panel.add_child(return_button)
	hud_layer.add_child(outcome_panel)


func _on_return_to_title() -> void:
	var server_pid: int = int(get_tree().root.get_meta("rl_policy_server_pid", -1))
	if server_pid > 0:
		OS.kill(server_pid)
		get_tree().root.set_meta("rl_policy_server_pid", -1)
	get_tree().change_scene_to_file(TITLE_SCENE)
