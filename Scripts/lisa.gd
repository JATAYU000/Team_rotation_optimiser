extends Node3D

signal attack_started(attack_name: String)

@onready var anim_tree: AnimationTree = $AnimationTree
var playback: AnimationNodeStateMachinePlayback

# Cooldown durations
var skill_cooldown : float = 8.0
var ult_cooldown : float = 32.0

# Current cooldown timers
var skill_timer : float = 0.0
var ult_timer : float = 0.0

# Stores the most recent input while an attack is playing
var queued_attack : String = ""

# Parry freeze — set by combat_manager when a parry occurs.
var is_frozen : bool = false

# Lisa-specific: Heavy attack (Aim -> Charge) loops Charge, switch back to Idle after 4.0s total.
var heavy_attack_timer : float = 0.0
var is_heavy_attacking : bool = false


func set_frozen(frozen: bool) -> void:
	is_frozen = frozen
	if frozen:
		queued_attack = ""
		is_heavy_attacking = false
		heavy_attack_timer = 0.0


func force_idle() -> void:
	queued_attack = ""
	is_heavy_attacking = false
	heavy_attack_timer = 0.0
	if playback != null:
		playback.start("Idle")


func play_parry() -> void:
	queued_attack = ""
	is_heavy_attacking = false
	heavy_attack_timer = 0.0
	if playback != null:
		playback.start("Parry")


func _ready() -> void:
	anim_tree.active = true
	playback = anim_tree.get("parameters/StateMachine/playback")
	if playback == null:
		push_error("Playback is null in Lisa! Check the parameter path.")


func _process(delta: float) -> void:
	if playback == null:
		return
	if is_frozen:
		return

	# 1. Process cooldown timers
	if skill_timer > 0.0:
		skill_timer -= delta
	if ult_timer > 0.0:
		ult_timer -= delta

	# 2. Lisa Heavy attack timing (Aim -> Charge loop):
	# Automatically switches Lisa back to Idle after 4.0s of heavy attack timing.
	var current_state: String = playback.get_current_node()
	if is_heavy_attacking:
		heavy_attack_timer += delta
		if heavy_attack_timer >= 4.0:
			is_heavy_attacking = false
			heavy_attack_timer = 0.0
			playback.travel("Idle")
			# If the StateMachine transition gets stuck in the Charge loop, start Idle directly
			if playback.get_current_node() != "Idle":
				playback.start("Idle")
		elif current_state not in ["Aim", "Charge"]:
			# Left heavy attack animation early (e.g. hit or parry)
			is_heavy_attacking = false
			heavy_attack_timer = 0.0
	elif current_state in ["Aim", "Charge"]:
		# Heavy started via another path
		is_heavy_attacking = true
		heavy_attack_timer = 0.0

	# 3. If the character is Idle and we have an attack queued, play it immediately
	if current_state == "Idle" and queued_attack != "":
		play_attack(queued_attack)
		queued_attack = ""


func _unhandled_input(event: InputEvent) -> void:
	if is_frozen or playback == null:
		return

	if event.is_action_pressed("basic_attack"):
		queue_or_play("Shoot")
	elif event.is_action_pressed("heavy_attack"):
		queue_or_play("Aim")
	elif event.is_action_pressed("skill") and skill_timer <= 0:
		queue_or_play("Yo")
	elif event.is_action_pressed("ultimate") and ult_timer <= 0:
		queue_or_play("Long")
	elif event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_Q:
		if ult_timer <= 0:
			queue_or_play("Long")


func queue_or_play(attack_name: String) -> void:
	if playback.get_current_node() == "Idle":
		play_attack(attack_name)
	else:
		queued_attack = attack_name


func play_attack(attack_name: String) -> void:
	if attack_name == "Aim":
		is_heavy_attacking = true
		heavy_attack_timer = 0.0
	else:
		is_heavy_attacking = false
		heavy_attack_timer = 0.0
	playback.travel(attack_name)
	attack_started.emit(attack_name)

	if attack_name == "Yo":
		skill_timer = skill_cooldown
	elif attack_name == "Long":
		ult_timer = ult_cooldown
