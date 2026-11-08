extends Node3D # Change to CharacterBody3D or Node2D if applicable

signal attack_started(attack_name: String)

@onready var anim_tree = $AnimationTree
var playback

# Cooldown durations
var skill_cooldown : float = 8.0
var ult_cooldown : float = 32.0

# Current cooldown timers
var skill_timer : float = 0.0
var ult_timer : float = 0.0

# Stores the most recent input while an attack is playing
var queued_attack : String = ""

# Parry freeze — set by combat_manager when a parry stuns the player briefly.
var is_frozen : bool = false


func set_frozen(frozen: bool) -> void:
	is_frozen = frozen
	# If frozen, also clear any queued input so it does not fire immediately after.
	if frozen:
		queued_attack = ""


func force_idle() -> void:
	queued_attack = ""
	if playback != null:
		playback.start("Idle")


func play_parry() -> void:
	queued_attack = ""
	if playback != null:
		playback.start("Parry")

func _ready():
	anim_tree.active = true
	# IMPORTANT: Update this path to match your exact playback property like last time!
	playback = anim_tree.get("parameters/StateMachine/playback") 
	
	if playback == null:
		push_error("Playback is null! Check the parameter path.")

func _process(delta):
	if playback == null:
		return
	# When frozen by a parry, block all input and queued actions.
	if is_frozen:
		return

	# 1. Process cooldown timers
	if skill_timer > 0:
		skill_timer -= delta
	if ult_timer > 0:
		ult_timer -= delta

	# 2. If the character is Idle and we have an attack queued, play it immediately
	if playback.get_current_node() == "Idle" and queued_attack != "":
		play_attack(queued_attack)
		queued_attack = ""


func _unhandled_input(event: InputEvent):
	if is_frozen or playback == null:
		return

	if event.is_action_pressed("basic_attack"):
		queue_or_play("Diag")
	elif event.is_action_pressed("heavy_attack"):
		queue_or_play("Load")
	elif event.is_action_pressed("skill") and skill_timer <= 0:
		queue_or_play("Raise")
	elif event.is_action_pressed("ultimate") and ult_timer <= 0:
		queue_or_play("Chop 2")
	elif event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_Q:
		if ult_timer <= 0:
			queue_or_play("Chop 2")

func queue_or_play(attack_name: String):
	# If we are already Idle, play the attack right away
	if playback.get_current_node() == "Idle":
		play_attack(attack_name)
	else:
		# Otherwise, overwrite the queue with this latest input
		queued_attack = attack_name

func play_attack(attack_name: String):
	playback.travel(attack_name)
	attack_started.emit(attack_name)
	
	# Start the respective cooldowns ONLY when the attack successfully plays
	if attack_name == "Raise":
		skill_timer = skill_cooldown
	elif attack_name == "Chop 2":
		ult_timer = ult_cooldown
