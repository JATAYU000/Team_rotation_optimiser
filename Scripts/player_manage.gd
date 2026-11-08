extends Node3D

@onready var char_1 = $Knight
@onready var char_2 = $Mage
@onready var char_3 = $Rogue

var current_char : Node3D
var is_switching : bool = false # Prevents input spam during the animation
var dead_characters: Dictionary = {}

func _ready():
	current_char = char_1
	enable_character(char_1)
	disable_character(char_2)
	disable_character(char_3)
	
	# Play the spawn animation for the first character on load
	play_animation(char_1, "Drop")

func _process(_delta):
	# Ignore inputs if we are currently in the middle of a swap animation
	if is_switching:
		return 

	if Input.is_action_just_pressed("switch_1") and current_char != char_1:
		swap_to(char_1)
	elif Input.is_action_just_pressed("switch_2") and current_char != char_2:
		swap_to(char_2)
	elif Input.is_action_just_pressed("switch_3") and current_char != char_3:
		swap_to(char_3)

func _unhandled_input(event: InputEvent) -> void:
	if is_switching:
		return
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_1 and current_char != char_1:
			swap_to(char_1)
		elif event.keycode == KEY_2 and current_char != char_2:
			swap_to(char_2)
		elif event.keycode == KEY_3 and current_char != char_3:
			swap_to(char_3)

func swap_to(new_character: Node3D):
	if not is_character_alive(new_character):
		return
	is_switching = true # Lock inputs
	
	# 1. Play the exit animation on the current character
	play_animation(current_char, "Switch")
	
	# 2. Wait for the exit animation to reach the "End" state
	var old_playback = current_char.get_node("AnimationTree").get("parameters/StateMachine/playback")
	while old_playback.get_current_node() != "End":
		if not is_inside_tree():
			return
		await get_tree().process_frame
		
	# 3. Save position and swap who is active
	var saved_position = current_char.global_position
	disable_character(current_char)
	
	new_character.global_position = saved_position
	enable_character(new_character)
	
	# 4. Play the entry animation on the new character
	# Using start() first resets them from their previous "End" state, then we travel to "Drop"
	var new_playback = new_character.get_node("AnimationTree").get("parameters/StateMachine/playback")
	new_playback.start("Start") 
	new_playback.travel("Drop")
	
	# 5. Wait for the entry animation to finish and reach "Idle" before unlocking inputs
	while new_playback.get_current_node() != "Idle":
		if not is_inside_tree():
			return
		await get_tree().process_frame
		
	current_char = new_character
	is_switching = false # Unlock inputs

# Helper function to easily trigger animations from the manager
func play_animation(character: Node3D, anim_name: String):
	var playback = character.get_node("AnimationTree").get("parameters/StateMachine/playback")
	if playback:
		playback.travel(anim_name)

func disable_character(character: Node3D):
	character.hide()
	character.process_mode = Node.PROCESS_MODE_DISABLED 

func enable_character(character: Node3D):
	character.show()
	character.process_mode = Node.PROCESS_MODE_INHERIT

func is_character_alive(character: Node3D) -> bool:
	return not dead_characters.get(character, false)

func kill_character(character: Node3D) -> void:
	if dead_characters.get(character, false):
		return
	dead_characters[character] = true
	var tree = character.get_node_or_null("AnimationTree")
	if tree != null:
		var playback = tree.get("parameters/StateMachine/playback")
		if playback != null:
			playback.travel("Death")
	var animation_player = character.get_node_or_null("AnimationPlayer")
	if animation_player != null and (tree == null or tree.get("parameters/StateMachine/playback") == null):
		animation_player.play("Player/Death_B")

func switch_to_next_alive() -> bool:
	for candidate in [char_1, char_2, char_3]:
		if is_character_alive(candidate):
			if current_char != candidate:
				_activate_after_death(candidate)
			return true
	return false

func _activate_after_death(new_character: Node3D) -> void:
	is_switching = true
	disable_character(current_char)
	enable_character(new_character)
	var playback = new_character.get_node("AnimationTree").get("parameters/StateMachine/playback")
	if playback == null:
		current_char = new_character
		is_switching = false
		return
	playback.start("Start")
	playback.travel("Drop")
	while playback.get_current_node() != "Idle":
		if not is_inside_tree():
			return
		await get_tree().process_frame
	current_char = new_character
	is_switching = false
