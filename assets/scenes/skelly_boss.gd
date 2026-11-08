extends Node3D # Change to Node2D or CharacterBody3D if needed

@onready var anim_tree = $AnimationTree
var playback 

# Declare the array of animation names at the top so the whole script can see it
var move_set = ["Summon", "Taunt", "Up"]

func _ready():
	# 1. Activate the tree first
	anim_tree.active = true
	
	# 2. Grab the playback object after activation
	playback = anim_tree.get("parameters/playback")
	
	# 3. Safety check
	if playback == null:
		push_error("Playback is null! Check the parameter path.")
		return
		
	start_boss_loop()

func start_boss_loop():
	# Wait until the initial "Start -> Awaken -> Idle" sequence finishes
	await wait_for_state("Idle")
	
	while true:
		# 1. Wait for 3 seconds while looping Idle
		await get_tree().create_timer(3.0).timeout
		
		# 2. Randomly pick Summon, Taunt, or Up
		var chosen_move = move_set.pick_random()
		
		# 3. Command the state machine to play the chosen move
		playback.travel(chosen_move)
		
		# Wait a single frame to ensure the state machine transitions OUT of Idle
		await get_tree().process_frame 
		
		# 4. Wait for the sequence to complete and return to Idle
		await wait_for_state("Idle")

# Helper function that yields until a specific state is reached
func wait_for_state(target_state: String):
	while playback.get_current_node() != target_state:
		await get_tree().process_frame
