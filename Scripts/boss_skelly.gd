extends Node3D # Change to Node2D or CharacterBody3D if needed

@onready var anim_tree = $AnimationTree
var playback 
var stopping := false

# Declare the array of animation names at the top so the whole script can see it
var move_set = ["Summon", "Taunt", "Up"]

func _ready():
	# 1. Activate the tree first
	anim_tree.active = true
	
	# 2. Grab the playback object after activation
	# Replace "parameters/playback" with the path from your tooltip
	playback = anim_tree.get("parameters/StateMachine/playback")
	
	# 3. Safety check
	if playback == null:
		push_error("Playback is null! Check the parameter path.")
		return
		
	start_boss_loop()

func _exit_tree():
	stopping = true

func start_boss_loop():
	# Wait until the initial Awaken sequence finishes
	await wait_for_state("Idle")
	if stopping or not is_inside_tree() or playback == null:
		return
	
	while not stopping and is_inside_tree():
		# Wait for 5 seconds while looping Idle
		var scene_tree := get_tree()
		if scene_tree == null:
			return
		await scene_tree.create_timer(5.0).timeout
		if stopping or not is_inside_tree() or playback == null:
			return
		
		var chosen_move = move_set.pick_random()
		playback.travel(chosen_move)
		
		# NEW: Wait until the state machine ACTUALLY leaves the Idle state
		while playback.get_current_node() == "Idle":
			if stopping or not is_inside_tree():
				return
			var frame_tree := get_tree()
			if frame_tree == null:
				return
			await frame_tree.process_frame 
			
		# Wait for the attack sequence to complete and return to Idle
		await wait_for_state("Idle")

# Helper function that yields until a specific state is reached
func wait_for_state(target_state: String):
	while playback != null and playback.get_current_node() != target_state:
		if stopping or not is_inside_tree():
			return
		var scene_tree := get_tree()
		if scene_tree == null:
			return
		await scene_tree.process_frame
