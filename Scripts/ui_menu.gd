extends Control

@onready var old_scroll_container = $ScrollContainer
@onready var old_background = $ColorRect

const BG = Color("#0f1013")
const SURFACE = Color("#171619")
const SURFACE_LIGHT = Color("#211f1e")
const INPUT_SURFACE = Color("#141417")
const BORDER = Color("#5b4a3d")
const BORDER_DARK = Color("#342b28")
const GOLD = Color("#d0b27a")
const MUTED_GOLD = Color("#927b58")
const TEXT = Color("#e1d7c7")
const MUTED = Color("#817b74")
const BLOOD = Color("#a34646")
const EMBER = Color("#6f3535")
const DEFAULT_STATS_PATH = "res://data/combat_stats.json"
const SAVED_STATS_PATH = "user://combat_stats.json"
const DILUC_PORTRAIT = "res://assets/charectors/Knight_knight_texture.png"
const LYNAE_PORTRAIT = "res://assets/charectors/Rogue_rogue_texture.png"
const LISA_PORTRAIT = "res://assets/charectors/Mage_mage_texture.png"
const BOSS_PORTRAIT = "res://assets/charectors/Skeleton_Warrior_skeleton_texture.png"

@export var ui_font: Font

const STAT_ICONS = {
	"HP": "♥",
	"ATK": "⚔",
	"DEF": "⬟",
	"Crit Rate (%)": "✦",
	"Crit Dmg (%)": "✧"
}
const STAT_ICON_ORDER = ["HP", "ATK", "DEF", "Crit Rate (%)", "Crit Dmg (%)"]

var diluc_stats = {
	"HP": 1000, "ATK": 40, "DEF": 10, "Crit Rate (%)": 65, "Crit Dmg (%)": 170
}
var diluc_attacks = {
	"Basic (2s)": "__",
	"Heavy (4s)": "X__X",
	"Skill (5s)": "XX__X",
	"Ultimate (8s)": "X__X___X"
}

var lynae_stats = {
	"HP": 1500, "ATK": 20, "DEF": 30, "Crit Rate (%)": 50, "Crit Dmg (%)": 150
}
var lynae_attacks = {
	"Basic (2s)": "X_",
	"Heavy (4s)": "XX__",
	"Skill (5s)": "XX__X",
	"Ultimate (8s)": "X_X____X"
}

var lisa_stats = {
	"HP": 3000, "ATK": 10, "DEF": 20, "Crit Rate (%)": 50, "Crit Dmg (%)": 150
}
var lisa_attacks = {
	"Basic (2s)": "_X",
	"Heavy (4s)": "X___",
	"Skill (5s)": "XXX_X",
	"Ultimate (8s)": "XX__XX_X"
}

var boss_stats = {
	"HP": 5000, "ATK": 20, "DEF": 10
}
var boss_attacks = {
	"Atk 1 UP (9s)": "XXX_X__X_",
	"Atk 2 Summon (10s)": "XXXXX_X__X",
	"Atk 3 Taunt (9s)": "XXXX_X_X_"
}
var config_data: Dictionary = {}
var sanitizing_attack_input := false
var attack_buffs: Dictionary = {}
var training_status_label: Label
var training_evaluation_label: Label
var model_dropdown: OptionButton
var device_dropdown: OptionButton
var editable_controls: Array[Control] = []
var device_selection := "auto"
var training_in_progress := false
var training_progress_path := "user://rl_training_progress.jsonl"
var training_progress_offset := 0

func _ready():
	_load_stats()
	old_scroll_container.hide()
	old_background.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	old_background.color = BG
	old_background.mouse_filter = Control.MOUSE_FILTER_IGNORE
	set_process(true)

	var frame = MarginContainer.new()
	frame.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	frame.add_theme_constant_override("margin_left", 26)
	frame.add_theme_constant_override("margin_top", 18)
	frame.add_theme_constant_override("margin_right", 26)
	frame.add_theme_constant_override("margin_bottom", 18)
	add_child(frame)

	var outer_column := VBoxContainer.new()
	outer_column.add_theme_constant_override("separation", 10)
	frame.add_child(outer_column)

	var layout: HBoxContainer = HBoxContainer.new()
	layout.add_theme_constant_override("separation", 12)
	layout.size_flags_vertical = Control.SIZE_EXPAND_FILL
	outer_column.add_child(layout)

	var dungeon_band = VBoxContainer.new()
	dungeon_band.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	dungeon_band.size_flags_stretch_ratio = 4.0
	dungeon_band.add_theme_constant_override("separation", 6)
	layout.add_child(dungeon_band)

	var character_card = build_character_sheet(dungeon_band, "DILUC", DILUC_PORTRAIT, diluc_stats, diluc_attacks, false)
	character_card.size_flags_vertical = Control.SIZE_EXPAND_FILL
	character_card.size_flags_stretch_ratio = 4.0
	character_card = build_character_sheet(dungeon_band, "LYNAE", LYNAE_PORTRAIT, lynae_stats, lynae_attacks, false)
	character_card.size_flags_vertical = Control.SIZE_EXPAND_FILL
	character_card.size_flags_stretch_ratio = 4.0
	character_card = build_character_sheet(dungeon_band, "LISA", LISA_PORTRAIT, lisa_stats, lisa_attacks, false)
	character_card.size_flags_vertical = Control.SIZE_EXPAND_FILL
	character_card.size_flags_stretch_ratio = 4.0
	var boss_card = build_boss_sheet(dungeon_band)
	boss_card.size_flags_vertical = Control.SIZE_EXPAND_FILL
	boss_card.size_flags_stretch_ratio = 3.0

	var utility_band = build_utility_band()
	utility_band.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	utility_band.size_flags_stretch_ratio = 1.0
	layout.add_child(utility_band)

func build_utility_band() -> Control:
	var band = PanelContainer.new()
	band.add_theme_stylebox_override("panel", panel_style(SURFACE, BORDER, 1, 2))

	var column = VBoxContainer.new()
	column.add_theme_constant_override("separation", 8)
	band.add_child(column)

	for button_data in [["SAVE", false], ["TRAIN", false], ["SIMULATE", false]]:
		var button = Button.new()
		button.text = button_data[0]
		button.custom_minimum_size = Vector2(0, 34)
		button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		button.add_theme_font_size_override("font_size", 11)
		button.add_theme_stylebox_override("normal", panel_style(SURFACE_LIGHT, BORDER_DARK, 1, 1))
		button.add_theme_stylebox_override("hover", panel_style(Color("#302a26"), GOLD, 1, 1))
		button.add_theme_stylebox_override("pressed", panel_style(Color("#131215"), GOLD, 2, 1))
		button.add_theme_color_override("font_color", GOLD if button_data[1] else TEXT)
		button.add_theme_color_override("font_hover_color", GOLD)
		if button_data[1]:
			button.pressed.connect(_on_start_pressed)
		elif button_data[0] == "SAVE":
			button.pressed.connect(_on_save_pressed)
		elif button_data[0] == "TRAIN":
			button.pressed.connect(_on_train_pressed)
		elif button_data[0] == "SIMULATE":
			button.pressed.connect(_on_simulate_pressed)
		elif button_data[0] == "QUIT":
			button.pressed.connect(_on_quit_pressed)
		column.add_child(button)

	var reserved_space := PanelContainer.new()
	reserved_space.size_flags_vertical = Control.SIZE_EXPAND_FILL
	reserved_space.custom_minimum_size.y = 80
	reserved_space.add_theme_stylebox_override("panel", panel_style(INPUT_SURFACE, BORDER_DARK, 1, 1))
	column.add_child(reserved_space)

	var evaluation_column := VBoxContainer.new()
	evaluation_column.add_theme_constant_override("separation", 5)
	reserved_space.add_child(evaluation_column)
	evaluation_column.add_child(make_section_label("RL MODEL"))
	model_dropdown = OptionButton.new()
	model_dropdown.custom_minimum_size = Vector2(0, 26)
	model_dropdown.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	model_dropdown.add_theme_font_size_override("font_size", 9)
	model_dropdown.add_theme_stylebox_override("normal", panel_style(INPUT_SURFACE, BORDER_DARK, 1, 1))
	model_dropdown.add_theme_stylebox_override("hover", panel_style(SURFACE_LIGHT, GOLD, 1, 1))
	model_dropdown.add_item("DILUC PPO")
	model_dropdown.add_item("DILUC DQN")
	model_dropdown.add_item("DILUC Q-LEARNING")
	model_dropdown.select(0)
	model_dropdown.tooltip_text = "Select RL architecture: PPO (Policy Gradient), DQN (Value/Replay), or Q-Learning (A2C)"
	_register_editable_control(model_dropdown)
	evaluation_column.add_child(model_dropdown)
	evaluation_column.add_child(make_section_label("RUN DEVICE"))
	device_dropdown = OptionButton.new()
	device_dropdown.custom_minimum_size = Vector2(0, 26)
	device_dropdown.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	device_dropdown.add_theme_font_size_override("font_size", 9)
	device_dropdown.add_theme_stylebox_override("normal", panel_style(INPUT_SURFACE, BORDER_DARK, 1, 1))
	device_dropdown.add_theme_stylebox_override("hover", panel_style(SURFACE_LIGHT, GOLD, 1, 1))
	device_dropdown.add_item("AUTO / GPU")
	device_dropdown.add_item("GPU / CUDA")
	device_dropdown.add_item("CPU")
	device_dropdown.select(0)
	device_dropdown.tooltip_text = "Auto selects CUDA when this Python environment supports it"
	device_dropdown.item_selected.connect(_on_device_selected)
	_register_editable_control(device_dropdown)
	evaluation_column.add_child(device_dropdown)
	evaluation_column.add_child(make_rule())
	evaluation_column.add_child(make_label("TRAINING EVALUATION", 10, MUTED_GOLD))
	training_status_label = make_label("READY", 10, MUTED)
	training_status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	evaluation_column.add_child(training_status_label)
	training_evaluation_label = make_label("UPDATE  —  DEVICE  —\nWIN RATE  —\nAVG DAMAGE  —\nAVG TIME  —\nREWARD  —\nLOSS  —\nPOLICY  —\nVALUE  —\nENTROPY  —", 10, TEXT)
	training_evaluation_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	evaluation_column.add_child(training_evaluation_label)

	var bottom_spacer := Control.new()
	bottom_spacer.custom_minimum_size.y = 8
	column.add_child(bottom_spacer)

	for button_data in [["START", true], ["QUIT", false]]:
		var button = Button.new()
		button.text = button_data[0]
		button.custom_minimum_size = Vector2(0, 34)
		button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		button.add_theme_font_size_override("font_size", 11)
		button.add_theme_stylebox_override("normal", panel_style(SURFACE_LIGHT, BORDER_DARK, 1, 1))
		button.add_theme_stylebox_override("hover", panel_style(Color("#302a26"), GOLD, 1, 1))
		button.add_theme_stylebox_override("pressed", panel_style(Color("#131215"), GOLD, 2, 1))
		button.add_theme_color_override("font_color", GOLD if button_data[1] else TEXT)
		button.add_theme_color_override("font_hover_color", GOLD)
		if button_data[0] == "START":
			button.pressed.connect(_on_start_pressed)
		else:
			button.pressed.connect(_on_quit_pressed)
		column.add_child(button)
	return band

func make_section_label(text_value: String) -> Label:
	return make_label(text_value, 9, MUTED_GOLD)

func _register_editable_control(control: Control) -> Control:
	editable_controls.append(control)
	return control

func _set_training_locked(locked: bool) -> void:
	for control in editable_controls:
		if control is LineEdit:
			control.editable = not locked
		if control is OptionButton:
			control.disabled = locked

func _on_device_selected(index: int) -> void:
	device_selection = ["auto", "cuda", "cpu"][index]

func _process(_delta: float) -> void:
	if not training_in_progress:
		return
	_read_training_progress()

func _read_training_progress() -> void:
	if not FileAccess.file_exists(training_progress_path):
		return
	var file := FileAccess.open(training_progress_path, FileAccess.READ)
	if file == null:
		return
	file.seek(training_progress_offset)
	while not file.eof_reached():
		var line := file.get_line()
		training_progress_offset = file.get_position()
		if line.is_empty():
			continue
		var parsed: Variant = JSON.parse_string(line)
		if parsed is Dictionary:
			var message := str(parsed.get("message", "training..."))
			training_status_label.text = message
			training_status_label.visible = true
			if parsed.has("error"):
				training_in_progress = false
				_set_training_locked(false)
				training_status_label.add_theme_color_override("font_color", BLOOD)
			var win_rate: Variant = parsed.get("win_rate", null)
			var average_damage: Variant = parsed.get("average_damage", null)
			var average_time: Variant = parsed.get("average_time", null)
			var reward: Variant = parsed.get("reward", null)
			if win_rate != null:
				training_evaluation_label.text = "UPDATE  %d\nDEVICE  %s\nWIN RATE  %.1f%%\nAVG DAMAGE  %.1f\nAVG TIME  %.1fs\nREWARD  %.3f\nPARRIES  %.1f  DODGES  %.1f\nSWITCHES %.1f  BUFFS   %.1f\nLOSS  %.4f\nENTROPY  %.4f" % [
					int(parsed.get("update", 0)),
					str(parsed.get("device", "—")).to_upper(),
					float(win_rate) * 100.0,
					float(average_damage),
					float(average_time),
					float(reward),
					float(parsed.get("average_parries", 0.0)),
					float(parsed.get("average_dodges", 0.0)),
					float(parsed.get("average_switches", 0.0)),
					float(parsed.get("average_buffs", 0.0)),
					float(parsed.get("loss", 0.0)),
					float(parsed.get("entropy", 0.0))
				]
			if int(parsed.get("percent", 0)) >= 100:
				training_in_progress = false
				training_status_label.add_theme_color_override("font_color", GOLD)
				_set_training_locked(false)
	file.close()

func build_character_sheet(parent: Control, char_name: String, portrait_path: String, stats: Dictionary, attacks: Dictionary, is_boss: bool):
	var card = PanelContainer.new()
	card.add_theme_stylebox_override("panel", panel_style(Color("#181719") if not is_boss else Color("#211719"), BLOOD if is_boss else BORDER, 1, 2))
	parent.add_child(card)

	var inset = MarginContainer.new()
	inset.add_theme_constant_override("margin_left", 10)
	inset.add_theme_constant_override("margin_top", 5)
	inset.add_theme_constant_override("margin_right", 10)
	inset.add_theme_constant_override("margin_bottom", 5)
	card.add_child(inset)

	var content = HBoxContainer.new()
	content.add_theme_constant_override("separation", 12)
	inset.add_child(content)

	var identity = VBoxContainer.new()
	identity.custom_minimum_size = Vector2(124, 0)
	identity.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	content.add_child(identity)

	var portrait_frame := PanelContainer.new()
	portrait_frame.custom_minimum_size = Vector2(116, 112)
	portrait_frame.size_flags_horizontal = Control.SIZE_SHRINK_CENTER
	portrait_frame.add_theme_stylebox_override("panel", panel_style(INPUT_SURFACE, BLOOD if is_boss else BORDER, 1, 1))
	var portrait := TextureRect.new()
	portrait.texture = load(portrait_path) as Texture2D
	portrait.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	portrait.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	portrait.mouse_filter = Control.MOUSE_FILTER_IGNORE
	portrait_frame.add_child(portrait)
	identity.add_child(portrait_frame)
	var name_label = make_label(char_name, 18, BLOOD if is_boss else GOLD)
	name_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	identity.add_child(name_label)

	var stat_grid = GridContainer.new()
	stat_grid.columns = 2
	stat_grid.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	stat_grid.add_theme_constant_override("h_separation", 6)
	stat_grid.add_theme_constant_override("v_separation", 3)
	content.add_child(stat_grid)
	for stat in stats:
		var label = make_stat_label(stat)
		stat_grid.add_child(label)
		var input = LineEdit.new()
		input.text = str(stats[stat])
		input.custom_minimum_size = Vector2(52, 21)
		input.alignment = HORIZONTAL_ALIGNMENT_CENTER
		input.add_theme_font_size_override("font_size", 11)
		input.add_theme_stylebox_override("normal", panel_style(INPUT_SURFACE, BORDER_DARK, 1, 1))
		input.add_theme_stylebox_override("focus", panel_style(INPUT_SURFACE, GOLD, 1, 1))
		input.add_theme_color_override("font_color", TEXT)
		stat_grid.add_child(input)
		_register_editable_control(input)
		input.text_changed.connect(_on_stat_changed.bind(stats, stat))

	var attack_column = VBoxContainer.new()
	attack_column.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	attack_column.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	attack_column.add_child(make_label("DAMAGE RHYTHM", 9, MUTED_GOLD))
	content.add_child(attack_column)
	for attack_name in attacks:
		var attack_row = HBoxContainer.new()
		attack_row.add_theme_constant_override("separation", 1)
		attack_row.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		var display_name = attack_name.split(" (")[0]
		var attack_label = make_label(display_name, 10, TEXT)
		attack_label.custom_minimum_size = Vector2(62, 21)
		attack_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
		attack_label.tooltip_text = attack_name
		attack_row.add_child(attack_label)
		var tokens: Array[String] = _pattern_tokens(attacks[attack_name])
		for index in range(tokens.size()):
			if tokens[index].to_upper() == "X":
				attack_row.add_child(make_token_box("X", false))
			else:
				var token_input := make_token_box(tokens[index], true) as LineEdit
				token_input.text_changed.connect(_on_attack_token_changed.bind(attacks, attack_name, index, token_input))
				attack_row.add_child(token_input)
		if display_name == "Skill" or display_name == "Ultimate":
			attack_row.add_child(make_buff_editor(char_name + ":" + attack_name))
		attack_column.add_child(attack_row)
	return card

func build_boss_sheet(parent: Control):
	var card = PanelContainer.new()
	card.add_theme_stylebox_override("panel", panel_style(Color("#211719"), BLOOD, 1, 2))
	parent.add_child(card)

	var inset = MarginContainer.new()
	inset.add_theme_constant_override("margin_left", 10)
	inset.add_theme_constant_override("margin_top", 4)
	inset.add_theme_constant_override("margin_right", 10)
	inset.add_theme_constant_override("margin_bottom", 4)
	card.add_child(inset)

	var row = HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	inset.add_child(row)

	var identity = VBoxContainer.new()
	identity.custom_minimum_size = Vector2(96, 0)
	identity.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	row.add_child(identity)
	var portrait_frame := PanelContainer.new()
	portrait_frame.custom_minimum_size = Vector2(88, 68)
	portrait_frame.size_flags_horizontal = Control.SIZE_SHRINK_CENTER
	portrait_frame.add_theme_stylebox_override("panel", panel_style(INPUT_SURFACE, BLOOD, 1, 1))
	var portrait := TextureRect.new()
	portrait.texture = load(BOSS_PORTRAIT) as Texture2D
	portrait.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	portrait.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	portrait.mouse_filter = Control.MOUSE_FILTER_IGNORE
	portrait_frame.add_child(portrait)
	identity.add_child(portrait_frame)
	var boss_name := make_label("THE SKELLY", 13, BLOOD)
	boss_name.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	identity.add_child(boss_name)

	var stat_grid = GridContainer.new()
	stat_grid.columns = 2
	stat_grid.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	stat_grid.add_theme_constant_override("h_separation", 8)
	stat_grid.add_theme_constant_override("v_separation", 3)
	row.add_child(stat_grid)
	for stat in boss_stats:
		stat_grid.add_child(make_stat_label(stat))
		var input = LineEdit.new()
		input.text = str(boss_stats[stat])
		input.custom_minimum_size = Vector2(56, 21)
		input.alignment = HORIZONTAL_ALIGNMENT_CENTER
		input.add_theme_font_size_override("font_size", 11)
		input.add_theme_stylebox_override("normal", panel_style(INPUT_SURFACE, BORDER_DARK, 1, 1))
		input.add_theme_stylebox_override("focus", panel_style(INPUT_SURFACE, GOLD, 1, 1))
		stat_grid.add_child(input)
		_register_editable_control(input)
		input.text_changed.connect(_on_stat_changed.bind(boss_stats, stat))

	var attacks = VBoxContainer.new()
	attacks.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	attacks.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	attacks.add_child(make_label("DAMAGE RHYTHM", 9, MUTED_GOLD))
	for attack_name in boss_attacks:
		var attack_row = HBoxContainer.new()
		attack_row.add_theme_constant_override("separation", 1)
		var attack_label = make_label(attack_name.split(" (")[0], 10, TEXT)
		attack_label.custom_minimum_size = Vector2(100, 21)
		attack_label.tooltip_text = attack_name
		attack_row.add_child(attack_label)
		var tokens: Array[String] = _pattern_tokens(boss_attacks[attack_name])
		for index in range(tokens.size()):
			if tokens[index].to_upper() == "X":
				attack_row.add_child(make_token_box("X", false))
			else:
				var token_input := make_token_box(tokens[index], true) as LineEdit
				token_input.text_changed.connect(_on_attack_token_changed.bind(boss_attacks, attack_name, index, token_input))
				attack_row.add_child(token_input)
		attacks.add_child(attack_row)
	row.add_child(attacks)
	return card

func make_token_box(value: String, editable: bool) -> Control:
	if editable:
		var input = LineEdit.new()
		input.text = value
		input.custom_minimum_size = Vector2(42, 21)
		input.alignment = HORIZONTAL_ALIGNMENT_CENTER
		input.add_theme_font_size_override("font_size", 11)
		input.add_theme_stylebox_override("normal", panel_style(INPUT_SURFACE, BORDER_DARK, 1, 1))
		input.add_theme_stylebox_override("focus", panel_style(INPUT_SURFACE, GOLD, 1, 1))
		_register_editable_control(input)
		return input

	var box = PanelContainer.new()
	box.custom_minimum_size = Vector2(24, 21)
	box.add_theme_stylebox_override("panel", panel_style(Color("#171619"), BORDER, 1, 1))
	var locked = make_label(value, 11, MUTED)
	locked.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	locked.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	box.add_child(locked)
	return box

func make_buff_editor(buff_key: String) -> Control:
	var spacer := Control.new()
	spacer.custom_minimum_size.x = 8
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	var editor := HBoxContainer.new()
	editor.custom_minimum_size.x = 120
	editor.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	editor.add_theme_constant_override("separation", 3)
	editor.add_child(spacer)

	var tag := make_label("BUFF:", 9, MUTED_GOLD)
	tag.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	editor.add_child(tag)

	var buff_input := LineEdit.new()
	var existing: Variant = attack_buffs.get(buff_key, "")
	if existing is String:
		buff_input.text = existing
	elif existing is Dictionary:
		var amt := str(existing.get("amount", "")).strip_edges()
		if not amt.is_empty():
			var stat_idx := int(existing.get("stat", 0))
			var abrs := ["H", "A", "D", "CR", "CD"]
			var abr: String = abrs[clampi(stat_idx, 0, abrs.size() - 1)]
			buff_input.text = abr + amt
	buff_input.custom_minimum_size = Vector2(110, 21)
	buff_input.alignment = HORIZONTAL_ALIGNMENT_LEFT
	buff_input.add_theme_font_size_override("font_size", 9)
	buff_input.add_theme_stylebox_override("normal", panel_style(INPUT_SURFACE, BORDER_DARK, 1, 1))
	buff_input.add_theme_stylebox_override("focus", panel_style(INPUT_SURFACE, GOLD, 1, 1))
	buff_input.add_theme_color_override("font_color", TEXT)
	buff_input.tooltip_text = "Buffs to all chars on move use:\nA (ATK), D (DEF), H (Heal), CR (Crit Rate %), CD (Crit Dmg %)\ne.g. H40 CR20 CD40 or A10D20\n(Skill: 6s, Ult: 30s, HP: instant heal)"
	buff_input.text_changed.connect(_on_buff_text_changed.bind(buff_key))
	editor.add_child(buff_input)
	_register_editable_control(buff_input)
	return editor

func _on_buff_text_changed(new_text: String, buff_key: String) -> void:
	attack_buffs[buff_key] = new_text.strip_edges()

func make_label(text_value: String, font_size: int, color: Color) -> Label:
	var label = Label.new()
	label.text = text_value
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	if ui_font != null:
		label.add_theme_font_override("font", ui_font)
	return label

func make_stat_label(stat_name: String) -> Label:
	var icon: String = STAT_ICONS.get(stat_name, "•")
	var label := make_label(icon + "  " + stat_name.to_upper(), 9, MUTED)
	label.tooltip_text = stat_name
	return label

func make_rule() -> HSeparator:
	var rule = HSeparator.new()
	rule.add_theme_constant_override("separation", 1)
	rule.add_theme_stylebox_override("separator", panel_style(BORDER_DARK, BORDER_DARK, 0, 0))
	return rule

func panel_style(fill: Color, border: Color, border_width: int, radius: int) -> StyleBoxFlat:
	var style = StyleBoxFlat.new()
	style.bg_color = fill
	style.border_color = border
	style.set_border_width_all(border_width)
	style.set_corner_radius_all(radius)
	style.content_margin_left = 9
	style.content_margin_right = 9
	style.content_margin_top = 4
	style.content_margin_bottom = 4
	style.shadow_color = Color(0, 0, 0, 0.32)
	style.shadow_size = 3
	return style

func _on_start_pressed():
	_on_save_pressed()
	get_tree().change_scene_to_file("res://main.tscn")

func _get_selected_model_config() -> Dictionary:
	var idx := model_dropdown.selected if model_dropdown != null else 0
	match idx:
		1:
			return {
				"name": "DQN",
				"script": "res://RL/train_dqn.py",
				"checkpoint": "res://RL/checkpoints/diluc_dqn.pt"
			}
		2:
			return {
				"name": "Q-LEARNING",
				"script": "res://RL/train_qlearning.py",
				"checkpoint": "res://RL/checkpoints/diluc_qlearning.pt"
			}
		_:
			return {
				"name": "PPO",
				"script": "res://RL/train_ppo.py",
				"checkpoint": "res://RL/checkpoints/diluc_ppo.pt"
			}

func _on_train_pressed():
	if training_in_progress:
		return
	_stop_policy_server()
	_on_save_pressed()
	var progress_file := FileAccess.open(training_progress_path, FileAccess.WRITE)
	if progress_file != null:
		progress_file.close()
	training_progress_offset = 0
	training_in_progress = true
	_set_training_locked(true)
	training_status_label.text = "saving..."
	training_status_label.visible = true
	var python_executable := "python"
	var virtualenv_python := ProjectSettings.globalize_path("res://.venv/Scripts/python.exe")
	if FileAccess.file_exists(virtualenv_python):
		python_executable = virtualenv_python

	var model_cfg := _get_selected_model_config()
	var trainer_script := ProjectSettings.globalize_path(model_cfg["script"])
	var stats_path := ProjectSettings.globalize_path(DEFAULT_STATS_PATH)
	var checkpoint := ProjectSettings.globalize_path(model_cfg["checkpoint"])
	var progress_path := ProjectSettings.globalize_path(training_progress_path)
	var process_id := OS.create_process(python_executable, [
		trainer_script,
		"--stats", stats_path,
		"--output", checkpoint,
		"--progress", progress_path,
		"--device", device_selection,
		"--updates", "500"
	], false)
	if process_id == -1:
		training_in_progress = false
		_set_training_locked(false)
		training_status_label.text = "training failed to start"
		push_error("Unable to start " + str(model_cfg["name"]) + " training with " + python_executable)

func _on_simulate_pressed():
	if training_in_progress:
		training_status_label.text = "wait for training to finish"
		return
	_stop_policy_server()
	_on_save_pressed()
	get_tree().root.set_meta("rl_simulation", true)
	var model_cfg := _get_selected_model_config()
	var checkpoint := ProjectSettings.globalize_path(model_cfg["checkpoint"])
	var server_script := ProjectSettings.globalize_path("res://RL/policy_server.py")
	var python_executable := "python"
	var virtualenv_python := ProjectSettings.globalize_path("res://.venv/Scripts/python.exe")
	if FileAccess.file_exists(virtualenv_python):
		python_executable = virtualenv_python
	var server_pid := OS.create_process(python_executable, [server_script, "--checkpoint", checkpoint], false)
	if server_pid == -1:
		push_error("Unable to start " + str(model_cfg["name"]) + " policy server with " + python_executable)
		get_tree().root.set_meta("rl_simulation", false)
		return
	get_tree().root.set_meta("rl_policy_server_pid", server_pid)
	get_tree().change_scene_to_file("res://main.tscn")

func _stop_policy_server() -> void:
	var server_pid: int = int(get_tree().root.get_meta("rl_policy_server_pid", -1))
	if server_pid > 0:
		OS.kill(server_pid)
		get_tree().root.set_meta("rl_policy_server_pid", -1)

func _load_stats():
	var parsed: Variant = _read_stats_file(SAVED_STATS_PATH)
	if not parsed is Dictionary:
		parsed = _read_stats_file(DEFAULT_STATS_PATH)
	if not parsed is Dictionary:
		push_error("Unable to load a valid combat stats file.")
		return
	config_data = parsed
	var characters: Dictionary = config_data.get("characters", {})
	_apply_character_stats(diluc_stats, characters.get("Knight", {}))
	_apply_character_stats(lynae_stats, characters.get("Rogue", {}))
	_apply_character_stats(lisa_stats, characters.get("Mage", {}))
	var configured_boss: Dictionary = config_data.get("boss", {})
	boss_stats["HP"] = configured_boss.get("max_hp", boss_stats["HP"])
	boss_stats["ATK"] = configured_boss.get("attack", boss_stats["ATK"])
	boss_stats["DEF"] = configured_boss.get("defense", boss_stats["DEF"])
	_load_attack_patterns(diluc_attacks, characters.get("Knight", {}))
	_load_attack_patterns(lynae_attacks, characters.get("Rogue", {}))
	_load_attack_patterns(lisa_attacks, characters.get("Mage", {}))
	_load_attack_patterns(boss_attacks, configured_boss)
	attack_buffs = config_data.get("attack_buffs", {}) as Dictionary

func _read_stats_file(path: String) -> Variant:
	if not FileAccess.file_exists(path):
		return null
	var file: FileAccess = FileAccess.open(path, FileAccess.READ)
	if file == null:
		return null
	var parsed: Variant = JSON.parse_string(file.get_as_text())
	file.close()
	return parsed

func _apply_character_stats(target: Dictionary, source: Dictionary):
	if source.is_empty():
		return
	target["HP"] = source.get("max_hp", target["HP"])
	target["ATK"] = source.get("attack", target["ATK"])
	target["DEF"] = source.get("defense", target["DEF"])
	target["Crit Rate (%)"] = float(source.get("crit_rate", 0.5)) * 100.0
	target["Crit Dmg (%)"] = float(source.get("crit_damage", 1.5)) * 100.0

func _on_stat_changed(value: String, target: Dictionary, key: String):
	var number = value.to_float()
	if value.strip_edges() != "" and number >= 0.0:
		target[key] = number

func _on_attack_token_changed(value: String, attacks: Dictionary, attack_name: String, index: int, input: LineEdit):
	if sanitizing_attack_input:
		return
	var digits_only := ""
	for character in value:
		if character >= "0" and character <= "9":
			digits_only += character
	if digits_only.is_empty():
		digits_only = "0"
	if digits_only != value:
		sanitizing_attack_input = true
		input.text = digits_only
		sanitizing_attack_input = false
	value = digits_only
	var tokens: Array[String] = _pattern_tokens(attacks.get(attack_name, ""))
	if index < 0 or index >= tokens.size():
		return
	tokens[index] = value.strip_edges()
	attacks[attack_name] = tokens

func _pattern_tokens(pattern: Variant) -> Array[String]:
	if pattern is Array:
		var tokens: Array[String] = []
		for token in pattern:
			var normalized_token := str(token)
			tokens.append("0" if normalized_token.is_empty() or normalized_token == "_" else normalized_token)
		return tokens
	var text := str(pattern).strip_edges()
	if text.contains(" "):
		var spaced_tokens: Array[String] = []
		for token in text.split(" ", false):
			spaced_tokens.append("0" if token.is_empty() or token == "_" else token)
		return spaced_tokens
	var char_tokens: Array[String] = []
	for index in range(text.length()):
		var token := text[index]
		char_tokens.append("0" if token == "_" else token)
	return char_tokens

func _load_attack_patterns(target: Dictionary, source: Dictionary):
	var patterns: Dictionary = source.get("attack_patterns", {}) as Dictionary
	for attack_name in target:
		if patterns.has(attack_name) and patterns[attack_name] is String:
			var loaded_tokens := _pattern_tokens(patterns[attack_name])
			if not _is_legacy_all_hit_pattern(loaded_tokens):
				target[attack_name] = loaded_tokens
		elif patterns.has(attack_name) and patterns[attack_name] is Array:
			target[attack_name] = patterns[attack_name]

func _is_legacy_all_hit_pattern(tokens: Array[String]) -> bool:
	if tokens.is_empty():
		return false
	for token in tokens:
		if token.to_upper() != "X":
			return false
	return true

func _on_save_pressed():
	if config_data.is_empty():
		return
	_write_character_stats(config_data["characters"]["Knight"], diluc_stats)
	_write_character_stats(config_data["characters"]["Rogue"], lynae_stats)
	_write_character_stats(config_data["characters"]["Mage"], lisa_stats)
	_write_attack_patterns(config_data["characters"]["Knight"], diluc_attacks)
	_write_attack_patterns(config_data["characters"]["Rogue"], lynae_attacks)
	_write_attack_patterns(config_data["characters"]["Mage"], lisa_attacks)
	config_data["boss"]["max_hp"] = boss_stats["HP"]
	config_data["boss"]["attack"] = boss_stats["ATK"]
	config_data["boss"]["defense"] = boss_stats["DEF"]
	_write_attack_patterns(config_data["boss"], boss_attacks)
	config_data["attack_buffs"] = attack_buffs.duplicate(true)
	var json_text := JSON.stringify(config_data, "\t")
	var saved_user_file: FileAccess = FileAccess.open(SAVED_STATS_PATH, FileAccess.WRITE)
	if saved_user_file == null:
		push_error("Unable to save combat stats to " + SAVED_STATS_PATH)
	else:
		saved_user_file.store_string(json_text)
		saved_user_file.flush()
		saved_user_file.close()
	var project_file: FileAccess = FileAccess.open(DEFAULT_STATS_PATH, FileAccess.WRITE)
	if project_file != null:
		project_file.store_string(json_text)
		project_file.flush()
		project_file.close()

func _write_attack_patterns(target: Dictionary, source: Dictionary):
	var patterns: Dictionary = {}
	for attack_name in source:
		var tokens: Array[String] = _pattern_tokens(source[attack_name])
		patterns[attack_name] = tokens.duplicate()
	target["attack_patterns"] = patterns

func _write_character_stats(target: Dictionary, source: Dictionary):
	target["max_hp"] = source["HP"]
	target["attack"] = source["ATK"]
	target["defense"] = source["DEF"]
	target["crit_rate"] = source["Crit Rate (%)"] / 100.0
	target["crit_damage"] = source["Crit Dmg (%)"] / 100.0

func _on_quit_pressed():
	get_tree().quit()
