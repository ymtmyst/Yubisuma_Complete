import sys
sys.path.append('.')
from yubisuma_logic import GameState
from yubisuma_turn_handler import TurnHandler
from yubisuma_constants import KEY_PLAYER, KEY_COMPUTER

gs = GameState()
gs.initialize_game()
gs.current_player_key = KEY_PLAYER

print("--- P1: スキップ ---")
gs.on_phase_start(KEY_PLAYER)
TurnHandler.resolve_turn(gs, KEY_PLAYER, "スキップ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
gs.on_phase_end(KEY_PLAYER)

print("\n--- C1: skipped ---")
gs.on_phase_start(KEY_COMPUTER)
if gs.computer.skip_phases > 0:
    print("Computer skipped")
gs.on_phase_end(KEY_COMPUTER)

print("\n--- P2: コピー ---")
gs.on_phase_start(KEY_PLAYER)
TurnHandler.resolve_turn(gs, KEY_PLAYER, "コピー", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
gs.on_phase_end(KEY_PLAYER)

print("\n--- C2: skipped ---")
gs.on_phase_start(KEY_COMPUTER)
if gs.computer.skip_phases > 0:
    print("Computer skipped")
gs.on_phase_end(KEY_COMPUTER)

print("\n--- P3: 1 ---")
gs.on_phase_start(KEY_PLAYER)
TurnHandler.resolve_turn(gs, KEY_PLAYER, 1, {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
gs.on_phase_end(KEY_PLAYER)

print(f"\nComputer skip_phases before C3: {gs.computer.skip_phases}")

print("\n--- C3: skipped? ---")
gs.on_phase_start(KEY_COMPUTER)
if gs.computer.skip_phases > 0:
    print("Computer skipped")
gs.on_phase_end(KEY_COMPUTER)
