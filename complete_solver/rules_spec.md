# Complete Solver-Facing Rule Notes

This package is the pure, solver-facing rules layer. It is intentionally
separate from the interactive game files.

## Perspective

`State` is canonicalized to the current turn player's perspective:

- `state.me` is the turn player.
- `state.opp` is the non-turn player.
- A normal turn switch swaps perspective.
- Extra turns keep the same perspective and decrement pending extra turns.

## Simultaneous Action Unit

The matrix-game unit is one public state.

- TP complete action: skill/number, TP thumb, and any choice/all-order data.
- NTP complete action: reaction and NTP thumb.
- Both actions are selected before resolution.

## Stock

Stock is a finite set, not a list.

- Duplicate copies of the same skill are not stored.
- Stockable skills are normal skills plus anti-counter skills.
- With mirror enabled: max 9 stock skills.
- With mirror disabled: max 8 stock skills.

## Scope

The current implementation targets exact-solver and learning-environment
foundations. It resolves one turn without printing, input, or randomness.
