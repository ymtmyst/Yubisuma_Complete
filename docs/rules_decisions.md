# Complete Solver Rule Decisions

This document fixes the solver-facing interpretation used by
`complete_solver`. It is intended to make tests independent from the
interactive implementation and from prose that may still be revised.

## Public State Perspective

- `State.me` is always the current turn player.
- `State.opp` is always the non-turn player.
- A normal turn switch swaps the perspective.
- Extra turns keep the same perspective.
- `previous_skill` is the immediately previous public declaration.

## Simultaneous Action Unit

- One matrix-game row is a complete TP action: skill or number, TP thumb,
  optional Choice target, and optional All order.
- One matrix-game column is a complete NTP action: reaction and NTP thumb.
- TP and NTP actions are selected before resolution.
- Choice is represented as a complete action target in the solver. This keeps
  the exact solver simultaneous and avoids hidden post-reaction decisions.

## Legal Skill Decisions

- Skip means the TP has no skill or number choice. The solver emits only
  `PASS`; legacy `get_valid_skills` returns no skills.
- Block is an NTP reaction and is not a TP skill.
- Mirror is controlled by `RulesConfig.enable_mirror`.
- Reversi is controlled by `RulesConfig.enable_reversi`.
- A used ultimate removes Boost, Reversi, and Time from TP legal skills.
- Choice, All, and Drop share the same one-phase limit.
- Drop removes only the skills currently in the acting player's stock from the
  opponent's next phase.

## Anti-Counter Skills Via Reference (designer ruling 2026-07-13)

- Feint and Lock only activate when the opponent declares Counter. This
  condition is INHERITED when they are executed via Copy, Choice, or All:
  with no counter they do NOTHING (no hand drop, no extra turn, no lock).
- With a counter, the countered reference resolves per the referenced skill
  (Copy: twice; Choice/All: once per execution).
- HISTORY: until 2026-07-13 both the solver and the interactive
  implementation executed referenced feint/lock unconditionally. Both shared
  the same misreading, so the May cross-validation could not catch it; the
  designer spotted it from the policy-distribution report (copy-finishes at
  76.8%). All models/tables trained before the fix are invalid.

## Reference And Stock

- Copy can reference a previous number or a previous referenceable skill.
- Stock can reference only previous non-number referenceable skills.
- Referenceable skills are normal skills plus anti-counter skills.
- Reference skills and ultimate skills are not referenceable.
- Solver stock is a finite set. It never stores duplicate copies of the same
  skill, and Stock is illegal when the previous skill is already stocked.
- With Mirror disabled, stockable skills exclude Mirror. With Mirror enabled,
  Mirror is stockable.
- All always executes every stocked skill; any omitted order entries are
  appended in deterministic sorted order.

## Buffs, Debuffs, And Field Effects

- Guard blocks a two-hand drop once and consumes itself.
- Guard grants at most one extra turn per phase.
- Charge is consumed by the next number declaration and makes that number
  resolve twice.
- Quick has levels 2 and 1, then expires.
- Lock is pending until the target's next NTP window, then blocks NTP reactions
  for that turn.
- Reversi swaps hands, cement, guard, charge, quick, mirror readiness, lock, and
  drop-blocked skills. It does not swap stock, skip phases, time, ultimate use,
  extra turns, or declaration history.
- Mirror reflects only effects where TP would lower hands or apply a debuff to
  NTP. Directly reflectable skills are Number, Flash, Cement, and Drop.
- Guard, Charge, Quick, Mirror preparation, and Skip are not Mirror-reflectable.
  If NTP declares Mirror against them, Mirror is consumed and the TP skill
  resolves normally.
- Feint and Lock do not trigger under Mirror because they require NTP to declare
  Counter. Boost, Reversi, and Time are not Mirror-reflectable.
- Copy, Stock, Choice, and All follow the referenced skill. Each referenced
  skill is checked against the same Mirror-reflectable list. All resolves every
  stocked skill, so multiple reflectable effects in one All are all reflected.

## Current Compatibility Tests

- `complete_solver/tests/test_legal_actions.py` compares solver TP skill
  legality against the existing `get_valid_skills` for representative states.
- The unique-stock rule is intentionally stricter than the current interactive
  list-based stock implementation and is fixed here for finite exact solving.
