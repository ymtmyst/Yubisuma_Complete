import sys
sys.path.insert(0, ".")

from rl.actions import decode_action, encode_tp_action, get_action_mask
from rl.config import TP_SKILL_OPTIONS, OBS_TOTAL
from rl.env import YubisumaEnv
from yubisuma_constants import KEY_PLAYER, KEY_COMPUTER


def tp_choice_action(target, thumbs=0):
    wanted = f"チョイス:{target}"
    for idx, skill in enumerate(TP_SKILL_OPTIONS):
        if skill == wanted:
            return encode_tp_action(idx, thumbs)
    raise AssertionError(f"missing action for {wanted}")


def test(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"OK: {name}")


env = YubisumaEnv()
obs, info = env.reset(seed=1)

env.agent_key = KEY_PLAYER
env.opponent_key = KEY_COMPUTER
env.game_state.current_player_key = KEY_PLAYER
env.game_state.effects.first_player_key = KEY_PLAYER
env.game_state.effects.is_first_phase_done[KEY_PLAYER] = True
env.game_state.effects.is_first_phase_done[KEY_COMPUTER] = True

flash = "フラッシュ"
feint = "フェイント"
env.game_state.player.stock = [flash, feint]
env._get_opponent_ntp_action = lambda: {
    "role": "ntp",
    "skill": None,
    "reaction": "カウンター",
    "thumbs": 0,
    "choice_target": None,
}

obs, reward, terminated, truncated, info = env.step(tp_choice_action(flash, thumbs=0))

test("choice declaration creates pending state", env.game_state.pending_choice is not None)
test("turn is not resolved before target selection", env.turn_count == 0)
test("revealed counter is encoded", obs.shape == (OBS_TOTAL,) and obs[-4:].tolist() == [1.0, 0.0, 1.0, 0.0])

mask = get_action_mask(env.game_state, env.agent_key)
valid_choice_targets = [
    decode_action(i)["choice_target"]
    for i, ok in enumerate(mask)
    if ok and decode_action(i)["choice_target"]
]
test("pending choice can select after counter", set(valid_choice_targets) == {flash, feint})

obs, reward, terminated, truncated, info = env.step(tp_choice_action(feint, thumbs=0))

test("post-counter feint target is applied", env.game_state.player.get_active_hands() == 1)
test("choice turn is recorded once", env.turn_count == 1 and len(env.episode_turns) == 1)
test("record stores selected target", env.episode_turns[0]["choice_target"] == feint)
test("pending state is cleared", env.game_state.pending_choice is None)
