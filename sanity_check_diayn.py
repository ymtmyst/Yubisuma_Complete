"""DIAYN minimal prototype の動作確認スクリプト.

検証項目:
- config OBS_TOTAL = 114
- env.reset() の observation shape と persona one-hot
- env.step() でエピソード内 persona 不変
- YubisumaFeaturesExtractor.persona_predictions の forward
- set_agent_persona による固定
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from gymnasium import spaces

from rl.config import (
    OBS_TOTAL, OBS_PERSONA, NUM_PERSONA_TP, NUM_PERSONA_NTP,
    LAMBDA_DIVERSITY_TP, LAMBDA_DIVERSITY_NTP,
)
from rl.env import YubisumaEnv
from rl.network import YubisumaFeaturesExtractor


def test_config():
    print(f"[1] OBS_TOTAL = {OBS_TOTAL}")
    assert OBS_TOTAL == 114, f"Expected 114, got {OBS_TOTAL}"
    print(f"    OBS_PERSONA = {OBS_PERSONA}")
    assert OBS_PERSONA == 10
    print(f"    NUM_PERSONA_TP = {NUM_PERSONA_TP}, NUM_PERSONA_NTP = {NUM_PERSONA_NTP}")
    assert NUM_PERSONA_TP == 7 and NUM_PERSONA_NTP == 3
    print(f"    LAMBDA_DIVERSITY_TP = {LAMBDA_DIVERSITY_TP}, "
          f"LAMBDA_DIVERSITY_NTP = {LAMBDA_DIVERSITY_NTP}")
    print("    OK\n")


def test_env_observation():
    print("[2] env.reset() の observation を確認")
    env = YubisumaEnv()
    obs, info = env.reset(seed=42)
    print(f"    obs.shape = {obs.shape}, dtype = {obs.dtype}")
    assert obs.shape == (OBS_TOTAL,), f"Expected ({OBS_TOTAL},), got {obs.shape}"

    persona_part = obs[-OBS_PERSONA:]
    tp_part = persona_part[:NUM_PERSONA_TP]
    ntp_part = persona_part[NUM_PERSONA_TP:]
    print(f"    persona TP one-hot: {tp_part}")
    print(f"    persona NTP one-hot: {ntp_part}")
    assert tp_part.sum() == 1.0
    assert ntp_part.sum() == 1.0

    assert 'agent_persona_tp' in info
    assert 'agent_persona_ntp' in info
    print(f"    info.agent_persona_tp = {info['agent_persona_tp']}")
    print(f"    info.agent_persona_ntp = {info['agent_persona_ntp']}")
    assert int(tp_part.argmax()) == info['agent_persona_tp']
    assert int(ntp_part.argmax()) == info['agent_persona_ntp']
    print("    OK\n")

    return env


def test_env_step(env):
    print("[3] env.step() を 20 回回して shape と persona 不変を確認")
    obs, info = env.reset(seed=42)
    initial_tp = info['agent_persona_tp']
    initial_ntp = info['agent_persona_ntp']

    for step in range(20):
        mask = env.action_masks()
        valid = np.where(mask)[0]
        if len(valid) == 0:
            print(f"    step {step}: no valid action, breaking")
            break
        action = int(np.random.choice(valid))
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_TOTAL,), f"shape mismatch at step {step}: {obs.shape}"
        if 'agent_persona_tp' in info:
            assert info['agent_persona_tp'] == initial_tp
            assert info['agent_persona_ntp'] == initial_ntp
        if terminated or truncated:
            print(f"    step {step}: episode ended "
                  f"(terminated={terminated}, truncated={truncated})")
            break
    print("    OK\n")


def test_network_forward():
    print("[4] YubisumaFeaturesExtractor + persona_predictions の forward")
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_TOTAL,), dtype=np.float32)
    fe = YubisumaFeaturesExtractor(obs_space)

    batch_size = 8
    obs_batch = torch.randn(batch_size, OBS_TOTAL)
    features = fe(obs_batch)
    print(f"    features.shape = {tuple(features.shape)}")
    assert features.shape == (batch_size, 256)

    aux_preds = fe.get_aux_predictions(features)
    print(f"    aux reaction: {tuple(aux_preds['reaction'].shape)}")
    print(f"    aux thumbs:   {tuple(aux_preds['thumbs'].shape)}")
    print(f"    aux skill:    {tuple(aux_preds['skill'].shape)}")
    print(f"    aux lookahead:{tuple(aux_preds['lookahead'].shape)}")

    tp_logits, ntp_logits = fe.persona_predictions(features)
    print(f"    persona_tp_logits:  {tuple(tp_logits.shape)}")
    print(f"    persona_ntp_logits: {tuple(ntp_logits.shape)}")
    assert tp_logits.shape == (batch_size, NUM_PERSONA_TP)
    assert ntp_logits.shape == (batch_size, NUM_PERSONA_NTP)
    print("    OK\n")


def test_force_persona():
    print("[5] set_agent_persona() による固定動作")
    env = YubisumaEnv()
    env.set_agent_persona(persona_tp=3, persona_ntp=1)
    obs, info = env.reset(seed=1)
    assert info['agent_persona_tp'] == 3
    assert info['agent_persona_ntp'] == 1
    persona_part = obs[-OBS_PERSONA:]
    assert persona_part[3] == 1.0
    assert persona_part[NUM_PERSONA_TP + 1] == 1.0
    print(f"    forced persona reflected: TP=3, NTP=1")
    print("    OK\n")


def test_diayn_loss_step():
    print("[6] DIAYN loss の forward/backward 動作確認")
    import torch.nn.functional as F

    obs_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_TOTAL,), dtype=np.float32)
    fe = YubisumaFeaturesExtractor(obs_space)

    batch = 16
    obs = torch.randn(batch, OBS_TOTAL)
    # persona one-hot 部分を zero-mask (DiversityLossCallback と同じ手順)
    obs[:, -OBS_PERSONA:] = 0.0
    tp_lbl = torch.randint(0, NUM_PERSONA_TP, (batch,))
    ntp_lbl = torch.randint(0, NUM_PERSONA_NTP, (batch,))

    features = fe.shared_net(obs)
    tp_logits, ntp_logits = fe.persona_predictions(features)

    loss_tp = F.cross_entropy(tp_logits, tp_lbl)
    loss_ntp = F.cross_entropy(ntp_logits, ntp_lbl)
    loss = LAMBDA_DIVERSITY_TP * loss_tp + LAMBDA_DIVERSITY_NTP * loss_ntp

    optimizer = torch.optim.Adam(fe.parameters(), lr=3e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"    loss_tp = {loss_tp.item():.4f}, loss_ntp = {loss_ntp.item():.4f}")
    print(f"    combined loss = {loss.item():.4f}")
    print(f"    expected initial loss ~ ln({NUM_PERSONA_TP})*{LAMBDA_DIVERSITY_TP} "
          f"+ ln({NUM_PERSONA_NTP})*{LAMBDA_DIVERSITY_NTP} = "
          f"{np.log(NUM_PERSONA_TP)*LAMBDA_DIVERSITY_TP + np.log(NUM_PERSONA_NTP)*LAMBDA_DIVERSITY_NTP:.4f}")
    print("    OK\n")


if __name__ == "__main__":
    test_config()
    env = test_env_observation()
    test_env_step(env)
    test_network_forward()
    test_force_persona()
    test_diayn_loss_step()
    print("=" * 50)
    print("All sanity checks passed!")
    print("=" * 50)
