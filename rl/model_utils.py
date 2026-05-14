"""Utilities for importing and loading MaskablePPO models safely."""

from __future__ import annotations

import io
import pathlib
import warnings
from contextlib import redirect_stderr
from typing import Any

import torch as th

with redirect_stderr(io.StringIO()):
    from stable_baselines3.common.base_class import (
        _convert_space,
        check_for_correct_spaces,
        get_system_info,
    )
    from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr


def import_maskable_ppo():
    """Import MaskablePPO while suppressing gym's unmaintained notice."""
    with redirect_stderr(io.StringIO()):
        from sb3_contrib import MaskablePPO

    return MaskablePPO


def _is_compatibility_load_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "aux_lookahead_head" in msg
        or "parameter group that doesn't match the size of optimizer's group" in msg
    )


def load_maskable_ppo(
    path: str | pathlib.Path,
    env=None,
    device: str | th.device = "auto",
    custom_objects: dict[str, Any] | None = None,
    print_system_info: bool = False,
    force_reset: bool = True,
    **kwargs,
):
    """
    Load a MaskablePPO model with backward compatibility for older checkpoints.

    Older checkpoints do not include the new auxiliary lookahead head, and their
    optimizer state no longer matches the current parameter groups. In that case,
    we reload weights non-strictly and skip optimizer state restoration.
    """
    MaskablePPO = import_maskable_ppo()
    try:
        return MaskablePPO.load(
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs,
        )
    except (RuntimeError, ValueError) as exc:
        if not _is_compatibility_load_error(exc):
            raise
        warnings.warn(
            "Loading checkpoint in compatibility mode. "
            "Missing new auxiliary weights will use current initialization, "
            "and optimizer state will be skipped."
        )
        return _load_maskable_ppo_compat(
            MaskablePPO,
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs,
        )


def _load_maskable_ppo_compat(
    algo_cls,
    path,
    env=None,
    device: str | th.device = "auto",
    custom_objects: dict[str, Any] | None = None,
    print_system_info: bool = False,
    force_reset: bool = True,
    **kwargs,
):
    if print_system_info:
        print("== CURRENT SYSTEM INFO ==")
        get_system_info()

    data, params, pytorch_variables = load_from_zip_file(
        path,
        device=device,
        custom_objects=custom_objects,
        print_system_info=print_system_info,
    )

    assert data is not None, "No data found in the saved file"
    assert params is not None, "No params found in the saved file"

    if "policy_kwargs" in data:
        if "device" in data["policy_kwargs"]:
            del data["policy_kwargs"]["device"]
        saved_net_arch = data["policy_kwargs"].get("net_arch")
        if saved_net_arch and isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
            data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

    if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
        raise ValueError(
            f"The specified policy kwargs do not equal the stored policy kwargs."
            f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
        )

    if "observation_space" not in data or "action_space" not in data:
        raise KeyError("The observation_space and action_space were not given, can't verify new environments")

    for key in {"observation_space", "action_space"}:
        data[key] = _convert_space(data[key])

    if env is not None:
        env = algo_cls._wrap_env(env, data["verbose"])
        check_for_correct_spaces(env, data["observation_space"], data["action_space"])
        if force_reset:
            data["_last_obs"] = None
        data["n_envs"] = env.num_envs
    elif "env" in data:
        env = data["env"]

    model = algo_cls(
        policy=data["policy_class"],
        env=env,
        device=device,
        _init_setup_model=False,
    )
    model.__dict__.update(data)
    model.__dict__.update(kwargs)
    model._setup_model()

    compat_params = dict(params)
    compat_params.pop("policy.optimizer", None)
    model.set_parameters(compat_params, exact_match=False, device=device)

    if pytorch_variables is not None:
        for name in pytorch_variables:
            if pytorch_variables[name] is None:
                continue
            recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

    if model.use_sde:
        model.policy.reset_noise()

    return model
