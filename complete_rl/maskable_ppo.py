"""MaskablePPO baseline for the Complete RL environment.

The environment is written from the current turn player's perspective, so a
single policy naturally controls both TP players as the turn perspective
switches.  NTP reactions are still supplied by the environment's
``opponent_policy`` hook; the default baseline uses random legal reactions.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from complete_solver import RulesConfig
from complete_rl.env import CompleteEnv, MIXED_NTP_POLICIES, NAMED_NTP_POLICIES, REWARD_MODES


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    wins: int
    losses: int
    truncations: int
    average_reward: float
    average_steps: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingArtifacts:
    model_path: Path | None
    metrics_path: Path | None


@dataclass(frozen=True)
class TrainingResult:
    evaluation: EvaluationResult
    artifacts: TrainingArtifacts


@dataclass(frozen=True)
class TrainingPreset:
    total_timesteps: int
    n_steps: int
    batch_size: int
    n_epochs: int
    eval_episodes: int
    max_steps: int
    learning_rate: float = 3e-4
    gamma: float = 0.99

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class SeedRunResult:
    seed: int
    evaluation: EvaluationResult
    model_path: Path | None
    metrics_path: Path | None
    reused_existing: bool = False

    def to_dict(self) -> dict[str, int | float | str | None]:
        row: dict[str, int | float | str | None] = {"seed": self.seed}
        row.update(self.evaluation.to_dict())
        row["model_path"] = str(self.model_path) if self.model_path else None
        row["metrics_path"] = str(self.metrics_path) if self.metrics_path else None
        row["reused_existing"] = str(self.reused_existing)
        return row


@dataclass(frozen=True)
class MultiSeedResult:
    runs: tuple[SeedRunResult, ...]
    summary_path: Path | None
    metrics_path: Path | None

    @property
    def mean_reward(self) -> float:
        if not self.runs:
            return 0.0
        return float(np.mean([run.evaluation.average_reward for run in self.runs]))

    @property
    def mean_steps(self) -> float:
        if not self.runs:
            return 0.0
        return float(np.mean([run.evaluation.average_steps for run in self.runs]))

    @property
    def total_wins(self) -> int:
        return sum(run.evaluation.wins for run in self.runs)

    @property
    def total_losses(self) -> int:
        return sum(run.evaluation.losses for run in self.runs)

    @property
    def total_truncations(self) -> int:
        return sum(run.evaluation.truncations for run in self.runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": [run.seed for run in self.runs],
            "runs": [run.to_dict() for run in self.runs],
            "aggregate": {
                "mean_reward": self.mean_reward,
                "mean_steps": self.mean_steps,
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
                "total_truncations": self.total_truncations,
            },
        }


@dataclass(frozen=True)
class ConfigRunResult:
    label: str
    config: RulesConfig
    result: MultiSeedResult

    def to_dict(self) -> dict[str, Any]:
        payload = self.result.to_dict()
        payload["label"] = self.label
        payload["config"] = _config_to_dict(self.config)
        return payload


@dataclass(frozen=True)
class AllConfigsResult:
    configs: tuple[ConfigRunResult, ...]
    summary_path: Path | None
    metrics_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "configs": [config.to_dict() for config in self.configs],
            "aggregate": {
                "mean_reward_by_config": {
                    config.label: config.result.mean_reward for config in self.configs
                },
                "mean_steps_by_config": {
                    config.label: config.result.mean_steps for config in self.configs
                },
            },
        }


@dataclass(frozen=True)
class EvaluationSuiteRow:
    config_label: str
    config: RulesConfig
    seed: int
    ntp_policy: str
    model_path: Path
    evaluation: EvaluationResult

    def to_dict(self) -> dict[str, int | float | str]:
        row: dict[str, int | float | str] = {
            "config_label": self.config_label,
            "enable_mirror": str(self.config.enable_mirror),
            "enable_reversi": str(self.config.enable_reversi),
            "seed": self.seed,
            "ntp_policy": self.ntp_policy,
            "model_path": str(self.model_path),
        }
        row.update(self.evaluation.to_dict())
        return row


@dataclass(frozen=True)
class EvaluationSuiteResult:
    rows: tuple[EvaluationSuiteRow, ...]
    summary_path: Path | None
    metrics_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "aggregate": {
                "mean_reward_by_config_and_policy": _aggregate_eval_rows(self.rows),
            },
        }


TRAINING_PRESETS: dict[str, TrainingPreset] = {
    "smoke": TrainingPreset(
        total_timesteps=8,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        eval_episodes=1,
        max_steps=20,
    ),
    "quick": TrainingPreset(
        total_timesteps=20_000,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        eval_episodes=20,
        max_steps=500,
    ),
    "standard": TrainingPreset(
        total_timesteps=250_000,
        n_steps=1024,
        batch_size=128,
        n_epochs=8,
        eval_episodes=100,
        max_steps=500,
    ),
}


ALL_CONFIGS: tuple[tuple[str, RulesConfig], ...] = (
    ("mirror_off_reversi_off", RulesConfig(enable_mirror=False, enable_reversi=False)),
    ("mirror_on_reversi_off", RulesConfig(enable_mirror=True, enable_reversi=False)),
    ("mirror_off_reversi_on", RulesConfig(enable_mirror=False, enable_reversi=True)),
    ("mirror_on_reversi_on", RulesConfig(enable_mirror=True, enable_reversi=True)),
)


def has_maskable_ppo_dependencies() -> bool:
    """Return True when optional training dependencies are importable."""
    try:
        import sb3_contrib  # noqa: F401
        import stable_baselines3  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def make_env(
    config: RulesConfig = RulesConfig(),
    *,
    seed: int | None = None,
    max_steps: int = 500,
    ntp_policy: str = "random",
    reward_mode: str = "terminal",
) -> CompleteEnv:
    """Build a single CompleteEnv for masked self-play training."""
    env = CompleteEnv(
        config=config,
        opponent_policy=ntp_policy,
        max_steps=max_steps,
        reward_mode=reward_mode,
    )
    env.reset(seed=seed)
    return env


def build_model(
    env: CompleteEnv,
    *,
    seed: int | None = 0,
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 4,
    gamma: float = 0.99,
    device: str = "auto",
    tensorboard_log: str | None = None,
    verbose: int = 0,
):
    """Create an untrained MaskablePPO model for CompleteEnv."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "MaskablePPO baseline requires sb3-contrib, stable-baselines3, and torch. "
            "Install requirements.txt before training."
        ) from exc

    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )


def evaluate_model(
    model,
    config: RulesConfig = RulesConfig(),
    *,
    episodes: int = 20,
    seed: int | None = 10_000,
    max_steps: int = 500,
    deterministic: bool = True,
    ntp_policy: str = "random",
    reward_mode: str = "terminal",
) -> EvaluationResult:
    """Evaluate a MaskablePPO-like model using action masks."""
    env = make_env(
        config=config,
        seed=seed,
        max_steps=max_steps,
        ntp_policy=ntp_policy,
        reward_mode=reward_mode,
    )
    rewards: list[float] = []
    steps_list: list[int] = []
    wins = 0
    losses = 0
    truncations = 0

    for episode in range(episodes):
        reset_seed = None if seed is None else seed + episode
        obs, _ = env.reset(seed=reset_seed)
        total_reward = 0.0
        steps = 0

        while True:
            action_masks = env.action_masks()
            action, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=action_masks,
            )
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += float(reward)
            steps += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        steps_list.append(steps)
        if total_reward > 0:
            wins += 1
        elif total_reward < 0:
            losses += 1
        if steps >= max_steps and total_reward == 0:
            truncations += 1

    return EvaluationResult(
        episodes=episodes,
        wins=wins,
        losses=losses,
        truncations=truncations,
        average_reward=float(np.mean(rewards)) if rewards else 0.0,
        average_steps=float(np.mean(steps_list)) if steps_list else 0.0,
    )


def train_maskable_ppo(
    *,
    total_timesteps: int,
    config: RulesConfig = RulesConfig(),
    output_dir: str | Path | None = None,
    seed: int = 0,
    max_steps: int = 500,
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 4,
    gamma: float = 0.99,
    eval_episodes: int = 20,
    ntp_policy: str = "random",
    reward_mode: str = "terminal",
    device: str = "auto",
    tensorboard_log: str | None = None,
    verbose: int = 1,
    use_bc_pretrain: bool = False,
    bc_max_states: int = 400,
    bc_epochs: int = 5,
    bc_lr: float = 1e-3,
) -> TrainingResult:
    """Train and optionally save a MaskablePPO baseline."""
    env = make_env(
        config=config,
        seed=seed,
        max_steps=max_steps,
        ntp_policy=ntp_policy,
        reward_mode=reward_mode,
    )
    model = build_model(
        env,
        seed=seed,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
    if use_bc_pretrain:
        from complete_rl.bc_pretrain import generate_bc_dataset, bc_pretrain as _bc_pretrain
        if verbose:
            print(f"BC pre-train: generating dataset (max_states={bc_max_states})...")
        bc_dataset = generate_bc_dataset(
            config=config, max_states=bc_max_states, gamma=gamma
        )
        if verbose:
            print(f"BC pre-train: {len(bc_dataset)} states, training {bc_epochs} epochs...")
        _bc_pretrain(
            model,
            bc_dataset,
            n_epochs=bc_epochs,
            learning_rate=bc_lr,
            seed=seed,
            verbose=bool(verbose),
        )
    model.learn(total_timesteps=total_timesteps)

    evaluation = evaluate_model(
        model,
        config=config,
        episodes=eval_episodes,
        seed=seed + 100_000,
        max_steps=max_steps,
        ntp_policy=ntp_policy,
        reward_mode="terminal",
    )

    artifacts = TrainingArtifacts(model_path=None, metrics_path=None)
    if output_dir is not None:
        artifacts = _write_artifacts(
            model=model,
            evaluation=evaluation,
            output_dir=Path(output_dir),
            config=config,
            total_timesteps=total_timesteps,
            seed=seed,
            ntp_policy=ntp_policy,
            reward_mode=reward_mode,
        )
    return TrainingResult(evaluation=evaluation, artifacts=artifacts)


def evaluate_saved_model(
    model_path: str | Path,
    config: RulesConfig = RulesConfig(),
    *,
    episodes: int = 100,
    seed: int = 200_000,
    max_steps: int = 500,
    deterministic: bool = True,
    ntp_policy: str = "random",
    reward_mode: str = "terminal",
) -> EvaluationResult:
    """Load and evaluate a saved MaskablePPO model."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Evaluating a saved MaskablePPO model requires sb3-contrib."
        ) from exc

    model = MaskablePPO.load(str(model_path))
    return evaluate_model(
        model,
        config=config,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
        deterministic=deterministic,
        ntp_policy=ntp_policy,
        reward_mode=reward_mode,
    )


def evaluate_saved_model_directory(
    model_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    config: RulesConfig = RulesConfig(),
    config_label: str = "selected_config",
    all_configs: bool = False,
    ntp_policies: Iterable[str] = ("random",),
    episodes: int = 100,
    seed: int = 200_000,
    max_steps: int = 500,
) -> EvaluationSuiteResult:
    """Evaluate saved seed models from a training output directory."""
    root = Path(model_dir)
    rows: list[EvaluationSuiteRow] = []

    if all_configs:
        config_entries = tuple(
            (label, cfg, root / label) for label, cfg in ALL_CONFIGS
        )
    else:
        config_entries = ((config_label, config, root),)

    for label, cfg, directory in config_entries:
        for seed_value, model_path in _iter_seed_models(directory):
            for policy in ntp_policies:
                evaluation = evaluate_saved_model(
                    model_path,
                    config=cfg,
                    episodes=episodes,
                    seed=seed,
                    max_steps=max_steps,
                    ntp_policy=policy,
                )
                rows.append(
                    EvaluationSuiteRow(
                        config_label=label,
                        config=cfg,
                        seed=seed_value,
                        ntp_policy=policy,
                        model_path=model_path,
                        evaluation=evaluation,
                    )
                )

    output = Path(output_dir) if output_dir is not None else root
    output.mkdir(parents=True, exist_ok=True)
    result = EvaluationSuiteResult(
        rows=tuple(rows),
        summary_path=output / "evaluation_summary.csv",
        metrics_path=output / "evaluation_summary.json",
    )
    _write_evaluation_suite_artifacts(result, output_dir=output)
    return result


def train_maskable_ppo_multi_seed(
    *,
    seeds: Iterable[int],
    output_dir: str | Path,
    total_timesteps: int,
    config: RulesConfig = RulesConfig(),
    max_steps: int = 500,
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 4,
    gamma: float = 0.99,
    eval_episodes: int = 20,
    ntp_policy: str = "random",
    reward_mode: str = "terminal",
    device: str = "auto",
    tensorboard_log: str | None = None,
    verbose: int = 1,
    force: bool = False,
    use_bc_pretrain: bool = False,
    bc_max_states: int = 400,
    bc_epochs: int = 5,
    bc_lr: float = 1e-3,
) -> MultiSeedResult:
    """Train the same MaskablePPO setting across multiple random seeds."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs: list[SeedRunResult] = []

    for seed in seeds:
        seed_dir = output / f"seed_{seed}"
        existing = (
            None
            if force
            else _load_existing_seed_run(seed_dir, seed, ntp_policy, reward_mode)
        )
        if existing is not None:
            runs.append(existing)
            continue

        tb_log = None
        if tensorboard_log:
            tb_log = str(Path(tensorboard_log) / f"seed_{seed}")
        result = train_maskable_ppo(
            total_timesteps=total_timesteps,
            config=config,
            output_dir=seed_dir,
            seed=seed,
            max_steps=max_steps,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            eval_episodes=eval_episodes,
            ntp_policy=ntp_policy,
            reward_mode=reward_mode,
            device=device,
            tensorboard_log=tb_log,
            verbose=verbose,
            use_bc_pretrain=use_bc_pretrain,
            bc_max_states=bc_max_states,
            bc_epochs=bc_epochs,
            bc_lr=bc_lr,
        )
        runs.append(
            SeedRunResult(
                seed=seed,
                evaluation=result.evaluation,
                model_path=result.artifacts.model_path,
                metrics_path=result.artifacts.metrics_path,
                reused_existing=False,
            )
        )

    summary_path = output / "summary.csv"
    metrics_path = output / "summary.json"
    multi = MultiSeedResult(tuple(runs), summary_path, metrics_path)
    _write_multi_seed_artifacts(multi, config=config, output_dir=output)
    return multi


def train_maskable_ppo_all_configs(
    *,
    seeds: Iterable[int],
    output_dir: str | Path,
    total_timesteps: int,
    max_steps: int = 500,
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 4,
    gamma: float = 0.99,
    eval_episodes: int = 20,
    ntp_policy: str = "random",
    reward_mode: str = "terminal",
    device: str = "auto",
    tensorboard_log: str | None = None,
    verbose: int = 1,
    force: bool = False,
    use_bc_pretrain: bool = False,
    bc_max_states: int = 400,
    bc_epochs: int = 5,
    bc_lr: float = 1e-3,
) -> AllConfigsResult:
    """Train one MaskablePPO batch for each Mirror/Reversi configuration."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_results: list[ConfigRunResult] = []

    for label, config in ALL_CONFIGS:
        tb_log = None
        if tensorboard_log:
            tb_log = str(Path(tensorboard_log) / label)
        result = train_maskable_ppo_multi_seed(
            seeds=seeds,
            output_dir=output / label,
            total_timesteps=total_timesteps,
            config=config,
            max_steps=max_steps,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            eval_episodes=eval_episodes,
            ntp_policy=ntp_policy,
            reward_mode=reward_mode,
            device=device,
            tensorboard_log=tb_log,
            verbose=verbose,
            force=force,
            use_bc_pretrain=use_bc_pretrain,
            bc_max_states=bc_max_states,
            bc_epochs=bc_epochs,
            bc_lr=bc_lr,
        )
        config_results.append(ConfigRunResult(label=label, config=config, result=result))

    all_result = AllConfigsResult(
        configs=tuple(config_results),
        summary_path=output / "all_configs_summary.csv",
        metrics_path=output / "all_configs_summary.json",
    )
    _write_all_configs_artifacts(all_result, output_dir=output)
    return all_result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a MaskablePPO self-play baseline for Complete Yubisuma."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(TRAINING_PRESETS),
        default="quick",
        help="Training preset. Individual numeric flags override the preset.",
    )
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "maskable_ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        help="Comma-separated seed list. When set, trains one model per seed.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument(
        "--ntp-policy",
        choices=_ntp_policy_choices(),
        default="random",
        help="NTP reaction policy used for training/evaluation.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=REWARD_MODES,
        default="terminal",
        help="Training reward mode. Evaluation defaults to terminal scoring.",
    )
    parser.add_argument(
        "--eval-model",
        type=Path,
        help="Evaluate a saved MaskablePPO model instead of training.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        help=(
            "Evaluate every seed_*/maskable_ppo_complete.zip under a directory. "
            "Use with --all-configs for four-config training outputs."
        ),
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        help="Directory for evaluation_summary.csv/json. Defaults to --eval-dir.",
    )
    parser.add_argument(
        "--ntp-policies",
        help="Comma-separated NTP policies for --eval-dir. Defaults to --ntp-policy.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tensorboard-log")
    parser.add_argument("--mirror", action="store_true", help="Enable mirror rules.")
    parser.add_argument("--reversi", action="store_true", help="Enable reversi rules.")
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Train/evaluate all four Mirror/Reversi configurations.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if seed directories already contain model and metrics files.",
    )
    parser.add_argument(
        "--bc-pretrain",
        action="store_true",
        help="Pre-train actor with behavioral cloning from the exact solver before RL.",
    )
    parser.add_argument(
        "--bc-max-states",
        type=int,
        default=400,
        help="State cap for BC dataset generation (default: 400).",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=5,
        help="Number of BC pre-training epochs (default: 5).",
    )
    parser.add_argument(
        "--bc-lr",
        type=float,
        default=1e-3,
        help="Adam learning rate for BC pre-training (default: 1e-3).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = RulesConfig(enable_mirror=args.mirror, enable_reversi=args.reversi)
    preset = TRAINING_PRESETS[args.preset]
    total_timesteps = args.timesteps if args.timesteps is not None else preset.total_timesteps
    max_steps = args.max_steps if args.max_steps is not None else preset.max_steps
    learning_rate = (
        args.learning_rate if args.learning_rate is not None else preset.learning_rate
    )
    n_steps = args.n_steps if args.n_steps is not None else preset.n_steps
    batch_size = args.batch_size if args.batch_size is not None else preset.batch_size
    n_epochs = args.n_epochs if args.n_epochs is not None else preset.n_epochs
    gamma = args.gamma if args.gamma is not None else preset.gamma
    eval_episodes = (
        args.eval_episodes if args.eval_episodes is not None else preset.eval_episodes
    )

    seeds = _parse_seeds(args.seeds)
    if args.eval_dir:
        result = evaluate_saved_model_directory(
            args.eval_dir,
            output_dir=args.eval_output,
            config=config,
            all_configs=args.all_configs,
            ntp_policies=_parse_ntp_policies(args.ntp_policies, args.ntp_policy),
            episodes=eval_episodes,
            seed=args.seed,
            max_steps=max_steps,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        if result.summary_path:
            print(f"Wrote {result.summary_path}")
        if result.metrics_path:
            print(f"Wrote {result.metrics_path}")
        return 0

    if args.eval_model:
        result = evaluate_saved_model(
            args.eval_model,
            config=config,
            episodes=eval_episodes,
            seed=args.seed,
            max_steps=max_steps,
            ntp_policy=args.ntp_policy,
            reward_mode="terminal",
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.all_configs:
        result = train_maskable_ppo_all_configs(
            seeds=seeds or (args.seed,),
            output_dir=args.output_dir,
            total_timesteps=total_timesteps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            eval_episodes=eval_episodes,
            ntp_policy=args.ntp_policy,
            reward_mode=args.reward_mode,
            device=args.device,
            tensorboard_log=args.tensorboard_log,
            verbose=0 if args.quiet else 1,
            force=args.force,
            use_bc_pretrain=args.bc_pretrain,
            bc_max_states=args.bc_max_states,
            bc_epochs=args.bc_epochs,
            bc_lr=args.bc_lr,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        if result.summary_path:
            print(f"Wrote {result.summary_path}")
        if result.metrics_path:
            print(f"Wrote {result.metrics_path}")
        return 0

    if seeds is not None:
        result = train_maskable_ppo_multi_seed(
            seeds=seeds,
            output_dir=args.output_dir,
            total_timesteps=total_timesteps,
            config=config,
            max_steps=max_steps,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            eval_episodes=eval_episodes,
            ntp_policy=args.ntp_policy,
            reward_mode=args.reward_mode,
            device=args.device,
            tensorboard_log=args.tensorboard_log,
            verbose=0 if args.quiet else 1,
            force=args.force,
            use_bc_pretrain=args.bc_pretrain,
            bc_max_states=args.bc_max_states,
            bc_epochs=args.bc_epochs,
            bc_lr=args.bc_lr,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        if result.summary_path:
            print(f"Wrote {result.summary_path}")
        if result.metrics_path:
            print(f"Wrote {result.metrics_path}")
        return 0

    result = train_maskable_ppo(
        total_timesteps=total_timesteps,
        config=config,
        output_dir=args.output_dir,
        seed=args.seed,
        max_steps=max_steps,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        eval_episodes=eval_episodes,
        ntp_policy=args.ntp_policy,
        reward_mode=args.reward_mode,
        device=args.device,
        tensorboard_log=args.tensorboard_log,
        verbose=0 if args.quiet else 1,
        use_bc_pretrain=args.bc_pretrain,
        bc_max_states=args.bc_max_states,
        bc_epochs=args.bc_epochs,
        bc_lr=args.bc_lr,
    )

    print(json.dumps(result.evaluation.to_dict(), ensure_ascii=False, indent=2))
    if result.artifacts.model_path:
        print(f"Wrote {result.artifacts.model_path}")
    if result.artifacts.metrics_path:
        print(f"Wrote {result.artifacts.metrics_path}")
    return 0


def _write_artifacts(
    *,
    model,
    evaluation: EvaluationResult,
    output_dir: Path,
    config: RulesConfig,
    total_timesteps: int,
    seed: int,
    ntp_policy: str,
    reward_mode: str,
) -> TrainingArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "maskable_ppo_complete.zip"
    metrics_path = output_dir / "metrics.json"
    model.save(model_path)
    payload: dict[str, Any] = {
        "algorithm": "MaskablePPO",
        "total_timesteps": total_timesteps,
        "seed": seed,
        "ntp_policy": ntp_policy,
        "reward_mode": reward_mode,
        "config": {
            **_config_to_dict(config),
        },
        "evaluation": evaluation.to_dict(),
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return TrainingArtifacts(model_path=model_path, metrics_path=metrics_path)


def _load_existing_seed_run(
    seed_dir: Path,
    seed: int,
    ntp_policy: str = "random",
    reward_mode: str = "terminal",
) -> SeedRunResult | None:
    model_path = seed_dir / "maskable_ppo_complete.zip"
    metrics_path = seed_dir / "metrics.json"
    if not model_path.exists() or not metrics_path.exists():
        return None

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("ntp_policy", "random") != ntp_policy:
            return None
        if payload.get("reward_mode", "terminal") != reward_mode:
            return None
        evaluation_payload = payload["evaluation"]
        evaluation = EvaluationResult(
            episodes=int(evaluation_payload["episodes"]),
            wins=int(evaluation_payload["wins"]),
            losses=int(evaluation_payload["losses"]),
            truncations=int(evaluation_payload["truncations"]),
            average_reward=float(evaluation_payload["average_reward"]),
            average_steps=float(evaluation_payload["average_steps"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    return SeedRunResult(
        seed=seed,
        evaluation=evaluation,
        model_path=model_path,
        metrics_path=metrics_path,
        reused_existing=True,
    )


def _parse_seeds(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def _parse_ntp_policies(value: str | None, default_policy: str) -> tuple[str, ...]:
    policies = (
        tuple(item.strip() for item in value.split(",") if item.strip())
        if value
        else (default_policy,)
    )
    valid = set(_ntp_policy_choices())
    unknown = [policy for policy in policies if policy not in valid]
    if unknown:
        raise ValueError(f"unknown NTP policies: {', '.join(unknown)}")
    return policies


def _ntp_policy_choices() -> list[str]:
    return ["random", *MIXED_NTP_POLICIES, *sorted(NAMED_NTP_POLICIES)]


def _iter_seed_models(directory: Path) -> tuple[tuple[int, Path], ...]:
    if not directory.exists():
        return ()
    models: list[tuple[int, Path]] = []
    for child in sorted(directory.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or not child.name.startswith("seed_"):
            continue
        try:
            seed = int(child.name.removeprefix("seed_"))
        except ValueError:
            continue
        model_path = child / "maskable_ppo_complete.zip"
        if model_path.exists():
            models.append((seed, model_path))
    return tuple(models)


def _aggregate_eval_rows(rows: tuple[EvaluationSuiteRow, ...]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        key = f"{row.config_label}:{row.ntp_policy}"
        buckets.setdefault(key, []).append(row.evaluation.average_reward)
    return {
        key: float(np.mean(values)) if values else 0.0
        for key, values in sorted(buckets.items())
    }


def _write_multi_seed_artifacts(
    result: MultiSeedResult,
    *,
    config: RulesConfig,
    output_dir: Path,
) -> None:
    summary_path = output_dir / "summary.csv"
    fieldnames = [
        "seed",
        "episodes",
        "wins",
        "losses",
        "truncations",
        "average_reward",
        "average_steps",
        "model_path",
        "metrics_path",
        "reused_existing",
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in result.runs:
            writer.writerow(run.to_dict())

    payload = result.to_dict()
    payload["config"] = _config_to_dict(config)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_all_configs_artifacts(
    result: AllConfigsResult,
    *,
    output_dir: Path,
) -> None:
    summary_path = output_dir / "all_configs_summary.csv"
    fieldnames = [
        "config_label",
        "enable_mirror",
        "enable_reversi",
        "seed",
        "episodes",
        "wins",
        "losses",
        "truncations",
        "average_reward",
        "average_steps",
        "model_path",
        "metrics_path",
        "reused_existing",
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for config_run in result.configs:
            config_dict = _config_to_dict(config_run.config)
            for seed_run in config_run.result.runs:
                row = seed_run.to_dict()
                row["config_label"] = config_run.label
                row["enable_mirror"] = str(config_dict["enable_mirror"])
                row["enable_reversi"] = str(config_dict["enable_reversi"])
                writer.writerow(row)

    (output_dir / "all_configs_summary.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_evaluation_suite_artifacts(
    result: EvaluationSuiteResult,
    *,
    output_dir: Path,
) -> None:
    summary_path = output_dir / "evaluation_summary.csv"
    fieldnames = [
        "config_label",
        "enable_mirror",
        "enable_reversi",
        "seed",
        "ntp_policy",
        "episodes",
        "wins",
        "losses",
        "truncations",
        "average_reward",
        "average_steps",
        "model_path",
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.rows:
            writer.writerow(row.to_dict())

    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _config_to_dict(config: RulesConfig) -> dict[str, bool]:
    return {
        "enable_mirror": config.enable_mirror,
        "enable_reversi": config.enable_reversi,
    }


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Partial seed outputs, if any, remain on disk.")
        raise SystemExit(130)
