"""Behavioral cloning warm-start from the Complete exact solver.

Generates (observation, optimal_tp_distribution) training pairs using
value iteration over the reachable state space, then pre-trains a
MaskablePPO actor network with cross-entropy loss against those target
distributions.

Typical usage
-------------
>>> from complete_rl.bc_pretrain import generate_bc_dataset, bc_pretrain
>>> dataset = generate_bc_dataset(max_states=400)
>>> losses = bc_pretrain(model, dataset, n_epochs=5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from complete_solver import RulesConfig
from complete_solver.actions import legal_ntp_actions, legal_tp_actions
from complete_solver.constants import ALL
from complete_solver.matrix_game import solve_zero_sum_matrix
from complete_solver.state_space import enumerate_reachable_states, value_iteration
from complete_solver.transition import transition
from complete_rl.env import build_action_mask, build_canonical_tp_actions
from complete_rl.obs import encode_state

if TYPE_CHECKING:
    pass


# Type alias: each entry is (obs_float32 shape=(OBS_SIZE,), probs_float32 shape=(n_actions,))
BCDataset = list[tuple[np.ndarray, np.ndarray]]


def generate_bc_dataset(
    config: RulesConfig = RulesConfig(),
    max_states: int = 400,
    gamma: float = 0.999,
    vi_epsilon: float = 1e-4,
    vi_max_iter: int = 500,
) -> BCDataset:
    """Generate behavioral cloning pairs from the exact solver.

    Enumerates reachable states, runs discounted value iteration to obtain
    V(s), then solves the induced matrix game at each state to extract the
    optimal TP mixed strategy.  Returns (obs, probs) pairs where *probs*
    is a probability vector over the canonical action space.

    Parameters
    ----------
    config:
        Which Mirror/Reversi rule variant to use.
    max_states:
        Safety cap on state enumeration.  Larger values give more training
        data but take longer.
    gamma:
        Discount factor for value iteration.
    vi_epsilon:
        Convergence threshold for value iteration.
    vi_max_iter:
        Maximum number of value-iteration sweeps.
    """
    states = enumerate_reachable_states(config=config, max_states=max_states)
    vi = value_iteration(
        states,
        config=config,
        gamma=gamma,
        epsilon=vi_epsilon,
        max_iterations=vi_max_iter,
    )
    V = vi.values

    canonical = build_canonical_tp_actions(config)

    # Pre-compute lookup maps for O(1) canonical matching.
    canonical_exact: dict = {}   # non-ALL actions → canonical idx
    all_thumb_idx: dict[int, int] = {}  # thumb → canonical idx for ALL

    for k, canon in enumerate(canonical):
        if canon.skill == ALL:
            all_thumb_idx[canon.thumb] = k
        else:
            canonical_exact[canon] = k

    dataset: BCDataset = []

    for state in states:
        tp_acts = legal_tp_actions(state, config)
        ntp_acts = legal_ntp_actions(state, config)
        if not tp_acts or not ntp_acts:
            continue

        # Build payoff matrix with V as leaf evaluator.
        matrix = np.zeros((len(tp_acts), len(ntp_acts)), dtype=float)
        for i, tp_a in enumerate(tp_acts):
            for j, ntp_a in enumerate(ntp_acts):
                res = transition(state, tp_a, ntp_a, config)
                if res.terminal_reward is not None:
                    matrix[i, j] = float(res.terminal_reward)
                else:
                    assert res.next_state is not None
                    v_next = V.get(res.next_state, 0.0)
                    sign = 1.0 if res.same_turn_player else -1.0
                    matrix[i, j] = gamma * sign * v_next

        solution = solve_zero_sum_matrix(matrix)
        tp_probs = solution.row_policy  # shape (len(tp_acts),)

        # Map solver probs onto the canonical action space.
        probs = np.zeros(len(canonical), dtype=np.float32)
        for act, prob in zip(tp_acts, tp_probs):
            if act.skill == ALL:
                k = all_thumb_idx.get(act.thumb)
                if k is not None:
                    probs[k] += float(prob)
            else:
                k = canonical_exact.get(act)
                if k is not None:
                    probs[k] += float(prob)

        total = float(probs.sum())
        if total <= 1e-10:
            # Fallback: uniform over legal actions.
            mask = build_action_mask(canonical, state, config)
            n_legal = int(mask.sum())
            if n_legal > 0:
                probs[mask] = 1.0 / n_legal
        else:
            probs /= total

        obs = encode_state(state)
        dataset.append((obs, probs))

    return dataset


def bc_pretrain(
    model,
    dataset: BCDataset,
    *,
    n_epochs: int = 5,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    seed: int | None = None,
    verbose: bool = False,
) -> list[float]:
    """Pre-train *model*'s actor network with behavioral cloning.

    Applies cross-entropy loss between the model's masked action logits and
    the solver-derived target distributions.  The critic and features
    extractor are updated jointly.

    Returns mean cross-entropy loss per epoch (lower is better).
    Modifies *model* in place.

    Parameters
    ----------
    model:
        An untrained or partially trained MaskablePPO instance.
    dataset:
        Output of :func:`generate_bc_dataset`.
    n_epochs:
        Number of full passes over the dataset.
    learning_rate:
        Adam learning rate for the pre-training phase.
    batch_size:
        Mini-batch size.
    seed:
        Optional RNG seed for deterministic shuffling.
    verbose:
        Print epoch loss if True.
    """
    if not dataset:
        return []

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "Behavioral cloning requires PyTorch. Install requirements.txt."
        ) from exc

    device = next(model.policy.parameters()).device
    rng = np.random.default_rng(seed)

    obs_arr = np.stack([x[0] for x in dataset], axis=0).astype(np.float32)
    probs_arr = np.stack([x[1] for x in dataset], axis=0).astype(np.float32)
    n = len(dataset)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    epoch_losses: list[float] = []
    indices = np.arange(n)

    for epoch in range(n_epochs):
        rng.shuffle(indices)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            obs_t = torch.as_tensor(
                obs_arr[batch_idx], dtype=torch.float32, device=device
            )
            target_t = torch.as_tensor(
                probs_arr[batch_idx], dtype=torch.float32, device=device
            )

            # Forward pass: features → actor latent → action logits.
            features = model.policy.features_extractor(obs_t)
            latent_pi, _ = model.policy.mlp_extractor(features)
            logits = model.policy.action_net(latent_pi)

            # Mask illegal actions (zero probability in target → illegal).
            legal_mask = target_t > 0
            logits_masked = logits.masked_fill(~legal_mask, float("-inf"))
            log_probs = F.log_softmax(logits_masked, dim=-1)

            # Cross-entropy: −Σ p_target · log p_model
            # Use torch.where to avoid NaN from 0 * (-inf) at masked positions.
            safe_log = torch.where(legal_mask, log_probs, torch.zeros_like(log_probs))
            loss = -(target_t * safe_log).sum(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        epoch_losses.append(avg)
        if verbose:
            print(f"BC epoch {epoch + 1}/{n_epochs}  loss={avg:.4f}")

    return epoch_losses
