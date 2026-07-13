"""New-generation Complete AI stack (AI_MASTER_PLAN_V2, N3+).

Search-centric architecture: depth-limited LP-backup search (complete_solver)
+ a learned value network for leaf evaluation. This package holds the value
network side: feature extraction, dataset generation on the compiled engine,
and training.
"""

from .features import FEATURE_SIZE, features_from_lanes, features_from_state  # noqa: F401
