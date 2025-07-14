from strategies.combiners import BaseCombiner, MeanCombiner, MedianCombiner
from typing import Any


def make_combiner(strategy_cfg: Any) -> BaseCombiner:
    """
    Factory function to create a combiner based on the strategy configuration.

    Expects `strategy_cfg.strategy.kind` to be one of:
    - "mean"
    - "median"

    Raises:
        ValueError: if an unsupported combiner kind is specified.
    """
    kind = getattr(strategy_cfg.strategy, "kind", None)

    if kind == "mean":
        return MeanCombiner()
    elif kind == "median":
        return MedianCombiner()
    else:
        raise ValueError(f"Unknown combiner kind: {kind}")