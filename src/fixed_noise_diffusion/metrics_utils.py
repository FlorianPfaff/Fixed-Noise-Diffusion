from __future__ import annotations


def _positive_count(name: str, value: int) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be at least 1")
    return parsed


def effective_kid_subset_size(
    kid_subset_size: int,
    sample_count: int,
    real_count: int | None,
) -> int:
    """Return a KID subset size that is valid for both metric populations."""
    sample_count = _positive_count("sample_count", sample_count)
    effective_real_count = sample_count if real_count is None else _positive_count("real_count", real_count)
    return min(int(kid_subset_size), sample_count, effective_real_count)
