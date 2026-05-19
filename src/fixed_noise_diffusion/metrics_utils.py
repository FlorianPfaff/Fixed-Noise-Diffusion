from __future__ import annotations


def effective_kid_subset_size(
    kid_subset_size: int,
    sample_count: int,
    real_count: int | None,
) -> int:
    """Return a KID subset size that is valid for both metric populations."""
    effective_real_count = real_count or sample_count
    return min(kid_subset_size, sample_count, effective_real_count)
