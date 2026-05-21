from __future__ import annotations

import argparse
import json
import math
from typing import Any


def expected_fixed_pool_exposure(pool_size: int, draws: int) -> dict[str, float | int]:
    """Return iid-with-replacement exposure diagnostics for a finite noise pool."""
    pool_size = int(pool_size)
    draws = int(draws)
    if pool_size < 1:
        raise ValueError("pool_size must be at least 1")
    if draws < 0:
        raise ValueError("draws must be non-negative")

    if draws == 0:
        expected_unique = 0.0
    elif pool_size == 1:
        expected_unique = 1.0
    else:
        expected_unique = pool_size * -math.expm1(draws * math.log1p(-1.0 / pool_size))
    expected_unique_fraction = expected_unique / pool_size
    duplicate_draw_fraction = 0.0 if draws == 0 else max(0.0, 1.0 - expected_unique / draws)
    return {
        "pool_size": pool_size,
        "draws": draws,
        "draws_per_pool_entry": draws / pool_size,
        "expected_unique_pool_entries": expected_unique,
        "expected_unique_pool_fraction": expected_unique_fraction,
        "expected_duplicate_draw_fraction": duplicate_draw_fraction,
    }


def _rounded(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in payload.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate expected fixed-noise-pool exposure under iid sampling with replacement."
    )
    parser.add_argument("--pool-size", type=int, required=True)
    parser.add_argument("--draws", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(_rounded(expected_fixed_pool_exposure(args.pool_size, args.draws)), sort_keys=True))


if __name__ == "__main__":
    main()