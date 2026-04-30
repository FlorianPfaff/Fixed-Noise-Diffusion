from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def save_figure(fig: Any, output: Path, *, dpi: int = 180) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
