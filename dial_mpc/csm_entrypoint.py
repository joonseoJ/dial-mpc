"""Console wrappers that configure native logging before importing JAX."""

from __future__ import annotations

import os


# XLA sometimes rejects its hinted GEMM candidates and safely retries the full
# search space.  The native runtime logs that fallback at WARNING on every new
# compilation.  Hide native INFO/WARNING messages while preserving ERROR and
# FATAL output.  Users can opt back in by exporting either variable themselves.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("MUJOCO_GL", "egl")


def train() -> None:
    from csm.energy_cli import main

    main()


def benchmark() -> None:
    from csm.benchmark_experts import main

    main()


def evaluate() -> None:
    from csm.eval import main

    main()
