from __future__ import annotations

from pathlib import Path


def artifact_root(root: Path) -> Path:
    return root / "artifacts"


def v1_artifacts_dir(root: Path) -> Path:
    return artifact_root(root) / "v1"


def v2_artifacts_dir(root: Path) -> Path:
    return artifact_root(root) / "v2"


def v3_artifacts_dir(root: Path) -> Path:
    return artifact_root(root) / "v3"


def v4_artifacts_dir(root: Path) -> Path:
    return artifact_root(root) / "v4"


def v4_coarse_10mhz_dir(root: Path) -> Path:
    return v4_artifacts_dir(root) / "coarse_0_300ns_10mhz"


def v4_fine_90_150ns_10mhz_dir(root: Path) -> Path:
    return v4_artifacts_dir(root) / "fine_90_150ns_0p5ns_10mhz"


def v4_single_300ns_10mhz_dir(root: Path) -> Path:
    return v4_artifacts_dir(root) / "single_300ns_10mhz"


def v4_validation_dir(root: Path) -> Path:
    return v4_artifacts_dir(root) / "validation"


def v5_artifacts_dir(root: Path) -> Path:
    return artifact_root(root) / "v5"


def v5_profile_dir(root: Path, profile: str) -> Path:
    return v5_artifacts_dir(root) / profile


def v5_coarse_10mhz_dir(root: Path, profile: str) -> Path:
    return v5_profile_dir(root, profile) / "coarse_0_300ns_10mhz"


def ma2023_time_optimal_2q_dir(root: Path) -> Path:
    return artifact_root(root) / "ma2023_time_optimal_2q"


def ensure_artifact_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
