from __future__ import annotations

import importlib
import math
import sys
import sysconfig
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import scipy.special as sp_special


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def rydcalc_submodule_path() -> Path:
    return _project_root() / "rydcalc"


def _rydcalc_package_path() -> Path:
    return rydcalc_submodule_path() / "rydcalc"


def _ensure_import_path() -> None:
    submodule = str(rydcalc_submodule_path())
    if submodule not in sys.path:
        sys.path.insert(0, submodule)


def _ensure_numpy2_compat() -> None:
    if not hasattr(np, "product"):
        np.product = np.prod  # type: ignore[attr-defined]
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]
    if not hasattr(np, "math"):
        np.math = math  # type: ignore[attr-defined]


def _ensure_scipy_compat() -> None:
    if not hasattr(sp_special, "sph_harm") and hasattr(sp_special, "sph_harm_y"):
        def _sph_harm(m: int, n: int, theta: float, phi: float) -> complex:
            return sp_special.sph_harm_y(n, m, phi, theta)

        sp_special.sph_harm = _sph_harm  # type: ignore[attr-defined]


def has_arc_c_extension() -> bool:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        return False
    return (_rydcalc_package_path() / f"arc_c_extensions{suffix}").exists()


def load_rydcalc(*, require_c_extension: bool = True) -> ModuleType:
    if not rydcalc_submodule_path().exists():
        raise ImportError(
            "rydcalc submodule is missing. Run: git submodule update --init --recursive"
        )

    if require_c_extension and not has_arc_c_extension():
        suffix = sysconfig.get_config_var("EXT_SUFFIX") or "<extension suffix>"
        expected = _rydcalc_package_path() / f"arc_c_extensions{suffix}"
        raise ImportError(
            "rydcalc ARC C extension is not built for the active Python interpreter. "
            f"Expected: {expected}. Run: ./.venv/bin/python scripts/build_rydcalc_extension.py"
        )

    _ensure_numpy2_compat()
    _ensure_scipy_compat()
    _ensure_import_path()
    return importlib.import_module("rydcalc")


def build_yb171_atom(*, use_db: bool = False, require_c_extension: bool = True, **kwargs: Any) -> Any:
    rydcalc = load_rydcalc(require_c_extension=require_c_extension)
    return rydcalc.Ytterbium171(use_db=use_db, **kwargs)
