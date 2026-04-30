#!/usr/bin/env python3
from __future__ import annotations

import sys
import sysconfig
from pathlib import Path
import os

import numpy as np
from setuptools import Extension, setup


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    package_dir = project_root() / "rydcalc" / "rydcalc"
    source = package_dir / "arc_c_extensions.c"
    include_dir = Path(os.environ.get("PYTHON_INCLUDE_DIR", sysconfig.get_paths()["include"]))
    include_dirs = [np.get_include(), str(include_dir)]
    if include_dir.parent not in (include_dir, Path(".")):
        include_dirs.append(str(include_dir.parent))
    python_h = include_dir / "Python.h"

    if not source.exists():
        print(f"Missing rydcalc C source: {source}", file=sys.stderr)
        return 1

    if not python_h.exists():
        print(
            "Missing Python development header for this interpreter: "
            f"{python_h}\nInstall the matching Python development headers "
            "(for example python3.12-dev on Ubuntu) and rerun this script.",
            file=sys.stderr,
        )
        return 1

    old_cwd = Path.cwd()
    try:
        os.chdir(package_dir)
        setup(
            script_args=["build_ext", "--inplace"],
            ext_modules=[
                Extension(
                    "arc_c_extensions",
                    ["arc_c_extensions.c"],
                    extra_compile_args=["-Wall", "-O3"],
                    include_dirs=include_dirs,
                )
            ],
        )
    finally:
        os.chdir(old_cwd)

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    output = package_dir / f"arc_c_extensions{suffix}"
    if not output.exists():
        print(f"Build completed but expected extension was not found: {output}", file=sys.stderr)
        return 1

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
