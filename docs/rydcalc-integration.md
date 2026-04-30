# rydcalc 初始化记录

`rydcalc/` 是从 `https://github.com/ThompsonLabPrinceton/rydcalc.git` 导入的 Git submodule，当前 pin 到 `d17b981 Fixed 3P2 MQDT perturbers in 171Yb models`。

## 作用

`rydcalc` 是 Thompson Lab 的 Rydberg 原子计算器，主要用于 MQDT 框架下的 Rydberg 态计算。它包含 `^174Yb` 和 `^171Yb` 模型，也包含 Alkali 原子模型用于对照。

对本项目最直接有用的能力包括：

- `Ytterbium171` / `Ytterbium174`：Yb Rydberg 态能级和 MQDT 模型。
- `single_basis` / `pair_basis`：单原子和双原子基组。
- `environment`：电场、磁场、温度、光强等环境参数。
- 径向波函数和 multipole matrix element，用于估计跃迁强度、Stark shift、pair interaction 和 blockade 相关参数。

## 初始化状态

已在本地完成以下验证。注意：Python 3.12 / NumPy 2 环境的推荐路径是为当前解释器编译 C Numerov 扩展，并通过项目适配层导入。

- 当前主项目 `.venv` 为 Python 3.12.3；本机缺少 `/usr/include/python3.12/Python.h` 时无法直接编译 C 扩展，需要先安装匹配的 Python 开发头文件。
- 上游 `tests/unit_tests.py` 的 Hydrogen norm 和 circular multipole 测试通过。
- `Ytterbium171(use_db=False)` 可实例化，并能计算示例态 `|171Yb:60.18,L=0,F=0.5,0.5>`。

当前验证命令：

```bash
./.venv/bin/python -m pip install -e '.[rydcalc]'
./.venv/bin/python scripts/build_rydcalc_extension.py
./.venv/bin/python -c "from neutral_yb.external.rydcalc_adapter import build_yb171_atom; yb = build_yb171_atom(use_db=False); print(yb.get_state((60,0,0.5,0.5)))"
PYTHONPATH=.. MPLCONFIGDIR=/tmp/matplotlib-rydcalc ../.venv/bin/python -m unittest tests.unit_tests -v
```

## 依赖

主项目 `pyproject.toml` 增加了可选依赖组：

```bash
./.venv/bin/python -m pip install -e '.[rydcalc]'
```

这个依赖组覆盖 `rydcalc` 额外需要的 `sympy`、`mpmath`、`dill`、`tqdm` 和编译脚本所需的 `setuptools`。

## 本地兼容修复

为了让上游代码能在本项目 Python 3.12 / NumPy 2 环境里运行，项目侧新增了 `neutral_yb.external.rydcalc_adapter`：

- 自动把 `rydcalc/` submodule 加入 import path。
- 检查当前 Python 解释器是否有对应的 `arc_c_extensions` C 扩展。
- 在导入 `rydcalc` 前补上 NumPy 2 兼容别名 `np.product = np.prod`，避免改动 submodule。

验证早期也做过三处上游小修复。由于 `rydcalc/` 按 submodule 管理，默认保留上游工作树；这些修复记录在 `patches/rydcalc-python312-numpy2.patch`，仅作为 no-extension fallback 或 upstream patch 参考：

```bash
git -C rydcalc apply ../patches/rydcalc-python312-numpy2.patch
```

- `rydcalc/setupc.py`：从已移除的 `distutils` / `numpy.distutils` 改为 `setuptools` + `numpy.get_include()`。
- `rydcalc/MQDTclass.py`：将 NumPy 2 中已移除的 `np.product` 改为 `np.prod`。
- `rydcalc/alkali.py`：请求 C Numerov 扩展但扩展不存在时，自动回退到纯 Python Numerov，避免 Python 开发头文件缺失时无法导入。

如需撤销该兼容 patch 并恢复干净 submodule：

```bash
git -C rydcalc apply -R ../patches/rydcalc-python312-numpy2.patch
```

## Submodule 操作

首次 clone 主项目后拉取 `rydcalc`：

```bash
git submodule update --init --recursive
```

更新到主项目记录的版本：

```bash
git submodule update --recursive
```

查看当前 pin：

```bash
git submodule status --recursive
```

## C 扩展说明

`rydcalc` 带有 ARC Numerov C 扩展 `arc_c_extensions.c`。当前推荐为项目 Python 3.12 解释器直接编译该扩展：

如果需要高性能径向波函数计算，先安装 Python 3.12 headers，再运行：

```bash
./.venv/bin/python scripts/build_rydcalc_extension.py
```

生成的 `.so` 和 `build/` 已被上游 `.gitignore` 忽略，不应提交。当前 Dockerfile 会安装编译工具、安装 `.[rydcalc]`，并在 build 阶段运行这个脚本。

## 建议集成方式

短期内把 `rydcalc/` 保持为外部上游代码，不直接混入 `src/neutral_yb/`。建议下一步在本项目中新增薄适配层，例如：

- `src/neutral_yb/config/rydcalc_adapter.py`
- 只暴露本项目需要的函数，例如 `get_yb171_state_energy_hz()`、`estimate_blockade_shift_hz()` 或 pair potential 扫描入口。
- 适配层负责 `PYTHONPATH` / import 失败提示、单位转换和缓存策略。
- 数值结果先落到新的版本化 `artifacts/rydcalc_*` 子目录，避免覆盖现有扫描产物。
