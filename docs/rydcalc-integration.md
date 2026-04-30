# rydcalc 初始化记录

`rydcalc/` 是从 `https://github.com/ThompsonLabPrinceton/rydcalc.git` 导入的独立上游仓库，当前 HEAD 为 `d17b981 Fixed 3P2 MQDT perturbers in 171Yb models`。

## 作用

`rydcalc` 是 Thompson Lab 的 Rydberg 原子计算器，主要用于 MQDT 框架下的 Rydberg 态计算。它包含 `^174Yb` 和 `^171Yb` 模型，也包含 Alkali 原子模型用于对照。

对本项目最直接有用的能力包括：

- `Ytterbium171` / `Ytterbium174`：Yb Rydberg 态能级和 MQDT 模型。
- `single_basis` / `pair_basis`：单原子和双原子基组。
- `environment`：电场、磁场、温度、光强等环境参数。
- 径向波函数和 multipole matrix element，用于估计跃迁强度、Stark shift、pair interaction 和 blockade 相关参数。

## 初始化状态

已在本地完成以下验证：

- 当前主项目 `.venv` 为 Python 3.12.3，可通过 `PYTHONPATH=rydcalc` 导入 `rydcalc`。
- 上游 `tests/unit_tests.py` 的 Hydrogen norm 和 circular multipole 测试通过。
- `Ytterbium171(use_db=False)` 可实例化，并能计算示例态 `|171Yb:60.18,L=0,F=0.5,0.5>`。

当前验证命令：

```bash
PYTHONPATH=rydcalc MPLCONFIGDIR=/tmp/matplotlib-rydcalc ./.venv/bin/python -c "import rydcalc; yb=rydcalc.Ytterbium171(use_db=False); print(yb.get_state((60,0,0.5,0.5)))"
PYTHONPATH=.. MPLCONFIGDIR=/tmp/matplotlib-rydcalc ../.venv/bin/python -m unittest tests.unit_tests -v
```

## 依赖

主项目 `pyproject.toml` 增加了可选依赖组：

```bash
./.venv/bin/python -m pip install -e '.[rydcalc]'
```

这个依赖组覆盖 `rydcalc` 额外需要的 `sympy`、`mpmath`、`dill`、`tqdm` 和编译脚本所需的 `setuptools`。

## 本地兼容修复

为了让上游代码能在本项目 Python 3.12 / NumPy 2 环境里运行，已对 `rydcalc` 做了三处小修复：

- `rydcalc/setupc.py`：从已移除的 `distutils` / `numpy.distutils` 改为 `setuptools` + `numpy.get_include()`。
- `rydcalc/MQDTclass.py`：将 NumPy 2 中已移除的 `np.product` 改为 `np.prod`。
- `rydcalc/alkali.py`：请求 C Numerov 扩展但扩展不存在时，自动回退到纯 Python Numerov，避免 Python 开发头文件缺失时无法导入。

## C 扩展说明

`rydcalc` 带有 ARC Numerov C 扩展 `arc_c_extensions.c`。当前主项目 `.venv` 缺少 Python 3.12 开发头文件，所以不能在 `.venv` 里编译该扩展；代码会回退到纯 Python 路径。

如果需要高性能径向波函数计算，先安装 Python 3.12 headers，再运行：

```bash
cd rydcalc/rydcalc
../../.venv/bin/python setupc.py build_ext --inplace
```

生成的 `.so` 和 `build/` 已被上游 `.gitignore` 忽略，不应提交。

## 建议集成方式

短期内把 `rydcalc/` 保持为外部上游代码，不直接混入 `src/neutral_yb/`。建议下一步在本项目中新增薄适配层，例如：

- `src/neutral_yb/config/rydcalc_adapter.py`
- 只暴露本项目需要的函数，例如 `get_yb171_state_energy_hz()`、`estimate_blockade_shift_hz()` 或 pair potential 扫描入口。
- 适配层负责 `PYTHONPATH` / import 失败提示、单位转换和缓存策略。
- 数值结果先落到新的版本化 `artifacts/rydcalc_*` 子目录，避免覆盖现有扫描产物。

