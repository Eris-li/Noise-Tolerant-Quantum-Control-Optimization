# 项目地图

这份文档回答两个问题：

1. 仓库里每个目录主要负责什么。
2. 当前哪些文件是主线，哪些文件是冻结参考，哪些只是阶段性结果。

## 目录职责

### `src/neutral_yb/config`

- [species.py](../src/neutral_yb/config/species.py)
  物种级配置，目前主要是 `^171Yb` 的语义化配置壳。

### `src/neutral_yb/models`

- [global_cz_4d.py](../src/neutral_yb/models/global_cz_4d.py)
  `v1` 冻结参考的理想 4 维 `global CZ` 模型。
- [finite_blockade_cz_5d.py](../src/neutral_yb/models/finite_blockade_cz_5d.py)
  `v2` 闭系统修正模型，保留有限 blockade 和 `|rr>`。
- [two_photon_cz_9d.py](../src/neutral_yb/models/two_photon_cz_9d.py)
  `v3` 双光子闭系统模型，显式保留中间态 `|e>`。
- [two_photon_cz_open_10d.py](../src/neutral_yb/models/two_photon_cz_open_10d.py)
  `v4` 双光子开放系统模型，新增 `|loss>` sink 和 Lindblad 噪声。

### `src/neutral_yb/optimization`

- [global_phase_grape.py](../src/neutral_yb/optimization/global_phase_grape.py)
  `v1`/早期版本的闭系统相位 GRAPE。
- [amplitude_phase_grape.py](../src/neutral_yb/optimization/amplitude_phase_grape.py)
  `v3` 当前主线，闭系统 lower-leg 振幅加单相位优化器。
- [open_system_grape.py](../src/neutral_yb/optimization/open_system_grape.py)
  `v4` 当前主线，直接优化 probe-based fidelity 的开放系统 Liouvillian GRAPE。
- [linear_control_grape.py](../src/neutral_yb/optimization/linear_control_grape.py)
  历史实验文件，保留作对照，不是当前主线。

### `experiments`

- [freeze_v1_global_cz_reference.py](../experiments/freeze_v1_global_cz_reference.py)
  `v1` 冻结参考实验。
- [two_stage_scan_closed_system_cz_v2.py](../experiments/two_stage_scan_closed_system_cz_v2.py)
  `v2` 的两阶段时间扫描。
- [coarse_scan_two_photon_cz_v3.py](../experiments/coarse_scan_two_photon_cz_v3.py)
  `v3` 的 coarse scan 主实验。
- [local_scan_two_photon_cz_v3_7p5_8p5.py](../experiments/local_scan_two_photon_cz_v3_7p5_8p5.py)
  `v3` 的高 restart 局部扫描。
- [run_two_photon_cz_v4_open_system_smoke.py](../experiments/run_two_photon_cz_v4_open_system_smoke.py)
  `v4` 的单点开放系统 smoke run。
- [coarse_scan_two_photon_cz_v4_open_system.py](../experiments/coarse_scan_two_photon_cz_v4_open_system.py)
  `v4` 的开放系统粗扫描主实验。
- [benchmark_v4_open_system_vs_v3_closed.py](../experiments/benchmark_v4_open_system_vs_v3_closed.py)
  `v4` 和 `v3` 的本地资源对比。

### `scripts`

- [plot_freeze_v1_global_cz.py](../scripts/plot_freeze_v1_global_cz.py)
  `v1` 出图。
- [plot_closed_system_cz_v2_two_stage.py](../scripts/plot_closed_system_cz_v2_two_stage.py)
  `v2` 出图。
- [plot_two_photon_cz_v3.py](../scripts/plot_two_photon_cz_v3.py)
  `v3` 出图。
- [create_venv.ps1](../scripts/create_venv.ps1)
  Windows 本地环境创建脚本。
- [run_python.ps1](../scripts/run_python.ps1)
  Windows 辅助运行脚本。

### `artifacts`

这里只放已经跑出来的结果，不放逻辑。按版本理解：

- `freeze_v1_*`
  冻结参考结果
- `closed_system_cz_v2_*`
  `v2` 结果
- `two_photon_cz_v3_*`
  `v3` 结果
- `two_photon_cz_v4_*`
  `v4` 结果
- `benchmark_v4_open_system_vs_v3_closed.json`
  闭系统和开放系统耗时比较

### `tests`

- `test_global_cz_4d.py`
  `v1` 理想参考模型和优化器的基本测试
- `test_finite_blockade_cz_5d.py`
  `v2` 闭系统修正版测试
- `test_two_photon_cz_9d.py`
  `v3` 双光子闭系统模型测试
- `test_amplitude_phase_grape.py`
  `v3` 振幅加单相位优化器测试
- `test_two_photon_cz_open_10d.py`
  `v4` 开放系统模型测试
- `test_open_system_grape.py`
  `v4` 开放系统 GRAPE 的 smoke 测试

`__pycache__` 和中间缓存不属于项目逻辑，不需要在迁移时保留。

## 当前主线

如果只看今天仍在推进的主线，应优先关注：

- `v3` 闭系统双光子：
  - [two_photon_cz_9d.py](../src/neutral_yb/models/two_photon_cz_9d.py)
  - [amplitude_phase_grape.py](../src/neutral_yb/optimization/amplitude_phase_grape.py)
- `v4` 开放系统双光子：
  - [two_photon_cz_open_10d.py](../src/neutral_yb/models/two_photon_cz_open_10d.py)
  - [open_system_grape.py](../src/neutral_yb/optimization/open_system_grape.py)

## 迁移到 WSL 后先做什么

建议顺序是：

1. 创建 `.venv` 并安装依赖。
2. 把仓库本身装进虚拟环境：`./.venv/bin/python -m pip install -e .`
3. 跑测试：`./.venv/bin/python -m unittest discover -s tests -v`
4. 读 [docs/version-history.md](version-history.md)
5. 读 [docs/references.md](references.md)
6. 再决定从 `v3` 还是 `v4` 继续开发
