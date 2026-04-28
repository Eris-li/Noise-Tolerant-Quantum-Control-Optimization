# Noise-Tolerant-Quantum-Control-Optimization

面向中性原子 `^171Yb` 量子计算的多比特量子门控制优化项目。

当前项目只关注两比特及以上的门，不实现单比特门。主线目标是围绕 `CZ`、后续 `CNOT` 和三比特门，逐步从理想闭系统模型推进到显式开放系统、含噪声和含损耗的控制优化。

## Nature 2023 time-optimal 两比特门复现

新增独立复现板块：[docs/ma2023-time-optimal-2q.md](docs/ma2023-time-optimal-2q.md)。

目标是从 Ma et al., Nature 622, 279-284 (2023) 的 Fig. 3 出发，先复现其 time-optimal two-qubit gate 所依赖的理想 global CZ / Jandura-Pupillo pulse，再逐步加入有限 blockade、Rydberg decay 和 erasure/loss 误差。当前起步入口：

```bash
./.venv/bin/python experiments/reproduce_ma2023_time_optimal_2q_gate.py --duration 7.612 --num-tslots 99 --max-iter 300 --num-restarts 4 --show-progress
```

结果写入 `artifacts/ma2023_time_optimal_2q/`，暂时不并入 `v4/v5` 主线。

## 当前状态

仓库里目前并行保留了 4 条有明确定位的版本线：

- `v1`
  理想、冻结的 `global CZ` 参考实验。用途是做论文对照和回归基准。
- `v2`
  闭系统、有效 5 维模型，加入有限 blockade、detuning 和振幅误差。
- `v3`
  闭系统、显式中间态的双光子 9 维模型，当前主线是 lower-leg 振幅加单相位控制。
- `v4`
  开放系统、`^171Yb` `clock -> Rydberg` 有效完整门模型。当前显式优化的是中间 `UV` 段，但固定前后 `clock shelving / unshelving` 单个 `\pi` 脉冲已经被并入总门传播，默认参数按 `PRX Quantum 6, 020334 (2025)` 与 `Phys. Rev. X 15, 011009 (2025)` 口径设定，使用仓库内自写的逐 slice 传播和 GRAPE，并以 QuTiP 负责算符构造与独立验证。

如果你刚接手这个仓库，建议先看：

1. [docs/project-map.md](docs/project-map.md)
2. [docs/version-history.md](docs/version-history.md)
3. [docs/references.md](docs/references.md)

## 平台说明

- `venv` 不是跨操作系统可复用的。
- 建议 `WSL/Linux` 使用 `.venv`。
- 建议 `Windows PowerShell` 使用 `.venv.win`。
- 如果你只想要一套跨平台运行方式，直接用 Docker。

## 快速开始

### Windows PowerShell

```powershell
.\scripts\create_venv.ps1
.\.venv.win\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### WSL / Linux

```bash
python3 -m venv .venv
./.venv/bin/python -m ensurepip --upgrade
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e .
```

### 容器

```bash
docker compose build
docker compose run --rm test
```

常用容器入口：

- 开发 shell：`docker compose run --rm dev`
- 全量测试：`docker compose run --rm test`
- `v4` 开放系统 smoke：`docker compose run --rm smoke-v4`

## 快速验证

### 本地 Python 环境

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

Windows:

```powershell
.\scripts\run_python.ps1 -m unittest discover -s tests -v
```

### Docker

```bash
docker compose run --rm test
```

如果 Docker 在当前机器上报 `permission denied` 访问 `docker.sock`，先修复宿主机 Docker 权限，再重新执行容器命令。

## 典型入口

### 冻结参考 `v1`

- 实验脚本：[freeze_v1_global_cz_reference.py](experiments/freeze_v1_global_cz_reference.py)
- 出图脚本：[plot_freeze_v1_global_cz.py](scripts/plot_freeze_v1_global_cz.py)
- 结果文件：
  - [freeze_v1_global_cz_coarse_scan.json](artifacts/v1/freeze_v1_global_cz_coarse_scan.json)
  - [freeze_v1_global_cz_fine_scan.json](artifacts/v1/freeze_v1_global_cz_fine_scan.json)
  - [freeze_v1_global_cz_optimal.json](artifacts/v1/freeze_v1_global_cz_optimal.json)
  - [freeze_v1_global_cz_fit.json](artifacts/v1/freeze_v1_global_cz_fit.json)
  - [freeze_v1_global_cz_summary.png](artifacts/v1/freeze_v1_global_cz_summary.png)

### `v3` 双光子闭系统

- 主实验：[coarse_scan_two_photon_cz_v3.py](experiments/coarse_scan_two_photon_cz_v3.py)
- 局部扫描：[local_scan_two_photon_cz_v3_7p5_8p5.py](experiments/local_scan_two_photon_cz_v3_7p5_8p5.py)
- 出图脚本：[plot_two_photon_cz_v3.py](scripts/plot_two_photon_cz_v3.py)
- 结果目录：[artifacts/v3](artifacts/v3)

### `v4` `^171Yb` 开放系统

- 粗扫描：[two_stage_scan_two_photon_cz_v4_0_300ns_10mhz.py](experiments/two_stage_scan_two_photon_cz_v4_0_300ns_10mhz.py)
- 细扫描：[fine_scan_two_photon_cz_v4_90_150ns_10mhz.py](experiments/fine_scan_two_photon_cz_v4_90_150ns_10mhz.py)
- 单点优化：[optimize_yb171_v4_full_gate_300ns_10mhz.py](experiments/optimize_yb171_v4_full_gate_300ns_10mhz.py)
- 验证：[validate_v4_dynamics_and_optimization.py](experiments/validate_v4_dynamics_and_optimization.py)
- 结果目录：
  - [粗扫](artifacts/v4/coarse_0_300ns_10mhz)
  - [细扫](artifacts/v4/fine_90_150ns_0p5ns_10mhz)
  - [单点 300 ns](artifacts/v4/single_300ns_10mhz)
  - [验证](artifacts/v4/validation)

## 当前技术结论

- `v3` 之前的主线优化基本是我们自己写的 `SciPy expm/expm_frechet` 闭系统 GRAPE。
- `v4` 现在按论文 Eq.(7) 传播未归一化特殊态 `|01> + |11>`，并在开放系统下优化对应的 phase-gate fidelity。
- `v4` 当前主模型已经换成 `^171Yb` 的有效 `clock -> Rydberg` 完整门模型，不再把旧的双光子 `ladder + |e>` 结构当主线。
- `v4` 当前默认 `^171Yb` 标定以 `Muniz et al. 2025` 和 `Peper et al. 2025` 为主：保留 Rydberg decay、由 `T2* / T2_echo` 拆分得到的慢 UV detuning 与快 Rydberg dephasing surrogate、以及 pulse-area 漂移；默认不再把测得的 `T2*` 直接整体映射成 Lindblad dephasing，也不再默认加入 blockade jitter。
- 实现上继续使用 `QuTiP` 生成开放系统模型与 Liouvillian 并做独立验证，但优化与扫描逻辑由仓库内的 `open_system_grape.py` 控制。
- 本地 benchmark 表明，开放系统优化的主要瓶颈仍是 Liouvillian propagator 与 Frechet 梯度，而不是 Python 胶水代码。

## 当前已知状态

- WSL 环境下，依赖安装和主测试入口已可运行。
- 测试导入链路已经统一，不再依赖测试执行顺序。
- 当前仍有 1 个已知测试问题：`tests/test_two_photon_cz_9d.py` 中对 `theta` 的断言过于刚性，可能把等价最优相位误判为失败。

## 文档索引

- 架构总览：[docs/neutral-yb-architecture.md](docs/neutral-yb-architecture.md)
- 项目地图：[docs/project-map.md](docs/project-map.md)
- 版本历史：[docs/version-history.md](docs/version-history.md)
- 文献索引：[docs/references.md](docs/references.md)
- Ma 2023 time-optimal 两比特门复现：[docs/ma2023-time-optimal-2q.md](docs/ma2023-time-optimal-2q.md)
- 闭系统噪声与修正哈密顿量：[docs/yb-noise-and-corrected-hamiltonian.md](docs/yb-noise-and-corrected-hamiltonian.md)
- `v3` 双光子闭系统模型：[docs/two-photon-cz-v3-model.md](docs/two-photon-cz-v3-model.md)
- `v4` `^171Yb` 开放系统模型：[docs/two-photon-cz-v4-open-system.md](docs/two-photon-cz-v4-open-system.md)
