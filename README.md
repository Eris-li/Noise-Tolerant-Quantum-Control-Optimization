# Noise-Tolerant-Quantum-Control-Optimization

面向中性原子 `^171Yb` 量子计算的多比特量子门控制优化项目。

当前项目只关注两比特及以上的门，不实现独立单比特门。主线目标是围绕 `CZ`，后续扩展到 `CNOT` 和三比特门，从理想闭系统模型逐步推进到显式开放系统、含噪声、含损耗和实验标定的控制优化。

## 开源定位

这是一个研究型开源仓库，目标是让外部读者可以：

- 复现项目中保留的 `^171Yb` Rydberg two-qubit gate 模型、扫描和 benchmark。
- 阅读每条版本线的物理假设、数值方法和验证记录。
- 在不破坏既有复现链路的前提下继续开发新的模型、优化器和实验脚本。

仓库当前处于 `0.1.0` alpha 阶段。公开 API 仍可能随研究主线变化而调整；稳定性优先级是“可读、可复现、可测试”，而不是发布到 PyPI 的通用库接口。

如果你是第一次接触本项目，建议按这个顺序开始：

1. 读本 README 的“快速开始”和“典型入口”。
2. 跑 `python -m unittest discover -s tests -v`，确认本地环境可用。
3. 读 [docs/project-map.md](docs/project-map.md) 和 [docs/version-history.md](docs/version-history.md)。
4. 选择一条版本线，从对应的 `experiments/`、`scripts/`、`docs/` 和 `artifacts/` 开始。

参与开发请先读 [CONTRIBUTING.md](CONTRIBUTING.md)。如果使用本仓库结果，请参考 [CITATION.cff](CITATION.cff) 和 [docs/references.md](docs/references.md) 同时引用原始物理论文。

## 当前架构

核心代码是 Python 3.12 包 `neutral_yb`，位于 `src/neutral_yb/`：

- `config/`：`^171Yb` 物种配置、artifact 路径、`v4/v5` 标定 profile、Ma 2023 标定。
- `models/`：历史参考模型、当前 `^171Yb clock -> Rydberg` 开放系统模型、Ma 2023 复现模型。
- `optimization/`：闭系统 GRAPE、开放系统 GRAPE、Ma 2023 six-level 优化器。

运行入口和资料分开管理：

- `experiments/`：扫描、优化、验证、benchmark。
- `scripts/`：画图、数据导入、环境辅助脚本。
- `tests/`：`unittest` 测试。
- `docs/`：项目地图、版本历史、物理模型说明、文献与集成记录。
- `data/`：外部数据和处理后数据。
- `artifacts/`：已生成并保留的 JSON/PNG 结果。
- `rydcalc/`：Thompson Lab `rydcalc` Git submodule，用于未来接入 MQDT/Rydberg 能级、矩阵元和 pair-potential 计算。
- `patches/`：外部依赖的本地兼容补丁。

如果刚接手，建议先读：

1. [docs/project-map.md](docs/project-map.md)
2. [docs/version-history.md](docs/version-history.md)
3. [docs/references.md](docs/references.md)
4. [docs/rydcalc-integration.md](docs/rydcalc-integration.md)

## 版本线

仓库里并行保留多条有明确定位的版本线：

- `v1`：冻结的理想 4D global `CZ` 参考实验，用作回归和论文对照。
- `v2`：闭系统 5D 有限 blockade 修正模型。
- `v3`：闭系统双光子 9D 模型，显式包含中间态 `|e>`。
- `v4`：开放系统 `^171Yb clock -> Rydberg` 有效完整门模型，固定前后 shelving/unshelving，优化中间 UV 段。
- `v5`：当前 profile-free 的 `^171Yb` 标定扫描线，保留完整门核心模型，噪声假设从单一 baseline 重新整理。
- `ma2023_time_optimal_2q`：独立复现 Ma et al., Nature 622, 279-284 (2023) Fig. 3 的 metastable-qubit Rydberg gate，不并入 `v4/v5` 主线。

## 快速开始

### WSL / Linux

```bash
git submodule update --init --recursive
python3 -m venv .venv
./.venv/bin/python -m ensurepip --upgrade
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e .
```

如果要验证或开发 `rydcalc` 相关能力：

```bash
./.venv/bin/python -m pip install -e '.[rydcalc]'
./.venv/bin/python scripts/build_rydcalc_extension.py
```

### Windows PowerShell

```powershell
git submodule update --init --recursive
.\scripts\create_venv.ps1
.\.venv.win\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Docker

```bash
docker compose build
docker compose run --rm test
```

常用容器入口：

- `docker compose run --rm dev`：开发 shell。
- `docker compose run --rm test`：全量测试。
- `docker compose run --rm smoke-v4`：`v4` 开放系统 smoke workflow。

## 快速验证

本地测试：

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

Windows:

```powershell
.\scripts\run_python.ps1 -m unittest discover -s tests -v
```

Docker:

```bash
docker compose run --rm test
```

当前本地基线为 `43` 个 `unittest` 测试通过。

## 典型入口

### `v1` 冻结参考

- 实验：[experiments/freeze_v1_global_cz_reference.py](experiments/freeze_v1_global_cz_reference.py)
- 出图：[scripts/plot_freeze_v1_global_cz.py](scripts/plot_freeze_v1_global_cz.py)
- 结果：[artifacts/v1](artifacts/v1)

### `v3` 双光子闭系统

- 粗扫：[experiments/coarse_scan_two_photon_cz_v3.py](experiments/coarse_scan_two_photon_cz_v3.py)
- 局部扫描：[experiments/local_scan_two_photon_cz_v3_7p5_8p5.py](experiments/local_scan_two_photon_cz_v3_7p5_8p5.py)
- 出图：[scripts/plot_two_photon_cz_v3.py](scripts/plot_two_photon_cz_v3.py)
- 结果：[artifacts/v3](artifacts/v3)

### `v4` `^171Yb` 开放系统

- smoke：[experiments/run_two_photon_cz_v4_open_system_smoke.py](experiments/run_two_photon_cz_v4_open_system_smoke.py)
- 粗扫：[experiments/two_stage_scan_two_photon_cz_v4_0_300ns_10mhz.py](experiments/two_stage_scan_two_photon_cz_v4_0_300ns_10mhz.py)
- 细扫：[experiments/fine_scan_two_photon_cz_v4_90_150ns_10mhz.py](experiments/fine_scan_two_photon_cz_v4_90_150ns_10mhz.py)
- 单点优化：[experiments/optimize_yb171_v4_full_gate_300ns_10mhz.py](experiments/optimize_yb171_v4_full_gate_300ns_10mhz.py)
- 验证：[experiments/validate_v4_dynamics_and_optimization.py](experiments/validate_v4_dynamics_and_optimization.py)
- 结果：[artifacts/v4](artifacts/v4)

### `v5` `^171Yb` 标定扫描

- 两阶段扫描：[experiments/two_stage_scan_yb171_v5_0_300ns_10mhz.py](experiments/two_stage_scan_yb171_v5_0_300ns_10mhz.py)
- 出图：[scripts/plot_yb171_v5_10mhz_coarse.py](scripts/plot_yb171_v5_10mhz_coarse.py)
- 结果：[artifacts/v5](artifacts/v5)

### Ma 2023 独立复现线

主要文档：[docs/ma2023-time-optimal-2q.md](docs/ma2023-time-optimal-2q.md)

常用流程：

```bash
./.venv/bin/python scripts/import_ma2023_dataverse.py
./.venv/bin/python experiments/evaluate_ma2023_fig3_pulse.py --num-tslots 96 --ensemble-size 1 --output ma2023_fig3_pulse_96slot.json
./.venv/bin/python experiments/reproduce_ma2023_from_method.py --num-tslots 96 --max-iter 160 --num-restarts 4 --ensemble-size 1 --show-progress
./.venv/bin/python scripts/compare_ma2023_method_to_dataverse.py
```

Six-level/noisy 相关入口：

- [experiments/reproduce_ma2023_six_level_from_method.py](experiments/reproduce_ma2023_six_level_from_method.py)
- [experiments/evaluate_ma2023_six_level_noisy.py](experiments/evaluate_ma2023_six_level_noisy.py)
- [scripts/compare_ma2023_six_level_to_dataverse.py](scripts/compare_ma2023_six_level_to_dataverse.py)
- [scripts/plot_ma2023_six_level_noisy_eval.py](scripts/plot_ma2023_six_level_noisy_eval.py)

结果写入 [artifacts/ma2023_time_optimal_2q](artifacts/ma2023_time_optimal_2q)。

## rydcalc Submodule

`rydcalc/` 是上游 `https://github.com/ThompsonLabPrinceton/rydcalc.git` 的 Git submodule，当前用于后续接入 `^171Yb/^174Yb` MQDT、Rydberg 态、矩阵元和 pair-potential 计算。

常用命令：

```bash
git submodule update --init --recursive
git submodule status --recursive
```

Python 3.12 / NumPy 2 本地验证时需要先临时应用兼容 patch：

```bash
./.venv/bin/python -m pip install -e '.[rydcalc]'
./.venv/bin/python scripts/build_rydcalc_extension.py
./.venv/bin/python -c "from neutral_yb.external.rydcalc_adapter import build_yb171_atom; yb = build_yb171_atom(use_db=False); print(yb.get_state((60,0,0.5,0.5)))"
```

本地编译需要当前 Python 解释器对应的开发头文件，例如 Ubuntu/WSL 上的 `python3.12-dev`。Docker 镜像会在 build 阶段安装编译工具并构建该扩展。

更多说明见 [docs/rydcalc-integration.md](docs/rydcalc-integration.md)。

## 当前技术结论

- `v3` 之前主要是仓库内自写的 `SciPy expm/expm_frechet` 闭系统 GRAPE。
- `v4/v5` 使用 `^171Yb clock -> Rydberg` 有效完整门模型，开放系统优化由 [open_system_grape.py](src/neutral_yb/optimization/open_system_grape.py) 控制。
- QuTiP 主要用于算符构造、Liouvillian 构造和独立验证；扫描与优化循环仍在仓库内实现。
- `v5` 默认区分严格文献最小模型和实验 surrogate full profile，避免把标定假设混在单一默认值里。
- Ma 2023 复现线保持独立，避免把 metastable-qubit gate 的模型假设混入 `clock -> Rydberg` 主线。
- `rydcalc` 暂时作为 submodule 和未来适配层来源，不直接混入 `src/neutral_yb/`。

## Artifact 约定

不要随意覆盖已提交的 artifact。新增扫描应写入描述性、版本化目录，例如 `artifacts/v5/<profile>/...` 或 `artifacts/ma2023_time_optimal_2q/...`。

长时间运行的结果应记录命令、profile、关键物理假设和输出路径。

## 开发与复现约定

- 新物理模型先写清楚 Hilbert space、basis、耦合、detuning、噪声项和时间单位。
- 新优化流程需要同时给出目标函数、控制变量、约束、随机种子或 restart 策略。
- 新结果需要能从 notebook、experiment 命令或 JSON/CSV artifact 追溯。
- 新公共函数或模型应放入 `src/neutral_yb/`，并添加聚焦的 `unittest`。
- 不直接修改外部子模块源码；`rydcalc` 或 ARC 兼容性问题应通过 `patches/` 或 `src/neutral_yb/external/` 处理。

## 文档索引

- 架构总览：[docs/neutral-yb-architecture.md](docs/neutral-yb-architecture.md)
- 项目地图：[docs/project-map.md](docs/project-map.md)
- 版本历史：[docs/version-history.md](docs/version-history.md)
- 文献索引：[docs/references.md](docs/references.md)
- 开源维护说明：[docs/open-source-maintenance.md](docs/open-source-maintenance.md)
- Ma 2023 time-optimal 两比特门复现：[docs/ma2023-time-optimal-2q.md](docs/ma2023-time-optimal-2q.md)
- `rydcalc` 集成记录：[docs/rydcalc-integration.md](docs/rydcalc-integration.md)
- 闭系统噪声与修正哈密顿量：[docs/yb-noise-and-corrected-hamiltonian.md](docs/yb-noise-and-corrected-hamiltonian.md)
- `v3` 双光子闭系统模型：[docs/two-photon-cz-v3-model.md](docs/two-photon-cz-v3-model.md)
- `v4` `^171Yb` 开放系统模型：[docs/two-photon-cz-v4-open-system.md](docs/two-photon-cz-v4-open-system.md)
- `v5` `^171Yb` 开放系统模型：[docs/yb171-v5-open-system.md](docs/yb171-v5-open-system.md)
