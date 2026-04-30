# rydcalc 接入 Ma 2023 六能级模型计划

## 目标边界

`rydcalc` 应先作为物理参数计算层接入，而不是替代当前的 GRAPE、Lindblad 或 Monte Carlo 时序演化器。当前 `neutral_yb` 代码继续负责：

- phase-rate Chebyshev pulse 生成与优化；
- 闭体系 unitary 演化；
- 开体系 Lindblad / no-jump / Monte Carlo 轨迹平均；
- gate/process fidelity 与 leakage 计算。

`rydcalc` 优先负责提供更真实的 Yb Rydberg 原子参数：

- `^171Yb` MQDT Rydberg 态能级；
- Rydberg 子能级 Zeeman 系数和磁场依赖；
- dipole / multipole matrix elements；
- pair-basis 相互作用谱、`C3/C6` 和有效 blockade shift；
- 后续可验证的 lifetime / blackbody transition rate。

## 已确认 API

本地子模块入口为 `rydcalc/rydcalc/__init__.py`。核心类和函数包括：

- `rydcalc.Ytterbium171(use_db=False, ...)`：`^171Yb` MQDT 原子模型。
- `Ytterbium171.get_state((v, L, F, m), tt="vlfm")`：获取 Rydberg 态；例如 S 态 `L=0`，Ma 2023 目标 manifold 对应 `F=3/2, m=-3/2,-1/2,1/2,3/2`。
- `state.get_energy_Hz()` / `state.energy_Hz`：态能量，单位 Hz。
- `state.get_multipole_me(other, k=1, qIn=...)`：multipole matrix element，原子单位。
- `rydcalc.environment(Bz_Gauss=..., Ez_Vcm=..., T_K=...)`：磁场、电场、温度环境。
- `rydcalc.pair(s1, s2)` 与 `rydcalc.pair_basis()`：构造双原子 pair basis。
- `pair_basis.computeHamiltonians(multipoles=[[1, 1]])`：预计算 field / interaction Hamiltonian。
- `pair_basis.computeHtot(env, rum, th, phi, interactions=True)`：给定距离和方向计算 pair spectrum，能量单位 Hz。
- `pair_basis.compute_Heff(subspace_pairs)`：在选定 pair 子空间中二阶消去得到有效 Hamiltonian，单位 Hz。
- `rydcalc.getCs(pb, env, rList_um=...)`：对 highlighted pair 拟合 `C6/C3`，输入距离 um，返回和 Hz/um 标度一致的相互作用系数。

## 当前环境限制

当前 `rydcalc` 子模块工作树保持干净，但在本项目 Python 3.12 / NumPy 2 环境中直接导入会失败：

```text
ModuleNotFoundError: No module named 'rydcalc.arc_c_extensions'
```

原因是 `Ytterbium171` 类定义阶段会实例化 `Ytterbium174(cpp_numerov=True)`，要求上游 C Numerov 扩展已编译。已有 `patches/rydcalc-python312-numpy2.patch` 记录了回退到 pure-Python Numerov 的兼容修复，但当前 `git -C rydcalc apply --check ../patches/rydcalc-python312-numpy2.patch` 未通过，需要在正式接入前刷新该 patch 或编译 C 扩展。

因此第一版 adapter 必须延迟导入 `rydcalc`，并在不可用时抛出明确错误，而不能让默认测试和普通模型导入失败。

## 与当前 Ma 2023 模型的映射

当前 `src/neutral_yb/models/ma2023_six_level.py` 使用论文 Methods 的无量纲模型：

- 单原子 basis：`|0>, |1>, |r_-3/2>, |r_-1/2>, |r_1/2>, |r_3/2>`。
- 完美 blockade 下分解为 `00`、`01/10`、`11` 三个 5D 子空间。
- Rydberg detuning 目前硬编码为 `(-3Δr, -2Δr, -Δr, 0)`。
- Clebsch 系数目前硬编码为 `1/2` 和 `1/(2 sqrt(3))`。
- 噪声使用无量纲 collapse rate，按 `rate / Ω_rad_s` 换算。

`rydcalc` 接入后，应该通过一个薄 adapter 生成显式参数对象，再传给 Ma 模型，而不是在 Hamiltonian 构造内部直接调用 `rydcalc`。建议参数包括：

- 四个 `m_F` Rydberg 子能级的相对 detuning，先以 Hz 保存，再除以 `effective_rabi_hz` 转为当前模型使用的 `Δ/Ω` 无量纲标度；等价地，也可以用角频率同时除以 `Ω_rad_s`。
- 四个 qubit-to-Rydberg coupling coefficient。短期保留论文 Clebsch 值；长期用 `rydcalc` matrix element 和激光偏振计算相对耦合。
- Rydberg blockade / pair interaction matrix。短期从 `computeHtot` 或 `getCs` 提取 effective blockade shift；长期替换当前 perfect-blockade 假设，进入有限 blockade 六能级或更大 pair-basis 模型。
- Rydberg lifetime。正式替换前必须和论文 `T1,r = 65 us` 做量级校准；不能直接假定 `rydcalc.total_decay` 对 Yb MQDT manifold 的 blackbody / branching 已满足实验模型。

## 噪声接入顺序

推荐按以下顺序接入，避免一次性把 Hamiltonian 和噪声都变成不可验证黑箱：

1. `rydcalc` import adapter：只负责可选导入、版本/路径诊断和错误提示。
2. Rydberg sublevel catalog：导出 Ma 2023 `6s59s 3S1 F=3/2` 四个子能级的 `state`、能量和 `g_F`。
3. Zeeman calibration：用 `g_F * μB B / h` 或 `environment(Bz_Gauss=...)` 得到 Hz 单位位移，再除以 `effective_rabi_hz` 验证 `Δr/Ω = 9.3 MHz / 1.6 MHz = 5.8`。
4. Pair interaction scan：用 `pair_basis` 计算 `|rr>` pair spectrum，提取 finite blockade shift，与当前 perfect-blockade 极限分开验证。
5. Matrix element / polarization：用 multipole matrix element 替换或校验论文 Clebsch-ratio coupling。
6. Decay model：先保持实验寿命 `65 us` 作为主参数；再用 `rydcalc` 的 decay/lifetime 结果作为对照或先验，不直接覆盖实验值。

## 第一版代码落点

建议新增：

- `src/neutral_yb/config/rydcalc_adapter.py`：可选导入、单位换算、Yb171 state catalog、pair interaction helper。
- `tests/test_rydcalc_adapter.py`：默认只测试 import failure 不污染主包；如果 `rydcalc` 可用，再跑 state catalog smoke test。
- `experiments/rydcalc_ma2023_pair_scan.py`：长运行 pair-basis 扫描入口，输出到 `artifacts/rydcalc_ma2023/...`。

默认 `unittest discover` 不应要求 `rydcalc` 可导入；需要真实 MQDT 计算的测试应当显式 skip。
