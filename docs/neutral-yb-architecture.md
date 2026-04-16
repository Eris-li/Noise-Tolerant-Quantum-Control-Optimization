# Neutral $^{171}\mathrm{Yb}$ 项目初步架构

## 1. 目标与边界

本项目聚焦中性原子 $^{171}\mathrm{Yb}$ 量子计算，不考虑离子阱路线。目标不是只做一个“门保真度计算器”，而是建立一个从物理模型到控制优化的闭环：

1. 定义中性 $^{171}\mathrm{Yb}$ 的编码、辅助能级和门实现方案
2. 在脉冲级对单比特门和双比特门进行仿真
3. 显式纳入实验相关噪声
4. 通过优化得到高 fidelity、抗参数漂移和抗噪声的控制脉冲
5. 向上抽象为门级噪声模型，支持后续电路级评估

## 2. 文献学习后的核心判断

### 2.1 为什么选中性 $^{171}\mathrm{Yb}$

中性 $^{171}\mathrm{Yb}$ 的优势在于：

- 核自旋自由度长相干，适合做计算子空间
- 可利用 optical clock / metastable 态和 Rydberg 态实现高保真控制与纠缠门
- 与 optical tweezer 阵列天然兼容，便于扩展到多原子和并行门
- 一部分方案具有“泄漏到计算子空间外”的结构化误差特征，适合后续做 erasure-aware 建模

### 2.2 建议的项目主线

建议把第一阶段主线设为：

- 计算比特：ground-state nuclear-spin qubit
- 辅助态：clock state 与 Rydberg state
- 双比特门：Rydberg blockade / dark-state optimized CZ
- 单比特门：Raman 或等效单原子控制

这样选的原因是：

- 2025 年已经出现基于 $^{171}\mathrm{Yb}$ ground-state nuclear-spin qubit 的高保真通用门集结果，适合作为近期可对标目标
- 2023 年的 metastable $^{171}\mathrm{Yb}$ 方案依然非常重要，尤其适合后续做“泄漏/erasure”扩展
- 从软件架构看，二者共享大量底层模块，区别主要集中在编码和可访问跃迁

因此，项目架构建议采用：

- `v1`：ground-state nuclear-spin 主线
- `v2`：metastable qubit 扩展

## 3. 建议建模对象

### 3.1 单原子能级模型

第一版不追求把完整原子谱都装进仿真器，而是用“任务相关的最小有效模型”：

- 计算态：`|0>`, `|1>`
- 辅助态：`|c0>`, `|c1>` 或等效 clock/metastable manifold
- Rydberg 态：`|r>`
- 可选泄漏态：`|l_k>`

建议把模型拆成两层：

- `effective model`
  直接用于门脉冲优化，维度小，速度快
- `physics-informed model`
  纳入更真实的散射、失谐、运动和 blockade 偏差，用于校验和误差预算

### 3.2 多原子模型

双原子门是项目的核心对象。建议按层次支持：

1. 单原子模型
2. 双原子模型
3. 小规模并行门模型
4. 电路级等效噪声模型

不要一开始就做大规模全 Hilbert 空间 Lindblad 仿真，因为：

- 计算代价会迅速失控
- 优化环节需要大量重复求解
- 实际研发更依赖“脉冲级精确 + 门级抽象”的分层方法

## 4. 需要覆盖的物理噪声

对中性 $^{171}\mathrm{Yb}$，建议至少覆盖以下噪声源。

### 4.1 控制噪声

- 激光幅度噪声
- 激光相位噪声 / 频率噪声
- 静态和慢漂移失谐
- 脉冲定时误差
- 光强空间不均匀带来的 site-to-site Rabi 波动

### 4.2 原子运动与几何噪声

- 有限温度导致的 Doppler 效应
- 光镊内位置涨落
- 原子间距偏差导致的 blockade 强度变化
- 并行门时的串扰与 spectator interaction

### 4.3 原子内禀与开放系统噪声

- 中间态或辅助态散射
- Rydberg 态有限寿命
- blackbody-induced decay / transfer
- 磁场波动引起的相位漂移
- 泄漏到计算子空间外
- 原子丢失与可探测 erasure

## 5. 仿真分层

这是项目最关键的架构选择。建议严格分成三层。

### 5.1 Pulse Layer

输入：

- 原子参数
- 脉冲参数
- 两原子几何
- 噪声样本

输出：

- 时间演化
- 目标门对应的过程矩阵 / unitary
- leakage、loss、population transfer、phase accumulation

这一层解决“某个脉冲到底做出了什么门”。

### 5.2 Gate Layer

将 pulse layer 的结果压缩成：

- 理想门
- coherent over-rotation / phase error
- stochastic Pauli / dephasing / loss / leakage 通道
- 对实验参数的灵敏度曲线

这一层解决“这个物理门如何作为一个可复用的门对象进入编译或电路仿真”。

### 5.3 Circuit Layer

在门级噪声抽象下支持：

- 小规模算法电路评估
- 随机基准测试流程复现
- 并行门调度下的误差累积
- 后续 QEC / erasure-aware 分析

## 6. 控制优化闭环

建议把控制优化做成项目主干，而不是附加功能。

### 6.1 优化目标

单比特门和双比特门分别优化：

- 平均门保真度
- 最坏情况保真度
- leakage / loss
- 对幅度误差、失谐误差、位置误差的鲁棒性

### 6.2 优化形式

建议从易到难分三步：

1. 有限维参数化脉冲
   例如 Gaussian, Blackman, spline, piecewise-constant
2. 鲁棒目标函数
   对参数分布采样求期望或 worst-case
3. 多目标优化
   同时压低 infidelity、leakage 和脉冲时长

### 6.3 推荐算法接口

- `scipy.optimize`：第一版够用，便于起步
- GRAPE 风格梯度优化：后续加入
- 可微模拟后端：后续可接 JAX

注意：第一阶段不要过早绑定单一优化算法，应该先把“目标函数 + 仿真器 + 参数化脉冲”的接口做稳。

## 7. 建议的软件目录结构

```text
Noise-Tolerant-Quantum-Control-Optimization/
├─ README.md
├─ docs/
│  └─ neutral-yb-architecture.md
├─ src/
│  └─ neutral_yb/
│     ├─ config/
│     │  ├─ species.py
│     │  ├─ hardware.py
│     │  └─ defaults.py
│     ├─ physics/
│     │  ├─ basis.py
│     │  ├─ levels.py
│     │  ├─ hamiltonians/
│     │  │  ├─ single_atom.py
│     │  │  ├─ two_atom.py
│     │  │  └─ blockade.py
│     │  ├─ dissipators.py
│     │  └─ observables.py
│     ├─ control/
│     │  ├─ pulses.py
│     │  ├─ schedules.py
│     │  ├─ parameterizations.py
│     │  └─ constraints.py
│     ├─ noise/
│     │  ├─ laser.py
│     │  ├─ motion.py
│     │  ├─ magnetic.py
│     │  ├─ leakage.py
│     │  └─ sampling.py
│     ├─ simulation/
│     │  ├─ solvers.py
│     │  ├─ pulse_simulator.py
│     │  ├─ gate_extractor.py
│     │  └─ circuit_model.py
│     ├─ gates/
│     │  ├─ single_qubit.py
│     │  ├─ cz.py
│     │  ├─ parallel_cz.py
│     │  └─ calibrations.py
│     ├─ optimization/
│     │  ├─ objectives.py
│     │  ├─ robust_objectives.py
│     │  ├─ optimize.py
│     │  └─ benchmarks.py
│     ├─ experiments/
│     │  ├─ benchmark_1q.py
│     │  ├─ benchmark_2q.py
│     │  ├─ sweep_detuning.py
│     │  └─ sweep_blockade.py
│     └─ utils/
│        ├─ units.py
│        ├─ linops.py
│        └─ io.py
├─ tests/
│  ├─ test_hamiltonians.py
│  ├─ test_noise_models.py
│  ├─ test_gate_extraction.py
│  └─ test_robust_optimization.py
└─ notebooks/
   ├─ 01_single_qubit_basics.ipynb
   ├─ 02_two_qubit_cz.ipynb
   └─ 03_robust_optimization.ipynb
```

## 8. 每层模块的职责

### `config/`

保存物种参数、默认实验参数和硬件约束，不要把这些常数散落在仿真脚本中。

### `physics/`

负责构造哈密顿量、耗散项和观测量，是整个项目的物理核心。

### `control/`

只负责“如何表达脉冲和时序”，不关心具体原子是什么。

### `noise/`

把噪声对象化、参数化，保证噪声可以被采样，也可以被扫参分析。

### `simulation/`

负责求解时间演化，并从演化结果提取门矩阵、过程矩阵和误差指标。

### `gates/`

封装常用门实现流程，避免每次都从头搭脉冲和观测量。

### `optimization/`

连接 pulse simulator 和目标函数，是后续鲁棒控制的主战场。

### `experiments/`

保存“像论文图一样”的实验脚本，方便复现曲线和比较不同控制策略。

## 9. 建议的开发顺序

### Milestone 1: 单原子与单比特门

- 建立单原子有效能级模型
- 支持基本单比特旋转
- 加入幅度误差、失谐、磁场漂移
- 输出 Bloch 球轨迹和门保真度

### Milestone 2: 双原子 Rydberg CZ

- 建立两原子 blockade 哈密顿量
- 实现基础 CZ 门脉冲
- 提取双比特门保真度、leakage 和 phase error
- 扫描 blockade 强度、失谐、Rabi 频率

### Milestone 3: 鲁棒优化

- 参数化脉冲
- 对幅度误差、位置误差、失谐做采样鲁棒优化
- 输出最优脉冲与误差预算

### Milestone 4: 并行门与门级抽象

- 加入多个同时执行的 CZ 对
- 建立 spectator interaction 模型
- 抽取门级等效噪声，用于小电路仿真

### Milestone 5: metastable / erasure 扩展

- 增加 metastable $^{171}\mathrm{Yb}$ 编码
- 显式建模泄漏与原子丢失
- 支持 erasure-aware gate/circuit analysis

## 10. 第一版实现建议

如果我们要尽快开始编码，第一版建议采用：

- 语言：Python
- 数值层：`numpy` + `scipy`
- 开放系统与量子对象：可先自写小型线性代数层，必要时接入 `QuTiP`
- 优化：`scipy.optimize`
- 数据管理：`pydantic` 或 `dataclass`
- 作图：`matplotlib`

理由是：

- 上手快
- 易于调试
- 适合先把模型和接口做对
- 之后若需要自动微分和大规模参数优化，再迁移或并接 JAX

## 11. 文献依据

下面这些文献最直接支撑本项目的架构选择。

1. Mark Saffman, “Quantum computing with neutral atoms”, *National Science Review* 6, 24–25 (2019).
   链接：[https://pmc.ncbi.nlm.nih.gov/articles/PMC8291449/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8291449/)
   用途：中性原子量子计算总体路线、Rydberg 纠缠门背景。

2. Simon J. Evered et al., “High-fidelity parallel entangling gates on a neutral-atom quantum computer”, *Nature* 622, 268–272 (2023).
   链接：[https://www.nature.com/articles/s41586-023-06481-y](https://www.nature.com/articles/s41586-023-06481-y)
   用途：并行高保真中性原子纠缠门、optimal-control 风格单脉冲 CZ、误差源分析。

3. Shuo Ma et al., “High-fidelity gates and mid-circuit erasure conversion in an atomic qubit”, *Nature* 622, 279–284 (2023).
   链接：[https://www.nature.com/articles/s41586-023-06438-1](https://www.nature.com/articles/s41586-023-06438-1)
   用途：metastable $^{171}\mathrm{Yb}$ 编码、单/双比特门、erasure 转换思路。

4. J. A. Muniz et al., “High-Fidelity Universal Gates in the $^{171}$Yb Ground-State Nuclear-Spin Qubit”, *PRX Quantum* 6, 020334 (2025).
   链接：[https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.6.020334](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.6.020334)
   用途：ground-state nuclear-spin 主线、通用门集、近期可对标 fidelity 指标。

5. Zhubing Jia et al., “An architecture for two-qubit encoding in neutral ytterbium-171 atoms”, *npj Quantum Information* 10, 106 (2024).
   链接：[https://www.nature.com/articles/s41534-024-00898-7](https://www.nature.com/articles/s41534-024-00898-7)
   用途：说明 $^{171}\mathrm{Yb}$ 具有超越单一 qubit 编码的可扩展空间，适合作为后续 ququart / 多自由度扩展参考。

6. Madhav Mohan, Robert de Keijzer, Servaas Kokkelmans, “Robust control and optimal Rydberg states for neutral atom two-qubit gates”, *Physical Review Research* 5, 033052 (2023).
   链接：[https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.033052](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.033052)
   用途：鲁棒控制目标设计、Rydberg 态选择、对控制偏差的鲁棒优化。

7. Matthew L. Day et al., “Limits on atomic qubit control from laser noise”, *npj Quantum Information* 8, 72 (2022).
   链接：[https://www.nature.com/articles/s41534-022-00586-4](https://www.nature.com/articles/s41534-022-00586-4)
   用途：激光噪声建模，支持把噪声从“白噪声抽象”推进到更真实的 PSD 级建模。

## 12. 最重要的架构原则

这类项目最容易失败的点，不是数学推导不够，而是把所有层都揉在一起。第一版一定要坚持下面三条：

- 物理参数、脉冲参数、噪声参数分离
- pulse-level 与 gate-level 分离
- “先做可验证的小模型，再逐层加复杂度”

如果遵循这三条，后续无论你要走 ground-state、metastable，还是加入 ququart 编码，项目都不会推倒重来。
