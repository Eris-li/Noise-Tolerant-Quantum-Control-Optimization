# Neutral `^171Yb` 项目框架

## 1. 当前定位

本项目当前只做**两比特及以上量子门**，不做单比特门。

原因很直接：

- 当前最清晰、最容易冻结成参考版本的是 two-qubit global `CZ`
- 单比特门和读出当然重要，但不是当前控制优化主线
- 后续我们更关心的是 `CZ`、`CNOT`、并行门、三比特门，以及这些门在真实噪声下的鲁棒优化

## 2. 当前冻结参考

已经冻结的 `v1` 是：

- 论文：`arXiv:2202.00903`
- 对象：ideal global `CZ`
- 模型：4 维对称约化
- 目标：phase-gate fidelity
- 输出：粗扫、细扫、近阈值拟合、time-optimal phase sequence

这条链路的意义是：

- 给后续所有更复杂模型提供一个无噪声基准
- 保证以后加噪声、加非对称项、加开放系统项时，有一个不会漂移的对照

## 3. 后续框架

建议后续分成三层。

### 3.1 Reference Layer

保存已经冻结的参考实验。

当前只有：

- `freeze_v1_global_cz_reference.py`

后续可以新增：

- `freeze_v2_noisy_cz_reference.py`
- `freeze_v3_cnot_reference.py`

但每一版一旦冻结，就不应再被随意改写。

### 3.2 Model Layer

保存物理模型。

当前：

- `GlobalCZ4DModel`

后续建议新增：

- `FiniteBlockadeCZ5DModel`
- `NoisyGlobalCZModel`
- `AsymmetricCNOTModel`
- `ThreeQubitGateModel`

### 3.3 Optimization Layer

保存控制优化问题，而不是只保存某个特定 gate。

当前：

- `PaperGlobalPhaseOptimizer`

后续建议新增：

- 带失谐和幅度误差鲁棒目标的优化器
- 带 Lindblad 项的开放系统优化器
- 多目标优化器：同时最小化 `1-F`、leakage、loss、gate time

## 4. 下一步最值得做的事

从当前项目状态看，最合理的推进顺序是：

1. 从 ideal 4D global `CZ` 升级到“有限 blockade + 失谐 + 衰减”的修正模型
2. 在这个模型上重新优化 `CZ`
3. 提取误差预算
4. 再扩展到 `CNOT`
5. 最后再做三比特门

## 5. 设计原则

- 优先冻结参考，不优先堆功能
- 优先做两比特及以上门，不优先做单比特门
- 噪声和开放系统项要从一开始预留接口
- ideal、noisy、open-system 三个层次不要混写在一个脚本里
