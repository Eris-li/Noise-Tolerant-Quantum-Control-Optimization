# `v4`：`^171Yb` 开放系统 `CZ` 模型详解

这份文档只做一件事：

- 把当前仓库里 **`v4` 的源码到底在模拟什么、怎么演化、怎么加噪声、怎么做优化** 讲清楚

假设读者只有量子力学基础，不假设你已经熟悉中性原子门、Rydberg blockade 或 `^171Yb` 的实验细节。

文档内容以**当前源码**为准，核心入口是：

- 模型：[yb171_clock_rydberg_cz_open.py](../src/neutral_yb/models/yb171_clock_rydberg_cz_open.py)
- 标定：[yb171_calibration.py](../src/neutral_yb/config/yb171_calibration.py)
- 优化器：[open_system_grape.py](../src/neutral_yb/optimization/open_system_grape.py)

## 1. 这份 `v4` 到底在模拟什么

当前 `v4` 不是旧版本里的“双光子 `ladder + |e>` 中间态”模型。

它现在模拟的是更贴近 `^171Yb` 实验门机制的**完整三段原生 `CZ` 门**：

1. 先用固定形状的 `clock shelving` 脉冲把逻辑态 `|1>` 转到长寿命 `clock` 态 `|c>`
2. 再用可优化的全局 UV 脉冲驱动 `|c> <-> |r>`，通过 Rydberg blockade 实现受控相位
3. 最后用固定形状的 `clock unshelving` 脉冲把 `|c>` 映回逻辑态 `|1>`

也就是说，当前代码现在显式传播的是：

- **完整门序列**

但真正作为优化变量的仍然只有：

- **中间那段 UV 脉冲的 `x/y` 两个 quadrature**

固定前后两段 `clock` 脉冲也被纳入总门时间和总门误差，只是不参与 GRAPE 优化。

## 2. 为什么这比旧的 ladder 模型更像 `^171Yb`

旧的 surrogate 模型更像“同时开两束光、远失谐经过一个短寿命中间态”的两光子门。

而当前 `^171Yb` 高保真门的实验图像是：

- qubit 在基态核自旋空间
- 先通过窄线宽时钟跃迁把 `|1>` 转到 metastable `clock` 态
- 再从 `clock` 态用 UV 单光子打到 Rydberg 态
- 通过 blockade 得到两比特相位

当前代码正是按这个物理流程抽象出的有效模型。

所以要把它理解成：

- “**完整门序列的有效模型**”

而不是：

- “把实验里所有微观能级逐字展开后的全原子结构模型”

## 3. 这个模型里的 Hilbert 空间是什么

当前模型用的是一个 **11 维有效空间**，定义在 [yb171_clock_rydberg_cz_open.py](../src/neutral_yb/models/yb171_clock_rydberg_cz_open.py)。

基底顺序是：

1. `|01>`
2. `|0c>`
3. `|0r>`
4. `|11>`
5. `|W_c> = (|1c> + |c1>) / sqrt(2)`
6. `|cc>`
7. `|W_r> = (|1r> + |r1>) / sqrt(2)`
8. `|W_cr> = (|cr> + |rc>) / sqrt(2)`
9. `|rr>`
10. `|leak>`
11. `|loss>`

这里要特别解释每一个态的物理意义。

### 3.1 `|01>` 和 `|11>`

这两个态是门优化真正关心的两个“active branches”。

为什么只显式保留这两个，而不是完整的 `|00>, |01>, |10>, |11>` 四维逻辑空间？

原因是当前采用的是论文 `Eq.(7)` 的特殊态 fidelity 公式。那个公式只需要追踪：

- 单个参与门的分支
- 两个都参与门的分支

在当前对称约化后，它们对应的就是：

- `|01>`
- `|11>`

可以这样理解：

- `|00>` 是完全 spectator，不被 UV 门段驱动
- `|10>` 与 `|01>` 在对称全局门近似下具有同样的动力学角色

所以优化时只显式跟踪最关键的两个分支。

### 3.2 `|0c>` 和 `|cc>`

这两个态对应前后两段固定 `clock` 脉冲真正要到达的 shelved 态：

- `|01> -> |0c>`
- `|11> -> |cc>`

如果 shelving 完全理想，进入 UV 段前系统主要就在这两个态上。

### 3.3 `|0r>`

这个态表示：

- 只有一个参与门的原子被激发到 Rydberg 态

它是 `|01>` 在 UV pulse 作用下最直接耦合到的中间态。

### 3.4 `|W_c>`, `|W_r>` 和 `|W_cr>`

这三个对称态分别表示：

- 一个原子在 `clock`、一个原子还在逻辑 `|1>`
- 一个原子在 `Rydberg`、一个原子还在逻辑 `|1>`
- 一个原子在 `clock`、一个原子在 `Rydberg`

它们之所以必须保留，是因为：

- 不完美 shelving 会让 `|11>` 在前缀脉冲后残留到 `|W_c>`
- UV 脉冲会把 `|W_c>` 耦合到 `|W_r>`
- 不完美 UV 演化会让 `|cc>`, `|W_cr>`, `|rr>` 在后缀脉冲阶段继续互相转化出残余振幅

所以只做 `UV` 段模型已经不够，完整门必须把这些“前后脉冲引出的对称支路”也显式放进来。

### 3.5 `|W_cr>`

定义为：

```math
|W_{cr}\rangle = \frac{|cr\rangle + |rc\rangle}{\sqrt{2}}
```

它表示：

- 在双原子都参与门时，系统先从 `|11>` 对称地耦合到“一个在 clock，一个在 Rydberg”的对称组合

这个态是两比特门中最关键的“中间受阻塞通道”。

### 3.6 `|rr>`

这个态表示：

- 两个原子都在 Rydberg 态

如果没有相互作用，它会被正常占据；有 blockade 时，它的能量被抬高，从而抑制双激发。

这正是 Rydberg blockade 门的核心。

### 3.7 `|leak>` 和 `|loss>`

这两个态不是逻辑空间的一部分。

它们是为了把实验里的非理想过程显式放进模型：

- `|leak>`：仍在系统 Hilbert 空间里，但已经跑到不想要的旁支子空间
- `|loss>`：真正不可恢复的损失，例如原子丢失或被当作永久离开逻辑空间

这样做的好处是：

- 优化器不仅会看到“相位做错了”
- 也会看到“人口跑丢了”

## 4. 当前模型的物理近似是什么

这个 `v4` 不是什么都模拟了，它做了几层明确的近似。

### 4.1 固定前后缀脉冲，不优化它们

当前已经显式模拟两个 `clock` 脉冲的时间演化，但它们是固定的 Blackman 形脉冲，不参与 GRAPE 变量搜索。

所以当前优化器只负责：

- 如何让中间这段 UV 脉冲在完整门序列里给逻辑态积累对的相位，同时不留下 `clock` / `Rydberg` 残余人口

### 4.2 对称约化

当前没有把两个原子的完整 tensor-product 空间全部保留下来，而是利用对称性把最关键的对称态抽出来。

这样做会让模型：

- 更快
- 更适合大量 GRAPE 迭代

代价是它不是一个“所有微观细节都显式展开”的超大模型。

### 4.3 噪声以“开放系统 + 准静态 ensemble”两层来表示

当前噪声不是只用一种方式表示，而是分成两类：

1. **真正的 Lindblad 开放系统噪声**
   例如 Rydberg decay
2. **shot-to-shot 变化但在单次门内近似不变的准静态误差**
   例如 detuning 漂移、pulse-area 漂移、blockade 偏差

这点很重要，因为它决定了为什么有些误差用 collapse operator 写，有些误差是通过 ensemble 采样写。

## 5. 哈密顿量怎么写

当前哈密顿量写成：

```math
H(t) = H_0 + u_x(t) H_x + u_y(t) H_y
```

其中：

- `H_0` 是 drift Hamiltonian
- `u_x(t), u_y(t)` 是两个可控的 Cartesian quadrature

这部分对应 [yb171_clock_rydberg_cz_open.py](../src/neutral_yb/models/yb171_clock_rydberg_cz_open.py) 的：

- `drift_hamiltonian()`
- `lower_leg_control_hamiltonians()`

这里虽然函数名还沿用了旧接口里的 `lower_leg` 字样，但在当前 `^171Yb` 模型里，它实际代表的是：

- **单束 UV 驱动的两个正交控制分量**

不是旧双光子模型里的“lower leg”了。

### 5.1 Drift Hamiltonian

在当前基底下，`H_0` 的非零对角项是：

- `|0r>` 带单原子分支 detuning
- `|W_cr>` 带双原子分支 detuning
- `|rr>` 带 `blockade_shift - 2 * detuning`

物理意义分别是：

- 单激发偏离共振多少
- 双原子对称单激发支路偏离共振多少
- 双 Rydberg 激发态因为相互作用被抬高了多少

其中最重要的量是 `blockade_shift`。

如果它足够大，`|rr>` 就很难被真正占据，系统主要在：

- `|11>`
- `|W_cr>`

之间绕一圈，最后回到 `|11>`，但多拿到一个条件相位。

### 5.2 控制哈密顿量

控制项只负责：

- 在 `|01> <-> |0r>`
- 在 `|11> <-> |W_cr>`
- 在 `|W_cr> <-> |rr>`

之间建立耦合。

仓库内部不直接把控制变量写成 “振幅 `Omega(t)` 与相位 `phi(t)`”，而是写成两个正交分量：

```math
u_x(t), \quad u_y(t)
```

它们和振幅、相位之间的关系是：

```math
\Omega(t) = \sqrt{u_x(t)^2 + u_y(t)^2}
```

```math
\phi(t) = \mathrm{atan2}(u_y(t), u_x(t))
```

这意味着：

- 当前优化器其实已经在同时优化振幅和相位
- 只是数值优化时用 Cartesian 变量更稳定

## 6. 为什么内部用 `u_x, u_y`，但结果里又能看到 `Omega(t), phi(t)`

这是数值优化里很常见的做法。

如果直接把变量写成：

- `Omega(t)`
- `phi(t)`

那么当 `Omega(t)` 接近 `0` 时，`phi(t)` 会变得不稳定，因为这时相位本身没什么物理意义。

而用：

- `u_x(t)`
- `u_y(t)`

则不会有这个奇点问题，梯度也更平滑。

所以当前代码的做法是：

1. 优化时用 `u_x, u_y`
2. 保存结果和画图时再转换成 `Omega, phi`

这就是为什么你会在结果文件里同时看到：

- `ctrl_x`
- `ctrl_y`
- `amplitudes`
- `phases`

## 7. 开放系统噪声怎么进模型

当前所有开放系统噪声都定义在：

- `Yb171ClockRydbergNoiseConfig`

源码位置：[yb171_clock_rydberg_cz_open.py](../src/neutral_yb/models/yb171_clock_rydberg_cz_open.py)

它包含这些字段：

- `common_uv_detuning`
- `differential_uv_detuning`
- `blockade_shift_offset`
- `uv_amplitude_scale`
- `rydberg_decay_rate`
- `rydberg_dephasing_rate`
- `neighboring_mf_leakage_rate`

这些量并不都通过同一种方式生效。

### 7.1 准静态哈密顿量误差

下面这些量是直接加进哈密顿量里的：

- `common_uv_detuning`
- `differential_uv_detuning`
- `blockade_shift_offset`
- `uv_amplitude_scale`

它们的含义分别是：

- **common UV detuning**：所有分支共同看到的 UV 失谐
- **differential UV detuning**：双原子相关分支额外看到的 detuning 差异
- **blockade shift offset**：真实 blockade 与名义 `160 MHz` 的偏差
- **uv_amplitude_scale**：pulse-area 或强度标定误差

这些误差在一条门脉冲内部被视为常数，但在不同 shots 之间可以变，这就是为什么它们很适合做 **quasistatic ensemble**。

### 7.2 Lindblad 噪声

下面这些量通过 collapse operators 进入：

- `rydberg_decay_rate`
- `rydberg_dephasing_rate`
- `neighboring_mf_leakage_rate`

#### Rydberg decay

这表示：

- 一旦原子在 Rydberg 态里，它有一定几率不可逆地丢失到 `|loss>`

实现上：

- `|0r> -> |loss>`
- `|W_cr> -> |loss>`
- `|rr> -> |loss>`，而且双激发的 loss 率是两倍

这和实验误差预算里“有限 Rydberg lifetime 带来的损失”是一致的。

#### Rydberg dephasing

这表示：

- Rydberg 态相对于非 Rydberg 态发生纯退相干

它是通过一个与 Rydberg 占据数相关的 Lindblad 算符实现的。

注意当前默认值是：

- `rydberg_dephasing_rate = 0`

这不是说实验里完全没有相位噪声，而是因为当前默认把 `T2*` 主要解释成**准静态 detuning**，不再默认把它整块又塞成 Markovian dephasing，以避免双重计数。

#### Neighboring `m_F` leakage

这表示：

- 被激发到不希望的近邻 Zeeman / hyperfine 支路

在当前有效模型里，它被简化成：

- `|0r>, |W_cr>, |rr>` 跑到 `|leak>`

这是一个很粗粒度但很实用的写法，目的是让优化器知道：

- “不只会掉到彻底丢失，还可能跑到错误但仍有残余占据的旁支”

## 8. 当前默认参数到底是多少

参数定义在 [yb171_calibration.py](../src/neutral_yb/config/yb171_calibration.py) 的 `Yb171ExperimentalCalibration` 里。

当前默认值是：

- `uv_rabi_hz = 10e6`
- `uv_rabi_hz_max = 10e6`
- `clock_shelving_rabi_hz = 7e3`
- `clock_pi_pulse_duration_s = 130e-6`
- `blockade_shift_hz = 160e6`
- `rydberg_lifetime_s = 65e-6`
- `rydberg_t2_star_s = 3.4e-6`
- `rydberg_t2_echo_s = 5.1e-6`
- `uv_pulse_area_fractional_rms = 0.004`
- `differential_uv_detuning_rms_hz = 0.0`
- `blockade_shift_jitter_hz = 0.0`
- `markovian_rydberg_dephasing_t2_s = None`
- `neighboring_mf_leakage_per_gate = 0.0`

这些数值的来源口径是：

- `Muniz et al., PRX Quantum 6, 020334 (2025)`
- `Peper et al., Phys. Rev. X 15, 011009 (2025)`

### 8.1 为什么 `uv_rabi` 在内部常常等于 `1`

这是很多人第一次看代码会困惑的地方。

当前内部使用的是无量纲时间单位：

```math
t_\Omega = (2\pi \Omega_{\mathrm{ref}}) t
```

这里参考频率就是你传入的 `effective_rabi_hz`，默认是 `10 MHz`。

所以一旦用这个 `Omega_ref` 做无量纲化，名义最大 UV Rabi 频率本身就会变成：

```math
\Omega_{\mathrm{UV}} / \Omega_{\mathrm{ref}} = 1
```

因此：

- `uv_rabi = 1`

不是说驱动弱，而是说：

- **内部单位已经把它拿来当标尺了**

而 `blockade_shift = 160 MHz` 在这个单位下就变成：

```math
160 / 10 = 16
```

这意味着：

- 阻塞强度大约是名义最大驱动的 `16` 倍

## 9. `T2*` 和 `T2_echo` 在程序里怎么理解

这两个量现在保留在标定里，但默认不直接进 Lindblad。

### 9.1 `T2* = 3.4 us`

这是非回波的相干时间。

它包含了：

- 慢激光相位噪声
- shot-to-shot detuning 漂移
- Doppler
- 其他低频误差

所以它更像“实验上看到的总表观退相干时间”，而不是一个可以直接等价成单个 `gamma_phi` 的纯 Markovian 量。

### 9.2 `T2_echo = 5.1 us`

这是做了 echo 之后剩余的相干时间，更接近快噪声极限。

因此如果未来要显式打开 `rydberg_dephasing_rate`，更自然的做法是：

- 只把真正不可 refocus 的快噪声映射到 Lindblad dephasing

而不是把整个 `T2*` 生吞进去。

## 10. 时间演化是怎么做的

当前仓库没有把 `QuTiP mesolve` 当主结果。

真正的主传播逻辑在 [open_system_grape.py](../src/neutral_yb/optimization/open_system_grape.py) 里，是仓库自己写的逐 slice propagation。

### 10.1 时间离散

总门时长 `T` 被切成 `num_tslots` 个小片。

每一片都假设控制量是常数：

```math
u_x(t) = u_x^{(k)}, \qquad u_y(t) = u_y^{(k)}
```

这叫做 piecewise-constant control。

单片时长是：

```math
\Delta t = T / N
```

### 10.2 完整 Liouvillian 传播

如果你调用：

- `evolve_density_matrix()`
- `trajectory()`

代码会在每个 slice 上构造：

```math
L_k = L_d + u_x^{(k)} L_x + u_y^{(k)} L_y
```

然后用：

```math
e^{\Delta t L_k}
```

推进密度矩阵的列向量化表示。

这部分是真正的开放系统传播。

### 10.3 优化器内部为什么又用了非厄米生成元

当前优化目标采用的是论文 `Eq.(7)` 的特殊态公式，它要求我们追踪：

- `|01>`
- `|11>`

两个分支的**复振幅**

而不是只看最终密度矩阵投影到 active 子空间后的实数概率。

为了保留这些相干振幅，优化器内部传播的是有效非厄米生成元：

```math
G = -iH - \frac{1}{2}\sum_k C_k^\dagger C_k
```

再强调一次：

- **分析接口** 用的是完整 Liouvillian
- **主优化目标** 用的是 `Eq.(7)` 所需的特殊态振幅传播

这不是随便混用，而是因为当前 fidelity 指标本身就是按这条思路定义的。

## 11. 当前优化目标到底是什么

这一点非常关键。

当前 `v4` 主优化目标跟 `2202.00903` 的 `Eq.(7)` 一致：

初态取未归一化特殊态：

```math
|\psi(0)\rangle = |01\rangle + |11\rangle
```

最终定义：

```math
a_{01} = e^{-i\theta} \langle 01|\psi(T)\rangle
```

```math
a_{11} = -e^{-2i\theta} \langle 11|\psi(T)\rangle
```

然后 fidelity 是：

```math
F = \frac{|1 + 2a_{01} + a_{11}|^2 + 1 + 2|a_{01}|^2 + |a_{11}|^2}{20}
```

这里的 `theta` 不是手工固定死的，而是会在每次评估中作为一个附带优化变量一起找最优值。

物理上，这相当于：

- 允许把最终门里那部分“单比特 `Z` 相位”吸收到一个可调的逻辑相位规范里
- 我们真正关心的是 entangling part 有没有达到 `CZ`

## 12. 为什么这个 fidelity 看起来只传播了一个态，却还能代表整个门

因为当前门的结构有很强的对称性。

对于这种全局受控相位门，`Eq.(7)` 已经把平均门保真度化简成了特殊态的表达式。

所以：

- 当前不是“随便挑一个 probe 态做 surrogate”
- 而是“用论文给出的、对这个门结构成立的化简公式”

这点和早期某些 probe-state surrogate 是不同的。

## 13. 优化变量是什么

当前优化器每次同时优化：

- `ctrl_x[0:N]`
- `ctrl_y[0:N]`
- 一个全局 `theta`

也就是说，如果一共有 `N` 个 time slices，那么总变量数是：

```math
2N + 1
```

例如：

- `N = 48` 时，总变量数是 `97`

## 14. 优化器是怎么更新这些变量的

主优化器是：

- `scipy.optimize.minimize(..., method="L-BFGS-B")`

也就是带边界约束的拟牛顿法。

### 14.1 为什么不用数值差分梯度

因为太慢。

当前每个时间片都要做一次矩阵指数传播，如果再对每个控制变量做有限差分，成本会爆炸。

所以当前代码使用：

- `scipy.linalg.expm_frechet`

来求矩阵指数对控制变量的导数。

这样就能得到解析梯度近似意义上的高效实现。

### 14.2 优化目标函数的完整形式

当前目标函数不是只有 fidelity，还可以加平滑正则：

```math
J = 1 - F + \lambda_s C_{\mathrm{smooth}} + \lambda_c C_{\mathrm{curv}}
```

其中：

- `F` 是上面那条 `Eq.(7)` fidelity
- `C_smooth` 是一阶差分平方平均
- `C_curv` 是二阶差分平方平均

如果两个权重都设成 `0`，那就是纯 fidelity 优化。

如果权重非零，优化器会更偏向平滑脉冲。

## 15. 平滑正则到底在惩罚什么

当前平滑项不是某个抽象魔法，它们就是：

### 一阶差分项

如果相邻两个 slices 差很多：

```math
u_{k+1} - u_k
```

就会被惩罚。

这对应“脉冲不要跳得太猛”。

### 二阶差分项

如果脉冲的弯曲太大：

```math
u_{k+2} - 2u_{k+1} + u_k
```

就会被惩罚。

这对应“脉冲不要锯齿状抖动”。

当前这两个正则都是分别对：

- `ctrl_x`
- `ctrl_y`

做的，然后相加。

## 16. 为什么代码里还保留了完整 Liouvillian 的 reduced-channel 诊断

虽然当前主优化目标是 `Eq.(7)`，但代码里还留着：

- `channel_superoperator()`
- `channel_fidelity()`

等 reduced-channel helper。

它们现在的角色是：

- **诊断和验证**

而不是主优化目标。

保留它们的原因是：

- 可以用来做“完整 Liouvillian 下的 active 子空间一致性检查”
- 可以独立核对当前特殊态公式是不是在合理范围内

## 17. 当前 builder 和实验脚本如何组织

你真正会从外部调用的入口在：

- [yb171_calibration.py](../src/neutral_yb/config/yb171_calibration.py)

最关键的两个函数是：

- `build_yb171_v4_calibrated_model()`
- `build_yb171_v4_quasistatic_ensemble()`

### 17.1 `build_yb171_v4_calibrated_model()`

它返回：

- 单个名义模型

也就是：

- 一个 `Yb171ClockRydbergCZOpenModel`
- 带当前默认 `^171Yb` 参数
- 如果 `include_noise=True`，则带名义噪声

### 17.2 `build_yb171_v4_quasistatic_ensemble()`

它返回：

- 多个独立采样的准静态模型 realization

也就是把：

- UV detuning
- pulse-area scale
- blockade offset

按给定随机种子抽样后，得到一个 model 列表。

优化器会对这些 model 的 fidelity 做平均。

## 18. 为什么当前默认优化有时会退回零控制基线

当前优化器里有一个有意保留的 safeguard：

- 它会先算一遍“零控制时最好的 `theta`”
- 如果后续所有搜索到的候选脉冲都更差，就直接返回这个 baseline

这样做的目的不是偷懒，而是防止出现：

- 明明“什么都不做”都比你搜出来的脉冲好
- 结果程序却还硬把坏脉冲当最优解

在当前 `Eq.(7)` 指标下，零控制的最佳 fidelity 是：

```math
F = 0.6
```

所以如果你看到某些粗扫点正好卡在 `0.6`，往往意味着：

- 当前优化预算还没把它推离零脉冲 basin

而不是说明“这个时间点物理上就只能做到 `0.6`”。

## 19. 当前模型有哪些已知限制

要想正确使用当前 `v4`，必须清楚它还没有做什么。

### 19.1 没有显式优化 clock 脉冲

当前不优化：

- `130 us` 的 blackman-shaped clock pulses
- clock phase noise 在整个三段门中的时域传播

这些目前只通过参数和误差解释间接体现。

### 19.2 不是完整四维逻辑过程保真度

当前主 fidelity 是 `Eq.(7)` 特殊态公式。

它对当前门结构是合理的，但它仍然不是：

- 把完整 `|00>, |01>, |10>, |11>` noisy process matrix 全部显式建出来以后，再直接算 average gate fidelity

### 19.3 对 leakage 的处理是有效模型

当前的 `|leak>` 和 `|loss>` 都是粗粒度吸收/泄漏通道，不是实验里每一个具体副能级都展开出来了。

### 19.4 仍然是有效模型，不是 ab initio MQDT 动力学

`Peper 2025` 给的是：

- interaction potentials
- Rydberg manifold identification
- 状态选择依据

当前代码把这些结果折叠成了：

- 一个有效 blockade shift
- 一个更合理的目标 Rydberg manifold 解释

所以它是“实验导向的有效门模型”，不是从多通道量子缺陷理论一路直接传播到门脉冲的第一性原理模拟。

## 20. 现在如何判断这个模型是否“写对了”

当前仓库已经有一套专门的验证脚本：

- [validate_v4_dynamics_and_optimization.py](../experiments/validate_v4_dynamics_and_optimization.py)

它会检查：

1. 自己写的 Liouvillian 传播是否和 QuTiP 参考传播一致
2. `Eq.(7)` fidelity 在模型和优化器里的实现是否一致
3. 梯度是否和有限差分一致
4. 小规模优化是否至少能把目标函数往正确方向推
5. 当前参数量级是否还在合理物理区间

这是当前判断“代码逻辑是否自洽”的最重要入口。

## 21. 如果你第一次接手这个 `v4`，建议按什么顺序读源码

推荐顺序是：

1. [yb171_clock_rydberg_cz_open.py](../src/neutral_yb/models/yb171_clock_rydberg_cz_open.py)  
   先看模型基底、Hamiltonian、collapse operators

2. [yb171_calibration.py](../src/neutral_yb/config/yb171_calibration.py)  
   再看物理参数怎么变成内部无量纲参数

3. [open_system_grape.py](../src/neutral_yb/optimization/open_system_grape.py)  
   最后看优化器如何传播特殊态、如何求梯度、如何组织 ensemble

如果你只想先跑起来，可以直接看：

- [physical_time_scan_two_photon_cz_v4_open_system.py](../experiments/physical_time_scan_two_photon_cz_v4_open_system.py)

## 22. 一句话总结

当前 `v4` 是一个：

- **以 `^171Yb` 的 clock-to-Rydberg 门机制为物理背景**
- **以完整门序列为传播对象，并只把中间 UV 段作为优化变量**
- **同时包含 Rydberg decay、准静态失谐、blockade 偏差和 leakage 的开放系统有效模型**

它已经不再是旧的双光子 `ladder surrogate` 主模型，但也还不是把整个实验序列的全部微观细节都一口气展开的最终版本。

对于当前仓库的目标来说，它是：

- 一个物理口径正确得多
- 同时仍然足够快、足够可优化

的 `^171Yb v4` 主线模型。
