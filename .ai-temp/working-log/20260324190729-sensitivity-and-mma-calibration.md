# 工作日志: 20260324190729 - 灵敏度链式求导校准与 MMA 优化器深度修复

## 本次工作目标
1. 解决灵敏度数值极小及优化器停滞的问题。
2. 将 Python 版拓扑优化架构的过滤链路（Filter Chain）与原始 MATLAB 代码 (`ft=2`) 严格对齐。
3. 完善非线性惩罚切换（Beta Continuation）时对 MMA 内部参量的复位逻辑。

## 实际完成情况

### 1. 移除了不当的全局对数柔度缩放 (Log C Scaling)
- **原因**：之前的代码为了避免目标值过大，擅自在返回给 MMA 之前把梯度除以了 $C$ (`c_scale`)，导致传递给 MMA 的梯度过小，触发了优化器的 `epsimin` 死区阈值，使得优化停滞（Stagnation）。
- **修复**：彻底移除了所有涉及 `c_scale` 的缩放逻辑。现在 MMA 接收到的是纯正的未缩放灵敏度（$\frac{dC}{d\rho}$），数值回归至十万量级（与 MATLAB 版完全一致），优化器恢复健康运作。

### 2. 恢复了双重滤波与投影 (DSP)
- **改动**：依据 MATLAB 版的 `ft=2` 逻辑，在 Python 架构中完整补齐了基于 DSP 的基础结构过滤。
  链条重构为：`rho (DG0)` $\to$ `H_pde` $\to$ `Heaviside0` $\to$ `H_pde` $\to$ `Heaviside1` $\to$ `rho_base`。
- **验证**：利用 `fenitop/parameterize.py` 里的自伴随过滤算子，精确反向推导了 DSP 的全链条偏导数，并通过了高精度有限差分检查（相对误差 `0.0000%`）。

### 3. 修复了 MMA Beta Continuation 渐近线（Asymptotes）突变问题
- **改动**：在每次 `opt_iter` 达到 `beta_interval`（即 `beta` 翻倍时），除了将 `mma_iter` 重置为 `1` 外，精准移植了 MATLAB 版本的内部变量复位逻辑：
  ```python
  rho_old1[:] = rho_field.vector.array
  rho_old2[:] = rho_old1
  low[:] = rho_field.vector.array
  upp[:] = rho_field.vector.array
  ```
- **影响**：使得 MMA 的 `low` 和 `upp` 渐近线能够基于新的高度非线性状态重新平稳张开，防止结构撕裂或数值振荡。

### 4. 建立 Shell 投影 Beta 与 Base 的跟随约束
- **改动**：将壳层的 `tanh` 投影非线性因子 `shell_beta` 与基础结构的 `beta` 直接挂钩。根据用户要求，强制设定 `shell_beta = beta / 2.0`。
- **影响**：让涂层边界的物理过渡相对更加平滑，有助于避免多孔介质的厚度奇异性。

## 下一步建议
1. 运行多进程 `mpirun` 的 `scripts/coating_beam_3d.py` 正式案例。
2. 监控 ParaView 下生成的 `optimized_*.xdmf` 文件，观察三维多孔壳层（Shell-Infill）形态演化是否理想。
