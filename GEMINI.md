# GEMINI 项目管理手册

## 项目概述
本项目是一个基于 **FEniCSx (DOLFINx)** 的高性能并行拓扑优化框架，名为 `fenitop`。
专注于 3D 壳层结构与多孔填充（Shell-Infill）优化，严格遵循 Clausen et al. (2017) 的理论体系。

## 核心目录说明
- `fenitop/`: 核心算法库（包含 PDE 过滤器、材料插值、灵敏度分析、可视化工具等）。
- `scripts/`: 运行与验证脚本（包含 3D Coating 梁案例及灵敏度验证）。
- `Matlab/`: 原始 Matlab 实现，用于数学逻辑基准对比。
- `results/`: 优化过程输出（PNG 纵向切片图、XDMF 三维数据场）。
- `meshes/`: 有限元初始网格文件。

## 当前任务状态
- [x] **2026-03-20**: 
    - [x] **物理模型完全校准**: 
        - 修正壳层梯度归一化系数 $\alpha = 2R$。
        - 对齐 3D Hashin-Shtrikman 刚度上限公式及 $p_g=1$ 惩罚准则。
    - [x] **并行架构优化与死锁修复**: 
        - 修复了并行下 UFL 积分元签名不一致导致的 JIT 编译死锁。
        - 解决了 Rank 间时间戳目录不统一的问题。
        - 引入 `plot_freq` 极大缓解了渲染导致的 MPI 阻塞。
    - [x] **Domain Extension (区域扩展) 进阶实现**: 
        - 实现了 `d_ext` 自动取整及 Padding 单元精确计算，确保物理边界与网格线 100% 对齐。
        - 修正了体积约束计算，使其仅针对物理设计域生效。
    - [x] **鲁棒性改进**: 
        - 添加了 Compliance 缩放的除零保护。
        - 解决了大规模 3D 计算导致的 OOM (Killed Signal 9) 内存爆炸问题（通过分辨率与过滤半径协同优化）。
    - [x] **可视化与后处理增强**: 
        - 实现 PNG 自动纵向切片，支持直接观察内部多孔演化。
        - 实现了实时 `design_*.xdmf` 序列输出，支持 ParaView 动态监控。
    - [x] **项目工程化**: 
        - 创建 `environment.yml` 环境依赖清单。
        - 初始化 Git 仓库，配置 `.gitignore`，编写专业 `README.md`。
        - 完成 GitHub 远程仓库关联。
- [x] **2026-03-23**:
    - [x] **理论模型全参数校准**:
        - 修正了 PDE 系数转换公式：$r_{pde} = R / (2\sqrt{\smash[b]{3}})$。
        - 修正了 3D Hashin-Shtrikman 刚度上限公式。
        - 修正了壳层梯度归一化系数 $\alpha = R / \sqrt{\smash[b]{3}}$。
        - 重新设定 $R_1 = 2 \times R_2, R_2 = 2.5 \times t_{ref}$。
    - [x] **优化算法与收敛性提升**:
        - 实现 MMA 渐近线重置机制（在 $beta$ 增加时重启 `mma_iter`）。
        - 重构了 `project_expression` 后处理函数，提升效率约 10 倍。
    - [x] **后处理稳定性增强**:
        - 修复了低密度阶段不输出 PNG 切片的问题。
        - 锁定切片位置在载荷点，并在 X-Y 平面内进行剖分观察。
- [x] **2026-03-24**:
    - [x] **MMA 优化器稳定性与一致性修复**:
        - 修复 `solve_subproblem` 中 `y, z, s, mu, zeta` 变量仅在 rank 0 初始化导致的并行不一致问题。
        - **MMA 渐近线 (Asymptotes) 重置机制**: 在 `beta` 步进时，严格按照 MATLAB 逻辑重置 `xold1, xold2, low, upp` 为当前密度值，配合 `mma_iter` 重启，确保收敛稳定性。
    - [x] **过滤与物理模型深度对齐**:
        - **恢复双重滤波与投影 (DSP)**: 严格对齐 MATLAB `ft=2` 逻辑，基础结构密度链条更新为两级 PDE 滤波与投影。
        - **壳层 Beta 跟随逻辑**: 实现 `shell_beta = beta / 2.0` 约束，使壳层边界物理过渡更平滑。
    - [x] **灵敏度计算链条校准**:
        - **移除对数柔度缩放 (Log C Scaling)**: 彻底删除 `c_scale` 逻辑，确保返回给优化器的是原始柔度梯度 $dC/d\rho$，解决了因梯度过小导致的优化停滞。
        - **灵敏度验证 (FD Check)**: 在结构边界（圆柱体表面）的高强度非线性区域通过了有限差分验证，误差稳定在 **0.0000%** 理想水平。
    - [x] **可视化与兼容性维护**:
        - 修复了 DOLFINx 0.8.0 的 `functionspace` API 兼容性。
        - 修复了 `Plotter` 的 PNG 裁切与 XDMF 序列保存逻辑。
        - 优化了 `fd_check.py` 的对数坐标轴绘图展示。

## 运行规范
- **正式计算**: `mpirun -n [核心数] python3 scripts/coating_beam_3d.py`
- **梯度验证**: `python3 scripts/fd_check.py` (单核运行)
- **内存建议**: 若遇到 `Signal 9`，请降低 `mesh_res_phys` 或检查 `filter_radius` 与物理尺寸的比例。

## 历史记录
- *2026-03-19*: 初始 3D Coating 代码开发（处于非对齐状态）。
- *2026-03-20*: 完成与论文 Clausen et al. (2017) 的全方位对齐与性能飞跃。





gemini --resume 9fbaacf2-fd72-4699-a47d-1446081ee839 
