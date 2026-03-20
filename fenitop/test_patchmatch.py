import numpy as np
import time
from scipy.ndimage import shift, rotate
from patchmatch_3d import patchmatch3d_rotate3axis

def create_test_volume(shape=(20, 20, 20)):
    """创建一个内部带有特征梯度的测试三维体，以保证每个patch都是唯一的"""
    vol = np.zeros(shape)
    # 在中心放置一个结构块
    vol[5:15, 5:15, 5:15] = 1.0
    
    # 叠加微小的空间梯度，使得每个体素的邻域都有唯一的特征
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    vol += 0.1 * z + 0.01 * y + 0.001 * x
    return vol

def test_identity():
    print("========== 测试 1: 恒等匹配 (Identity) ==========")
    print("设定: volA 与 volB 完全相同。")
    volA = create_test_volume()
    volB = volA.copy()
    
    # 运行 PatchMatch
    offsets, costs = patchmatch3d_rotate3axis(volA, volB, iterations=3, patch_size=5, angle_range=(-5.0, 5.0), walk_levels=3)
    
    # 检查中心点 (避免受边界截断的影响)
    cz, cy, cx = 10, 10, 10
    best_offset = offsets[cz, cy, cx]
    
    print(f"中心点 ({cz}, {cy}, {cx}) 的最小匹配误差(Cost): {costs[cz, cy, cx]:.6f}")
    print(f"找到的目标坐标与角度 (z, y, x, alpha, beta, gamma):")
    print(f"[{best_offset[0]:.2f}, {best_offset[1]:.2f}, {best_offset[2]:.2f}, {best_offset[3]:.2f}, {best_offset[4]:.2f}, {best_offset[5]:.2f}]")
    print(f"预期目标坐标: ({cz}, {cy}, {cx})，预期角度: (0, 0, 0)\n")

def test_translation():
    print("========== 测试 2: 刚体平移 (Translation) ==========")
    volA = create_test_volume()
    
    # 将 volA 平移 (2, -3, 1) 得到 volB
    shift_vec = (2, -3, 1)
    print(f"设定: volB 是 volA 平移了 {shift_vec} 的结果。")
    volB = shift(volA, shift_vec, order=1, mode='constant', cval=0.0)
    
    offsets, costs = patchmatch3d_rotate3axis(volA, volB, iterations=4, patch_size=5, angle_range=(-5.0, 5.0), walk_levels=4)
    
    cz, cy, cx = 10, 10, 10
    best_offset = offsets[cz, cy, cx]
    
    print(f"中心点 ({cz}, {cy}, {cx}) 的最小匹配误差(Cost): {costs[cz, cy, cx]:.6f}")
    
    mapped_z, mapped_y, mapped_x = best_offset[0], best_offset[1], best_offset[2]
    print(f"找到的目标坐标: ({mapped_z:.2f}, {mapped_y:.2f}, {mapped_x:.2f})")
    print(f"预期的目标坐标: ({cz + shift_vec[0]}, {cy + shift_vec[1]}, {cx + shift_vec[2]})\n")

if __name__ == '__main__':
    # 预热 Numba JIT 编译器
    print("正在预热 JIT 编译器...")
    _ = patchmatch3d_rotate3axis(np.random.rand(8,8,8), np.random.rand(8,8,8), iterations=1, patch_size=3)
    print("预热完成。\n")
    
    test_identity()
    test_translation()