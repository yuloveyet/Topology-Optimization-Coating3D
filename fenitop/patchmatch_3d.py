import numpy as np
import numba as nb
from numba import njit, prange
import math

@njit
def trilinear_interpolate(vol, z, y, x):
    """Trilinear interpolation for 3D volume."""
    D, H, W = vol.shape
    
    # Coordinates inside the volume
    z0 = int(math.floor(z))
    y0 = int(math.floor(y))
    x0 = int(math.floor(x))
    
    z1 = z0 + 1
    y1 = y0 + 1
    x1 = x0 + 1
    
    # Check boundaries
    if z0 < 0 or z1 >= D or y0 < 0 or y1 >= H or x0 < 0 or x1 >= W:
        return 0.0 # Out of bounds fill value
        
    zd = z - z0
    yd = y - y0
    xd = x - x0
    
    # Interpolate along x
    c00 = vol[z0, y0, x0] * (1 - xd) + vol[z0, y0, x1] * xd
    c01 = vol[z0, y1, x0] * (1 - xd) + vol[z0, y1, x1] * xd
    c10 = vol[z1, y0, x0] * (1 - xd) + vol[z1, y0, x1] * xd
    c11 = vol[z1, y1, x0] * (1 - xd) + vol[z1, y1, x1] * xd
    
    # Interpolate along y
    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd
    
    # Interpolate along z
    c = c0 * (1 - zd) + c1 * zd
    return c

@njit
def get_rotation_matrix(alpha, beta, gamma):
    """Create a 3D rotation matrix from Euler angles (in degrees)."""
    a = math.radians(alpha)
    b = math.radians(beta)
    g = math.radians(gamma)
    
    # Rz(gamma) * Ry(beta) * Rx(alpha)
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cg, sg = math.cos(g), math.sin(g)
    
    R = np.zeros((3, 3))
    R[0, 0] = cb * cg
    R[0, 1] = cg * sa * sb - ca * sg
    R[0, 2] = ca * cg * sb + sa * sg
    
    R[1, 0] = cb * sg
    R[1, 1] = ca * cg + sa * sb * sg
    R[1, 2] = -cg * sa + ca * sb * sg
    
    R[2, 0] = -sb
    R[2, 1] = cb * sa
    R[2, 2] = ca * cb
    
    return R

@njit
def patch_distance_rotate3(volA, volB, posA, posB, alpha, beta, gamma, patch_size):
    """Compute SSD between a patch in volA and a rotated patch in volB."""
    D, H, W = volA.shape
    d = patch_size // 2
    
    za, ya, xa = posA
    zb, yb, xb = posB
    
    R = get_rotation_matrix(alpha, beta, gamma)
    
    dist = 0.0
    valid_pixels = 0
    
    for dz in range(-d, d + 1):
        for dy in range(-d, d + 1):
            for dx in range(-d, d + 1):
                # A coords
                az = za + dz
                ay = ya + dy
                ax = xa + dx
                
                # Check A bounds
                if az < 0 or az >= D or ay < 0 or ay >= H or ax < 0 or ax >= W:
                    continue
                
                valA = volA[int(az), int(ay), int(ax)]
                
                # B coords (rotate the offset)
                rot_dz = R[0, 0] * dz + R[0, 1] * dy + R[0, 2] * dx
                rot_dy = R[1, 0] * dz + R[1, 1] * dy + R[1, 2] * dx
                rot_dx = R[2, 0] * dz + R[2, 1] * dy + R[2, 2] * dx
                
                bz = zb + rot_dz
                by = yb + rot_dy
                bx = xb + rot_dx
                
                valB = trilinear_interpolate(volB, bz, by, bx)
                
                dist += (valA - valB) ** 2
                valid_pixels += 1
                
    if valid_pixels == 0:
        return np.inf
    
    return dist / valid_pixels

@njit(parallel=True)
def patchmatch3d_rotate3axis(volA, volB, iterations=5, patch_size=5, angle_range=(-15.0, 15.0), walk_levels=5):
    """
    PatchMatch 3D with 6-DOF (translation + rotation).
    volA, volB: 3D numpy arrays
    """
    D, H, W = volA.shape
    
    offsets = np.zeros((D, H, W, 6), dtype=np.float64)
    costs = np.full((D, H, W), np.inf, dtype=np.float64)
    
    min_angle, max_angle = angle_range
    max_spatial_step = max(D, H, W) / 2.0
    max_angle_step = (max_angle - min_angle) / 2.0
    
    # 1. Random Initialization
    for z in prange(D):
        for y in range(H):
            for x in range(W):
                offsets[z, y, x, 0] = np.random.randint(0, D)
                offsets[z, y, x, 1] = np.random.randint(0, H)
                offsets[z, y, x, 2] = np.random.randint(0, W)
                offsets[z, y, x, 3] = min_angle + np.random.random() * (max_angle - min_angle)
                offsets[z, y, x, 4] = min_angle + np.random.random() * (max_angle - min_angle)
                offsets[z, y, x, 5] = min_angle + np.random.random() * (max_angle - min_angle)
                
                costs[z, y, x] = patch_distance_rotate3(
                    volA, volB, 
                    (int(z), int(y), int(x)), 
                    (int(offsets[z, y, x, 0]), int(offsets[z, y, x, 1]), int(offsets[z, y, x, 2])),
                    offsets[z, y, x, 3], offsets[z, y, x, 4], offsets[z, y, x, 5], 
                    patch_size
                )
                
    # 2. PatchMatch Iterations
    for it in range(iterations):
        # Determine propagation direction
        z_step = 1 if it % 2 == 0 else -1
        y_start, y_end, y_step = (0, H, 1) if it % 2 == 0 else (H - 1, -1, -1)
        x_start, x_end, x_step = (0, W, 1) if it % 2 == 0 else (W - 1, -1, -1)
        
        # We process Z sequentially based on iteration direction to avoid
        # complex dynamic bounds inside prange which break Numba's lowering.
        # But we can parallelize over Y instead!
        
        z_start, z_end = (0, D) if it % 2 == 0 else (D - 1, -1)
        
        for z in range(z_start, z_end, z_step):
            for y in prange(H):
                # Apply the correct sweep direction for Y
                actual_y = int(y) if it % 2 == 0 else int(H - 1 - y)
                for x in range(x_start, x_end, x_step):
                    
                    z_idx = int(z)
                    y_idx = int(actual_y)
                    x_idx = int(x)
                    
                    cur_z = offsets[z_idx, y_idx, x_idx, 0]
                    cur_y = offsets[z_idx, y_idx, x_idx, 1]
                    cur_x = offsets[z_idx, y_idx, x_idx, 2]
                    cur_alpha = offsets[z_idx, y_idx, x_idx, 3]
                    cur_beta = offsets[z_idx, y_idx, x_idx, 4]
                    cur_gamma = offsets[z_idx, y_idx, x_idx, 5]
                    cur_cost = costs[z_idx, y_idx, x_idx]
                    
                    # Propagation
                    for n_idx in range(3):
                        if n_idx == 0:
                            nz, ny, nx = z_idx - z_step, y_idx, x_idx
                        elif n_idx == 1:
                            nz, ny, nx = z_idx, y_idx - y_step, x_idx
                        else:
                            nz, ny, nx = z_idx, y_idx, x_idx - x_step
                            
                        if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                            nz_idx = int(nz)
                            ny_idx = int(ny)
                            nx_idx = int(nx)
                            
                            cand_z = offsets[nz_idx, ny_idx, nx_idx, 0]
                            cand_y = offsets[nz_idx, ny_idx, nx_idx, 1]
                            cand_x = offsets[nz_idx, ny_idx, nx_idx, 2]
                            cand_alpha = offsets[nz_idx, ny_idx, nx_idx, 3]
                            cand_beta = offsets[nz_idx, ny_idx, nx_idx, 4]
                            cand_gamma = offsets[nz_idx, ny_idx, nx_idx, 5]
                            
                            # Standard PatchMatch spatial shift using rotation
                            d_z = float(z_idx - nz_idx)
                            d_y = float(y_idx - ny_idx)
                            d_x = float(x_idx - nx_idx)
                            
                            R = get_rotation_matrix(cand_alpha, cand_beta, cand_gamma)
                            
                            rot_dz = R[0, 0] * d_z + R[0, 1] * d_y + R[0, 2] * d_x
                            rot_dy = R[1, 0] * d_z + R[1, 1] * d_y + R[1, 2] * d_x
                            rot_dx = R[2, 0] * d_z + R[2, 1] * d_y + R[2, 2] * d_x
                            
                            prop_z = min(max(cand_z + rot_dz, 0.0), float(D - 1))
                            prop_y = min(max(cand_y + rot_dy, 0.0), float(H - 1))
                            prop_x = min(max(cand_x + rot_dx, 0.0), float(W - 1))
                            
                            c = patch_distance_rotate3(
                                volA, volB, 
                                (z_idx, y_idx, x_idx), 
                                (prop_z, prop_y, prop_x),
                                cand_alpha, cand_beta, cand_gamma, 
                                patch_size
                            )
                            if c < cur_cost:
                                cur_z, cur_y, cur_x = prop_z, prop_y, prop_x
                                cur_alpha, cur_beta, cur_gamma = cand_alpha, cand_beta, cand_gamma
                                cur_cost = c
                                
                    # Random Walk Search (Coarse-to-fine)
                    for level in range(walk_levels):
                        spatial_step = max(1.0, max_spatial_step / (2 ** level))
                        angle_step = max(0.5, max_angle_step / (2 ** level))
                        
                        dz = cur_z + (np.random.random() * 2.0 - 1.0) * spatial_step
                        dy = cur_y + (np.random.random() * 2.0 - 1.0) * spatial_step
                        dx = cur_x + (np.random.random() * 2.0 - 1.0) * spatial_step
                        
                        dz = min(max(dz, 0.0), float(D - 1))
                        dy = min(max(dy, 0.0), float(H - 1))
                        dx = min(max(dx, 0.0), float(W - 1))
                        
                        alpha = cur_alpha + (np.random.random() * 2.0 - 1.0) * angle_step
                        beta  = cur_beta + (np.random.random() * 2.0 - 1.0) * angle_step
                        gamma = cur_gamma + (np.random.random() * 2.0 - 1.0) * angle_step
                        
                        alpha = min(max(alpha, float(min_angle)), float(max_angle))
                        beta  = min(max(beta, float(min_angle)), float(max_angle))
                        gamma = min(max(gamma, float(min_angle)), float(max_angle))
                        
                        c = patch_distance_rotate3(
                            volA, volB, 
                            (int(z), actual_y, int(x)), 
                            (dz, dy, dx),
                            alpha, beta, gamma, 
                            patch_size
                        )
                        
                        if c < cur_cost:
                            cur_z, cur_y, cur_x = dz, dy, dx
                            cur_alpha, cur_beta, cur_gamma = alpha, beta, gamma
                            cur_cost = c
                            
                    offsets[z_idx, y_idx, x_idx, 0] = cur_z
                    offsets[z_idx, y_idx, x_idx, 1] = cur_y
                    offsets[z_idx, y_idx, x_idx, 2] = cur_x
                    offsets[z_idx, y_idx, x_idx, 3] = cur_alpha
                    offsets[z_idx, y_idx, x_idx, 4] = cur_beta
                    offsets[z_idx, y_idx, x_idx, 5] = cur_gamma
                    costs[z_idx, y_idx, x_idx] = cur_cost

    return offsets, costs

if __name__ == "__main__":
    import time
    print("Testing 3D PatchMatch with Rotation...")
    A = np.random.rand(16, 16, 16)
    B = np.random.rand(16, 16, 16)
    
    print("Warming up JIT compiler...")
    start = time.time()
    offsets, costs = patchmatch3d_rotate3axis(A, B, iterations=1, patch_size=3, walk_levels=2)
    print(f"Compilation and Warmup took {time.time() - start:.2f}s")
    
    print("Running full benchmark...")
    start = time.time()
    offsets, costs = patchmatch3d_rotate3axis(A, B, iterations=3, patch_size=5, angle_range=(-20.0, 20.0), walk_levels=5)
    print(f"Execution took {time.time() - start:.2f}s")
    
    min_idx = np.argmin(costs)
    z_min, y_min, x_min = np.unravel_index(min_idx, costs.shape)
    print(f"Minimum cost: {costs[z_min, y_min, x_min]:.4f} at ({z_min}, {y_min}, {x_min})")
    print(f"Best offset/angles: {offsets[z_min, y_min, x_min]}")
