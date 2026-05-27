import numpy as np
import matplotlib.pyplot as plt

def cpg_sin(beta, phase, amp, base, length):
    """生成CPG轨迹"""
    peak_x = round(length * beta)
    shift_num = round(length * phase)
    x1 = np.linspace(0, 0.5, peak_x)
    traj_1 = -np.cos(2 * np.pi * x1)
    x2 = np.linspace(0.5, 1, length - peak_x)
    traj_2 = -np.cos(2 * np.pi * x2)
    traj = amp * np.append(traj_1, traj_2) + base
    return shift_array(traj, shift_num)

def shift_array(arr, n):
    """数组移位函数"""
    length = arr.shape[0]
    n = n % length
    if n == 0:
        shifted_array = arr.copy()
    else:
        shifted_array = np.concatenate((arr[-n:], arr[:-n]))
    return shifted_array

def plot_cpg_trajectory(beta=0.5, phase=0.0, amp=0.5, base=0.0, length=53):
    """
    绘制CPG轨迹
    
    参数:
    beta: 波形不对称性 (0.1-0.9)
    phase: 相位偏移 (0.0-1.0)
    amp: 幅值 (0.1-2.0)
    base: 基准位置 (-2.0-2.0)
    length: 轨迹长度
    """
    # 生成轨迹
    trajectory = cpg_sin(beta, phase, amp, base, length)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制轨迹
    x = np.arange(length)
    plt.plot(x, trajectory, 'b-', linewidth=2, label='CPG轨迹')
    plt.scatter(x, trajectory, color='red', s=30, zorder=5, alpha=0.7)
    
    # 设置图形属性
    plt.title(f'CPG轨迹 (beta={beta}, phase={phase}, amp={amp}, base={base})', fontsize=14)
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('关节位置', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    max_val = np.max(trajectory)
    min_val = np.min(trajectory)
    range_val = max_val - min_val
    
    stats_text = f'最大值: {max_val:.3f}\n最小值: {min_val:.3f}\n范围: {range_val:.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return trajectory

# 示例使用 - 只需修改这里的参数即可
if __name__ == "__main__":
    # 示例1: 躯干关节轨迹 (对称)
    print("示例1: 躯干关节轨迹")
    trajectory1 = plot_cpg_trajectory(beta=0.5, phase=0.0, amp=0.5, base=0.0)
    
    # 示例2: 后腿关节轨迹
    print("\n示例2: 后腿关节轨迹")
    trajectory2 = plot_cpg_trajectory(beta=0.4, phase=0.0, amp=0.5, base=-0.)
