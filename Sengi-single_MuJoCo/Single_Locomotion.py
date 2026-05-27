"""
单次仿真程序 - 使用参数搜索得到的最优参数
带可视化界面
"""

import mujoco_py
import numpy as np
import time

# ==================== 配置参数 ====================
# 模型路径（与参数搜索保持一致）
MODEL_PATH = r"D:\Code\Model\Sengi_simple_single\Sengi_simple_single.xml"

# 定义关节名称列表（全局常量）
JOINT_NAMES = [
    "spine_hind_joint", "hindleg_1_joint", "hindleg_2_joint", "hindleg_3_joint",
    "foreleg_1_joint", "foreleg_2_joint", "foreleg_3_joint"
]

# PD控制器参数（与参数搜索完全一致）
KP = 2
KD = 0.1
TOR = 0.5  # 扭矩限制

# 固定运动参数（与参数搜索完全一致）
F = 3                       # 频率
A_SPINE = -0.5              # 脊柱振幅
A_LEGF_HIP = -0.25          # 前腿髋关节振幅（固定）

# 最优参数（从参数搜索结果中获取）
OPTIMAL_A_LEGH_HIP = 1.3   # 后腿髋关节振幅
OPTIMAL_A_LEGH_KNEE = 0.1   # 后腿膝关节振幅
OPTIMAL_PHASE_LAG = np.pi/20*7     # 相位滞后

# 仿真参数
TOTAL_TIME = 10.0           # 总仿真时间（增加到10秒以便观察）
HIGH_LEVEL_FREQ = 240       # Hz（与参数搜索一致）
HIGH_LEVEL_DT = 1.0 / HIGH_LEVEL_FREQ

# 可视化参数
RENDER_EVERY = 1            # 每帧都渲染（可以改为更大的数以加快仿真速度）
VIEWER_WIDTH = 1200         # 查看器窗口宽度
VIEWER_HEIGHT = 800         # 查看器窗口高度

# ==================== 轨迹生成函数 ====================
def create_sin_params(a_legH_hip, a_legH_knee, phase_lag):
    """
    根据给定参数创建Sin轨迹参数矩阵
    """
    sin_params = np.array([
        [A_SPINE, F, 0.0, 0.0],                                    # back_joint
        [a_legH_hip, F, phase_lag, -0.6 + 0.7 * a_legH_hip],      # hindleg_joint_1
        [a_legH_knee, F, phase_lag, -1.5],                         # hindleg_joint_2
        [a_legH_knee, F, phase_lag, -1.5],                         # hindleg_joint_3
        [A_LEGF_HIP, F, 0.0, -0.6 + A_LEGF_HIP],                  # foreleg_joint_1
        [0., F, 0.0, -1.0],                                        # foreleg_joint_2
        [0., F, 0.0, -1.0],                                        # foreleg_joint_3
    ])
    return sin_params

def get_sin_trajectory(t, params):
    """Sin轨迹生成函数"""
    A = params[0]
    f = params[1]
    phase = params[2]
    A0 = params[3]
    
    pos = A * np.sin(2 * np.pi * f * t + phase) + A0
    vel = A * 2 * np.pi * f * np.cos(2 * np.pi * f * t + phase)
    
    return pos, vel

def get_all_joint_targets(t, sin_params, num_joints):
    """批量生成所有关节的Sin轨迹"""
    target_pos = np.zeros(num_joints)
    target_vel = np.zeros(num_joints)
    
    for i in range(num_joints):
        target_pos[i], target_vel[i] = get_sin_trajectory(t, sin_params[i])
    
    return target_pos, target_vel

def PDcontrol(target_pos, target_vel, current_pos, current_vel):
    """PD控制器"""
    pos_error = target_pos - current_pos
    vel_error = target_vel - current_vel
    torque = KP * pos_error + KD * vel_error
    torque = np.clip(torque, -TOR, TOR)
    return torque

# ==================== 主仿真函数 ====================
def run_single_simulation_with_viewer():
    """运行带可视化界面的单次仿真"""
    
    print("=" * 60)
    print("单次仿真 - 使用最优参数（带可视化界面）")
    print(f"a_legH_hip = {OPTIMAL_A_LEGH_HIP}")
    print(f"a_legH_knee = {OPTIMAL_A_LEGH_KNEE}")
    print(f"phase_lag = {OPTIMAL_PHASE_LAG:.4f} (≈{OPTIMAL_PHASE_LAG/np.pi:.2f}π)")
    print(f"仿真时间: {TOTAL_TIME} 秒")
    print("=" * 60)
    print("\n可视化控制说明:")
    print("  - 鼠标左键拖动: 旋转视角")
    print("  - 鼠标右键拖动: 平移视角")
    print("  - 滚动滚轮: 缩放")
    print("  - 按 'ESC' 或关闭窗口退出仿真")
    print("=" * 60)
    
    # 加载模型
    print("\n加载模型中...")
    model = mujoco_py.load_model_from_path(MODEL_PATH)
    sim = mujoco_py.MjSim(model)
    
    # 创建查看器
    viewer = mujoco_py.MjViewer(sim)
    
    # 设置查看器窗口大小
    if hasattr(viewer, 'window'):
        viewer.window.width = VIEWER_WIDTH
        viewer.window.height = VIEWER_HEIGHT
    
    # 获取关节ID
    joint_pos_ids = []
    joint_vel_ids = []
    
    for name in JOINT_NAMES:
        joint_id = model.joint_name2id(name)
        joint_pos_ids.append(model.jnt_qposadr[joint_id])
        joint_vel_ids.append(model.jnt_dofadr[joint_id])
    
    base_link_id = model.body_name2id('base_link')
    dt = model.opt.timestep
    
    # 设置初始关节位置
    initial_joint_pos = np.array([0, -0.5, -1.2, -1.2, -0.5, -1.2, -1.2])
    sim.data.qpos[joint_pos_ids] = initial_joint_pos
    sim.data.qvel[joint_vel_ids] = np.zeros(len(joint_vel_ids))
    
    # 创建Sin参数矩阵
    sin_params = create_sin_params(OPTIMAL_A_LEGH_HIP, OPTIMAL_A_LEGH_KNEE, OPTIMAL_PHASE_LAG)
    
    # 打印参数信息
    print("\nSin轨迹参数设置:")
    print("-" * 70)
    print(f"{'关节名称':<20} {'振幅A':<8} {'频率f':<8} {'相位':<10} {'偏置A0':<8}")
    print("-" * 70)
    for i, name in enumerate(JOINT_NAMES):
        print(f"{name:<20} {sin_params[i,0]:<8.2f} {sin_params[i,1]:<8.2f} "
              f"{sin_params[i,2]:<10.4f} {sin_params[i,3]:<8.2f}")
    print("=" * 70)
    
    # 记录数据
    base_x_positions = []
    times = []
    
    # 运行仿真
    print("\n开始仿真 - 可视化窗口已打开...")
    current_time = 0.0
    last_high_level_time = 0.0
    num_joints = len(JOINT_NAMES)
    
    # 初始化目标位置
    high_level_target_pos, high_level_target_vel = get_all_joint_targets(0, sin_params, num_joints)
    
    # 仿真循环
    sim_step = 0
    start_time = time.time()
    last_print_time = 0
    
    try:
        while current_time < TOTAL_TIME:
            # 获取当前状态
            current_pos = sim.data.qpos[joint_pos_ids].copy()
            current_vel = sim.data.qvel[joint_vel_ids].copy()
            
            # 高层控制更新
            if current_time - last_high_level_time >= HIGH_LEVEL_DT - 1e-9 or current_time == 0:
                last_high_level_time = current_time
                high_level_target_pos, high_level_target_vel = get_all_joint_targets(
                    current_time, sin_params, num_joints
                )
            
            # 应用PD控制
            torque = PDcontrol(high_level_target_pos, high_level_target_vel, current_pos, current_vel)
            for i in range(min(len(torque), sim.model.nu)):
                sim.data.ctrl[i] = torque[i]
            
            # 记录基座位置
            current_base_x = sim.data.body_xpos[base_link_id][0]
            base_x_positions.append(current_base_x)
            times.append(current_time)
            
            # 渲染可视化（每RENDER_EVERY步渲染一次）
            if sim_step % RENDER_EVERY == 0:
                viewer.render()
            
            # 仿真步进
            sim.step()
            current_time += dt
            sim_step += 1
            
            # 打印进度（每秒打印一次）
            if current_time - last_print_time >= 1.0:
                progress = current_time / TOTAL_TIME * 100
                current_speed = (current_base_x - base_x_positions[-int(1/dt)]) if len(base_x_positions) > int(1/dt) else 0
                print(f"  进度: {progress:.1f}% | 时间: {current_time:.1f}s | 当前位置: {current_base_x:.3f}m")
                last_print_time = current_time
                
    except KeyboardInterrupt:
        print("\n\n用户中断仿真")
    except Exception as e:
        print(f"\n\n仿真出错: {e}")
    
    elapsed_time = time.time() - start_time
    
    # ==================== 计算平均速度 ====================
    if len(times) > 1:
        idx_1s = np.argmin(np.abs(np.array(times) - 1.0))
        idx_4s = np.argmin(np.abs(np.array(times) - 4.0))
        
        pos_1s = base_x_positions[idx_1s]
        pos_4s = base_x_positions[idx_4s]
        time_diff = times[idx_4s] - times[idx_1s]
        
        avg_velocity = (pos_4s - pos_1s) / time_diff if time_diff > 0 else 0.0
        total_distance = base_x_positions[-1] - base_x_positions[0]
    else:
        avg_velocity = 0.0
        total_distance = 0.0
    
    # ==================== 打印结果 ====================
    print("\n" + "=" * 60)
    print("仿真完成！")
    print("=" * 60)
    print(f"仿真参数:")
    print(f"  总仿真时间: {TOTAL_TIME:.1f} 秒")
    print(f"  仿真步数: {sim_step}")
    print(f"  实际计算时间: {elapsed_time:.2f} 秒")
    print(f"  仿真/实时比: {TOTAL_TIME/elapsed_time:.1f}x")
    print("-" * 60)
    print(f"运动结果:")
    print(f"  初始X位置: {base_x_positions[0]:.4f} 米")
    print(f"  最终X位置: {base_x_positions[-1]:.4f} 米")
    print(f"  总前进距离: {total_distance:.4f} 米")
    print(f"  平均速度 (1s-4s): {avg_velocity:.4f} 米/秒")
    print("=" * 60)
    
    return avg_velocity

# ==================== 主程序入口 ====================
if __name__ == '__main__':
    # 运行带可视化的仿真
    avg_speed = run_single_simulation_with_viewer()
    
    print(f"\n最终结果: 平均速度 = {avg_speed:.4f} m/s")
    print("\n可视化窗口已关闭。")