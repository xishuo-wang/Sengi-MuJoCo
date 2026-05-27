"""
单次仿真程序 - 使用参数搜索得到的最优参数
带可视化界面和地面接触力监测（聚焦XZ方向力分析）
"""

import mujoco_py
import numpy as np
import time
import matplotlib.pyplot as plt

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
TOR_SPINE = 0.6

# 固定运动参数（与参数搜索完全一致）
F = 3                       # 频率
A_SPINE = -0.5              # 脊柱振幅
A_LEGF_HIP = -0.25          # 前腿髋关节振幅（固定）

# 最优参数（从参数搜索结果中获取）
OPTIMAL_A_LEGH_HIP = 1.0   # 后腿髋关节振幅
OPTIMAL_A_LEGH_KNEE = 0.2   # 后腿膝关节振幅
OPTIMAL_PHASE_LAG = np.pi/20*4     # 相位滞后

# 仿真参数
TOTAL_TIME = 10.0           # 总仿真时间
HIGH_LEVEL_FREQ = 240       # Hz（与参数搜索一致）
HIGH_LEVEL_DT = 1.0 / HIGH_LEVEL_FREQ

# 可视化参数
RENDER_EVERY = 1            # 每帧都渲染
VIEWER_WIDTH = 1200         # 查看器窗口宽度
VIEWER_HEIGHT = 800         # 查看器窗口高度

# 接触力分析参数
ANALYSIS_START_TIME = 0.0   # 分析开始时间（秒）
ANALYSIS_END_TIME = 2.0     # 分析结束时间（秒）

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

def PDcontrol(target_pos, target_vel, current_pos, current_vel, joint_indices=None):
    """PD控制器 - 支持不同关节的不同力矩限制"""
    pos_error = target_pos - current_pos
    vel_error = target_vel - current_vel
    torque = KP * pos_error + KD * vel_error
    
    # 为不同关节设置不同的力矩限制
    num_joints = len(torque)
    torque_limits = np.ones(num_joints) * TOR  # 默认力矩限制
    
    # 为back_joint和front_joint设置特殊力矩限制
    back_joint_idx = JOINT_NAMES.index("spine_hind_joint")  # 0
    
    torque_limits[back_joint_idx] = TOR_SPINE
    
    # 应用不同的力矩限制
    torque = np.clip(torque, -torque_limits, torque_limits)
    
    return torque

# ==================== 接触力获取函数 ====================
def get_contact_force(sim, geom_name):
    """
    获取指定几何体与地面的接触力
    
    参数:
    - sim: MuJoCo仿真对象
    - geom_name: 几何体名称（如 'hindleg_3_geom'）
    
    返回:
    - total_force: 总接触力向量 [fx, fy, fz]
    - total_force_magnitude: 总接触力大小
    """
    total_force = np.zeros(3)
    
    # 获取几何体ID
    geom_id = sim.model.geom_name2id(geom_name)
    
    # 遍历所有接触点
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        
        # 检查是否涉及该几何体
        if contact.geom1 == geom_id or contact.geom2 == geom_id:
            # 获取接触力（在接触坐标系中）
            force_in_contact_frame = np.zeros(6)
            mujoco_py.functions.mj_contactForce(sim.model, sim.data, i, force_in_contact_frame)
            
            # 接触力在世界坐标系中的表示
            contact_frame = contact.frame.reshape(3, 3)
            
            # 法向力（contact.frame的第一列是法向）
            normal_force = contact_frame[:, 0] * force_in_contact_frame[0]
            
            # 切向力（contact.frame的第二和第三列是切向）
            tangential_force1 = contact_frame[:, 1] * force_in_contact_frame[1]
            tangential_force2 = contact_frame[:, 2] * force_in_contact_frame[2]
            
            # 判断力的方向（确保作用在几何体上的力方向正确）
            if contact.geom1 == geom_id:
                # geom1受力，力方向需要取反
                force = -(normal_force + tangential_force1 + tangential_force2)
            else:
                force = normal_force + tangential_force1 + tangential_force2
            
            total_force += force
    
    total_force_magnitude = np.linalg.norm(total_force)
    return total_force, total_force_magnitude

# ==================== 主仿真函数 ====================
def run_single_simulation_with_viewer():
    """运行带可视化界面的单次仿真"""
    
    print("=" * 60)
    print("单次仿真 - 使用最优参数（带可视化界面和XZ方向力分析）")
    print(f"a_legH_hip = {OPTIMAL_A_LEGH_HIP}")
    print(f"a_legH_knee = {OPTIMAL_A_LEGH_KNEE}")
    print(f"phase_lag = {OPTIMAL_PHASE_LAG:.4f} (≈{OPTIMAL_PHASE_LAG/np.pi:.2f}π)")
    print(f"仿真时间: {TOTAL_TIME} 秒")
    print(f"接触力分析时段: {ANALYSIS_START_TIME}-{ANALYSIS_END_TIME} 秒")
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
    
    # 获取hindleg_3相关的几何体名称
    print("\n查找hindleg_3相关几何体...")
    hindleg_3_geom_names = []
    for i in range(model.ngeom):
        geom_name = model.geom_id2name(i)
        if geom_name and 'hindleg_3' in geom_name.lower():
            hindleg_3_geom_names.append(geom_name)
            print(f"  找到几何体: {geom_name}")
    
    if not hindleg_3_geom_names:
        print("警告: 未找到与 'hindleg_3' 相关的几何体")
        print("可用的几何体名称:")
        for i in range(model.ngeom):
            geom_name = model.geom_id2name(i)
            if geom_name:
                print(f"  {geom_name}")
        print("\n尝试使用几何体名称 'hindleg_3_geom'（如果不存在请修改代码）")
        hindleg_3_geom_names = ['hindleg_3_geom']
    
    # 记录数据
    base_x_positions = []
    times = []
    
    # 接触力数据存储
    contact_forces = {name: {'time': [], 'fx': [], 'fy': [], 'fz': [], 'magnitude': []} 
                     for name in hindleg_3_geom_names}
    
    # 运行仿真
    print("\n开始仿真 - 可视化窗口已打开...")
    print("正在监测接触力数据（重点关注X和Z方向）...")
    current_time = 0.0
    last_high_level_time = 0.0
    num_joints = len(JOINT_NAMES)
    
    # 初始化目标位置
    high_level_target_pos, high_level_target_vel = get_all_joint_targets(0, sin_params, num_joints)
    
    # 仿真循环
    sim_step = 0
    start_time = time.time()
    last_print_time = 0
    force_record_interval = int(0.001 / dt)  # 每1ms记录一次接触力
    
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
            
            # 记录接触力数据（按间隔记录以节省内存）
            if sim_step % force_record_interval == 0:
                for geom_name in hindleg_3_geom_names:
                    try:
                        force, force_mag = get_contact_force(sim, geom_name)
                        contact_forces[geom_name]['time'].append(current_time)
                        contact_forces[geom_name]['fx'].append(force[0])
                        contact_forces[geom_name]['fy'].append(force[1])
                        contact_forces[geom_name]['fz'].append(force[2])
                        contact_forces[geom_name]['magnitude'].append(force_mag)
                    except Exception as e:
                        print(f"  获取接触力时出错 ({geom_name}): {e}")
            
            # 渲染可视化
            if sim_step % RENDER_EVERY == 0:
                viewer.render()
            
            # 仿真步进
            sim.step()
            current_time += dt
            sim_step += 1
            
            # 打印进度
            if current_time - last_print_time >= 1.0:
                progress = current_time / TOTAL_TIME * 100
                # 显示最近的接触力信息
                latest_forces = {}
                for geom_name in hindleg_3_geom_names:
                    if contact_forces[geom_name]['magnitude']:
                        latest_forces[geom_name] = contact_forces[geom_name]['magnitude'][-1]
                
                force_str = ", ".join([f"{name}: {force:.2f}N" for name, force in latest_forces.items()])
                print(f"  进度: {progress:.1f}% | 时间: {current_time:.1f}s | "
                      f"位置: {current_base_x:.3f}m | 接触力 - {force_str}")
                last_print_time = current_time
                
    except KeyboardInterrupt:
        print("\n\n用户中断仿真")
    except Exception as e:
        print(f"\n\n仿真出错: {e}")
        import traceback
        traceback.print_exc()
    
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
    
    # ==================== 绘制XZ方向力分析图 ====================
    plot_xz_force_analysis(contact_forces, avg_velocity)
    
    return avg_velocity

# 在 plot_xz_force_analysis 函数开始处添加冲量计算
def plot_xz_force_analysis(contact_forces, avg_velocity):
    """
    绘制4-5秒稳定阶段XZ方向地面反作用力分析图
    
    包含：
    1. X和Z方向力随时间变化
    2. XZ合力方向随时间变化
    3. XZ合力大小随时间变化
    """
    
    print(f"\n正在生成XZ方向力分析图表（{ANALYSIS_START_TIME}-{ANALYSIS_END_TIME}秒）...")
    
    # ==================== 计算指定时间段的冲量 ====================
    impulse_start = ANALYSIS_START_TIME + 0.05
    impulse_end = ANALYSIS_START_TIME + 0.2
    
    print(f"\n{'='*60}")
    print(f"冲量计算 (Impulse Calculation)")
    print(f"时间区间: {impulse_start:.2f}s - {impulse_end:.2f}s")
    print(f"持续时间: {impulse_end - impulse_start:.4f}s")
    print(f"{'='*60}")
    
    for geom_name, forces in contact_forces.items():
        if not forces['time']:
            print(f"警告: {geom_name} 没有接触力数据")
            continue
        
        # 转换为numpy数组
        times_array = np.array(forces['time'])
        fx_array = np.array(forces['fx'])
        fz_array = np.array(forces['fz'])
        
        # 选择指定时间段的数据
        mask = (times_array >= impulse_start) & (times_array <= impulse_end)
        
        if np.sum(mask) < 2:
            print(f"警告: {geom_name} 在{impulse_start:.2f}-{impulse_end:.2f}秒数据点不足")
            continue
        
        impulse_times = times_array[mask]
        impulse_fx = fx_array[mask]
        impulse_fz = fz_array[mask]
        
        # 计算时间步长（使用梯形法则进行积分）
        time_steps = np.diff(impulse_times)
        
        # 使用梯形法则计算冲量
        # Fx冲量
        fx_mid = (impulse_fx[:-1] + impulse_fx[1:]) / 2
        impulse_x = np.sum(fx_mid * time_steps)
        
        # Fz冲量
        fz_mid = (impulse_fz[:-1] + impulse_fz[1:]) / 2
        impulse_z = np.sum(fz_mid * time_steps)
        
        # 合冲量
        impulse_magnitude = np.sqrt(impulse_x**2 + impulse_z**2)
        
        # 计算平均力（用于验证）
        avg_fx_interval = np.mean(impulse_fx)
        avg_fz_interval = np.mean(impulse_fz)
        
        print(f"\n{geom_name}:")
        print(f"  {'─'*50}")
        print(f"  Fx方向冲量 (水平):")
        print(f"    Ix = {impulse_x:.6f} N·s")
        print(f"    平均Fx = {avg_fx_interval:.4f} N")
        print(f"    数据点数 = {len(impulse_fx)}")
        print(f"  {'─'*50}")
        print(f"  Fz方向冲量 (垂直):")
        print(f"    Iz = {impulse_z:.6f} N·s")
        print(f"    平均Fz = {avg_fz_interval:.4f} N")
        print(f"    数据点数 = {len(impulse_fz)}")
        print(f"  {'─'*50}")
        print(f"  合冲量大小: |I| = {impulse_magnitude:.6f} N·s")
        print(f"  冲量方向角度: {np.degrees(np.arctan2(impulse_z, impulse_x)):.2f}°")
        
        # 额外分析：分段冲量（更细的时间分段）
        num_segments = 5
        segment_duration = (impulse_end - impulse_start) / num_segments
        
        print(f"\n  细分时间段冲量分析 ({num_segments}段, 每段{segment_duration*1000:.1f}ms):")
        print(f"  {'时间段':<20} {'Ix (N·s)':<12} {'Iz (N·s)':<12} {'|I| (N·s)':<12}")
        print(f"  {'─'*56}")
        
        for i in range(num_segments):
            seg_start = impulse_start + i * segment_duration
            seg_end = seg_start + segment_duration
            
            seg_mask = (times_array >= seg_start) & (times_array < seg_end)
            
            if np.sum(seg_mask) < 2:
                continue
            
            seg_times = times_array[seg_mask]
            seg_fx = fx_array[seg_mask]
            seg_fz = fz_array[seg_mask]
            
            seg_time_steps = np.diff(seg_times)
            seg_fx_mid = (seg_fx[:-1] + seg_fx[1:]) / 2
            seg_fz_mid = (seg_fz[:-1] + seg_fz[1:]) / 2
            
            seg_impulse_x = np.sum(seg_fx_mid * seg_time_steps)
            seg_impulse_z = np.sum(seg_fz_mid * seg_time_steps)
            seg_impulse_mag = np.sqrt(seg_impulse_x**2 + seg_impulse_z**2)
            
            print(f"  {seg_start:.3f}-{seg_end:.3f}s    "
                  f"{seg_impulse_x:+.6f}    {seg_impulse_z:+.6f}    {seg_impulse_mag:.6f}")
        
        print(f"  {'─'*56}")
        print(f"  {'总计':<20} {impulse_x:+.6f}    {impulse_z:+.6f}    {impulse_magnitude:.6f}")
        
        # 只处理第一个几何体
        break
    
    print(f"\n{'='*60}\n")
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    
    # 提取4-5秒的数据
    for geom_name, forces in contact_forces.items():
        if not forces['time']:
            print(f"警告: {geom_name} 没有接触力数据")
            continue
        
        # 转换为numpy数组
        times_array = np.array(forces['time'])
        fx_array = np.array(forces['fx'])
        fz_array = np.array(forces['fz'])
        
        # 选择4-5秒时间段的数据
        mask = (times_array >= ANALYSIS_START_TIME) & (times_array <= ANALYSIS_END_TIME)
        
        if np.sum(mask) == 0:
            print(f"警告: {geom_name} 在{ANALYSIS_START_TIME}-{ANALYSIS_END_TIME}秒没有数据")
            continue
        
        analysis_times = times_array[mask] - ANALYSIS_START_TIME  # 相对于4秒的时间
        analysis_fx = fx_array[mask]
        analysis_fz = -fz_array[mask]
        
        # 计算XZ合力
        analysis_fxz_magnitude = np.sqrt(analysis_fx**2 + analysis_fz**2)
        # 计算合力方向（与水平面的夹角，度）
        analysis_fxz_angle = np.degrees(np.arctan2(analysis_fz, analysis_fx))
        
        # 统计信息
        max_fx = np.max(np.abs(analysis_fx))
        max_fz = np.max(np.abs(analysis_fz))
        max_fxz = np.max(analysis_fxz_magnitude)
        mean_fx = np.mean(analysis_fx)
        mean_fz = np.mean(analysis_fz)
        mean_fxz = np.mean(analysis_fxz_magnitude)
        
        print(f"\n{geom_name} XZ方向力统计 ({ANALYSIS_START_TIME}-{ANALYSIS_END_TIME}秒):")
        print(f"  Fx: 最大值={max_fx:.2f}N, 平均值={mean_fx:.2f}N")
        print(f"  Fz: 最大值={max_fz:.2f}N, 平均值={mean_fz:.2f}N")
        print(f"  Fxz合力: 最大值={max_fxz:.2f}N, 平均值={mean_fxz:.2f}N")
        
        # ===== 子图1: X和Z方向力随时间变化（增强版，标注冲量计算区间） =====
        ax1 = plt.subplot(3, 1, 1)
        
        # 绘制Fx
        ax1.plot(analysis_times, analysis_fx, 'b-', 
                label=f'{geom_name} Fx (水平力)', linewidth=2, alpha=0.8)
        # 绘制Fz
        ax1.plot(analysis_times, analysis_fz, 'r-', 
                label=f'{geom_name} Fz (垂直力)', linewidth=2, alpha=0.8)
        
        # 高亮冲量计算区间
        impulse_start_rel = impulse_start - ANALYSIS_START_TIME
        impulse_end_rel = impulse_end - ANALYSIS_START_TIME
        ax1.axvspan(impulse_start_rel, impulse_end_rel, alpha=0.2, color='yellow', 
                   label=f'Impulse region\n({impulse_start:.2f}-{impulse_end:.2f}s)')
        
        # 添加零线
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        ax1.set_xlabel('Time (s) [relative to 4s]', fontsize=12)
        ax1.set_ylabel('Force (N)', fontsize=12)
        ax1.set_title(f'X and Z Direction Ground Reaction Forces\n'
                     f'(Yellow region: Impulse calculation zone)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 添加文本标注（包含冲量信息）
        ax1.text(0.02, 0.98, 
                f'Max |Fx|: {max_fx:.2f}N\n'
                f'Max |Fz|: {max_fz:.2f}N\n'
                f'Ix = {impulse_x:.4f} N·s\n'
                f'Iz = {impulse_z:.4f} N·s', 
                transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # [其余子图代码保持不变...]
        # ===== 子图2和子图3的代码保持不变 =====
        
        # 只处理第一个找到的几何体
        break
    
    # [其余代码保持不变...]
    plt.tight_layout()
    plt.savefig('xz_force_analysis.png', dpi=150, bbox_inches='tight')
    print("\nXZ方向力分析图已保存为 'xz_force_analysis.png'")
    plt.show()

# ==================== 主程序入口 ====================
if __name__ == '__main__':
    # 运行带可视化的仿真
    avg_speed = run_single_simulation_with_viewer()
    
    print(f"\n最终结果: 平均速度 = {avg_speed:.4f} m/s")
    print("\n可视化窗口已关闭。")
    print("接触力数据已记录并可视化完成。")