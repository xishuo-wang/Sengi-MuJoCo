"""
单次仿真程序 - 使用参数搜索得到的最优参数
带可视化界面和地面接触力监测（聚焦XZ方向力分析）
"""

import mujoco_py
import numpy as np
import time
from datetime import datetime
from Double_Data_Visualization import DataLogger, show_visualization


# ==================== 配置参数 ====================
# 模型路径
MODEL_PATH = r"D:\Code\Model\Sengi\Sengi.xml"

# 数据保存路径
CSV_DATA_DIR = r"D:\Code\Sengi-MuJoCo\Sengi-double-MuJoCo/Data_Locomotion"
CSV_DATA_PREFIX = "Simulation_Data"

# 关节名称列表
JOINT_NAMES = [
    "back_joint", "hindleg_joint_1", "hindleg_joint_2", "hindleg_joint_3",
    "front_joint", "foreleg_joint_1", "foreleg_joint_2", "foreleg_joint_3"
]

# PD控制器参数
KP = 2
KD = 0.1
TOR = 0.5
TOR_SPINE = TOR * 1.5

# 固定运动参数
F = 3
A_SPINE = -0.5
A_LEGF_HIP = -0.25

# 最优参数
OPTIMAL_A_LEGH_HIP = 1.4
OPTIMAL_A_LEGH_KNEE = 0.8
OPTIMAL_PHASE_LAG = -np.pi / 20 * 0

# 仿真参数
INITIAL_PHASE_OFFSET = 0.55 / F
HOLD_TIME = 0.5
TOTAL_TIME = 5 + HOLD_TIME
HIGH_LEVEL_FREQ = 240
HIGH_LEVEL_DT = 1.0 / HIGH_LEVEL_FREQ

# 可视化参数
RENDER_EVERY = 1
VIEWER_WIDTH = 1200
VIEWER_HEIGHT = 800

# 接触力分析参数
ANALYSIS_START_TIME = HOLD_TIME
ANALYSIS_END_TIME = 5
IMPULSE_START = ANALYSIS_START_TIME + 0.1
IMPULSE_END = ANALYSIS_START_TIME + 0.2

# 数据记录参数
SAVE_DATA = False


# ==================== 轨迹生成函数 ====================
# 创建Sin轨迹参数矩阵
def create_sin_params(a_legH_hip, a_legH_knee, phase_lag):
    sin_params = np.array([
        [A_SPINE, F, 0.0, 0.0],                                    # back_joint
        [a_legH_hip, F, phase_lag, -0.6 + 0.7 * a_legH_hip],      # hindleg_joint_1
        [a_legH_knee, F, phase_lag, -1.5],                         # hindleg_joint_2
        [a_legH_knee, F, phase_lag, -1.5],                         # hindleg_joint_3
        [A_SPINE, F, 0.0, 0.0],                                    # front_joint
        [A_LEGF_HIP, F, 0.0, -0.6 + A_LEGF_HIP],                  # foreleg_joint_1
        [0., F, 0.0, -1.0],                                        # foreleg_joint_2
        [0., F, 0.0, -1.0],                                        # foreleg_joint_3
    ])
    return sin_params


# Sin轨迹生成
def get_sin_trajectory(t, params):
    A, f, phase, A0 = params[0], params[1], params[2], params[3]
    pos = A * np.sin(2 * np.pi * f * t + phase) + A0
    vel = A * 2 * np.pi * f * np.cos(2 * np.pi * f * t + phase)
    return pos, vel


# 批量生成所有关节的Sin轨迹
def get_all_joint_targets(t, sin_params, num_joints):
    target_pos = np.zeros(num_joints)
    target_vel = np.zeros(num_joints)
    for i in range(num_joints):
        target_pos[i], target_vel[i] = get_sin_trajectory(t, sin_params[i])
    return target_pos, target_vel


# PD控制器
def PDcontrol(target_pos, target_vel, current_pos, current_vel):
    pos_error = target_pos - current_pos
    vel_error = target_vel - current_vel
    torque = KP * pos_error + KD * vel_error

    num_joints = len(torque)
    torque_limits = np.ones(num_joints) * TOR

    back_joint_idx = JOINT_NAMES.index("back_joint")
    front_joint_idx = JOINT_NAMES.index("front_joint")
    torque_limits[back_joint_idx] = TOR_SPINE
    torque_limits[front_joint_idx] = TOR_SPINE

    torque = np.clip(torque, -torque_limits, torque_limits)
    return torque


# ==================== 接触力获取函数 ====================
# 获取指定几何体与地面的接触力
def get_contact_force(sim, geom_name):
    total_force = np.zeros(3)
    geom_id = sim.model.geom_name2id(geom_name)

    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]

        if contact.geom1 == geom_id or contact.geom2 == geom_id:
            force_in_contact_frame = np.zeros(6)
            mujoco_py.functions.mj_contactForce(sim.model, sim.data, i, force_in_contact_frame)

            contact_frame = contact.frame.reshape(3, 3)
            normal_force = contact_frame[:, 0] * force_in_contact_frame[0]
            tangential_force1 = contact_frame[:, 1] * force_in_contact_frame[1]
            tangential_force2 = contact_frame[:, 2] * force_in_contact_frame[2]

            force = normal_force + tangential_force1 + tangential_force2

            if contact.geom1 == geom_id:
                force = -force

            total_force += force

    total_force_magnitude = np.linalg.norm(total_force)
    return total_force, total_force_magnitude


# ==================== 文件路径工具 ====================
# 生成带时间戳的CSV文件名
def generate_csv_filename():
    import os
    os.makedirs(CSV_DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    filename = f"{CSV_DATA_PREFIX}_{timestamp}.csv"
    return os.path.join(CSV_DATA_DIR, filename)


# ==================== 主仿真函数 ====================
# 运行带可视化界面的单次仿真
def run_single_simulation_with_viewer(save_data=True):
    print("单次仿真 - 使用最优参数")
    print(f"a_legH_hip = {OPTIMAL_A_LEGH_HIP}")
    print(f"a_legH_knee = {OPTIMAL_A_LEGH_KNEE}")
    print(f"phase_lag = {OPTIMAL_PHASE_LAG:.4f} (≈{OPTIMAL_PHASE_LAG/np.pi:.2f}π)")
    print("=" * 50)

    # 加载模型
    model = mujoco_py.load_model_from_path(MODEL_PATH)
    sim = mujoco_py.MjSim(model)

    # 创建查看器
    viewer = mujoco_py.MjViewer(sim)
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
    initial_joint_pos = np.array([0, -0.5, -1.2, -1.2, 0, -0.5, -1.2, -1.2])
    sim.data.qpos[joint_pos_ids] = initial_joint_pos
    sim.data.qvel[joint_vel_ids] = np.zeros(len(joint_vel_ids))

    # 创建Sin参数矩阵
    sin_params = create_sin_params(OPTIMAL_A_LEGH_HIP, OPTIMAL_A_LEGH_KNEE, OPTIMAL_PHASE_LAG)

    # 初始化数据记录器
    data_logger = DataLogger(JOINT_NAMES)

    # 查找hindleg_3相关几何体
    hindleg_3_geom_names = []
    for i in range(model.ngeom):
        geom_name = model.geom_id2name(i)
        if geom_name and 'hindleg_3' in geom_name.lower():
            hindleg_3_geom_names.append(geom_name)

    if not hindleg_3_geom_names:
        hindleg_3_geom_names = ['hindleg_3_geom']

    # 运行仿真
    current_time = 0.0
    last_high_level_time = 0.0
    num_joints = len(JOINT_NAMES)

    high_level_target_pos, high_level_target_vel = get_all_joint_targets(
        INITIAL_PHASE_OFFSET, sin_params, num_joints
    )

    sim_step = 0
    start_time = time.time()
    force_record_interval = int(0.001 / dt)

    try:
        print(f"\n阶段1: 初始位置保持 {HOLD_TIME} 秒...")
        current_time = 0.0
        last_high_level_time = 0.0
        num_joints = len(JOINT_NAMES)

        # 保持阶段的目标位置（轨迹在偏移时间处的值）
        hold_target_pos, _ = get_all_joint_targets(INITIAL_PHASE_OFFSET, sin_params, num_joints)

        sim_step = 0
        start_time = time.time()
        force_record_interval = int(0.001 / dt)

        while current_time < TOTAL_TIME:
            current_pos = sim.data.qpos[joint_pos_ids].copy()
            current_vel = sim.data.qvel[joint_vel_ids].copy()

            # 判断当前阶段
            if current_time < HOLD_TIME:
                # 阶段1: 保持静止
                target_pos = hold_target_pos
                target_vel = np.zeros(num_joints)
            else:
                # 阶段2: 正常运动
                if current_time - last_high_level_time >= HIGH_LEVEL_DT - 1e-9 or current_time == HOLD_TIME:
                    last_high_level_time = current_time
                    motion_time = current_time - HOLD_TIME + INITIAL_PHASE_OFFSET
                    target_pos, target_vel = get_all_joint_targets(
                        motion_time, sin_params, num_joints
                    )

            # 高层控制更新时的数据记录
            if current_time - last_high_level_time >= HIGH_LEVEL_DT - 1e-9 or current_time == 0:
                data_logger.record_high_level_data(
                    current_time, target_pos, current_pos, current_vel
                )
                last_high_level_time = current_time

            # 应用PD控制
            torque = PDcontrol(target_pos, target_vel, current_pos, current_vel)
            for i in range(min(len(torque), sim.model.nu)):
                sim.data.ctrl[i] = torque[i]

            # 记录数据
            data_logger.record_torque_data(current_time, torque)

            current_base_x = sim.data.body_xpos[base_link_id][0]
            data_logger.record_base_pos(current_time, current_base_x)

            # 记录接触力
            if sim_step % force_record_interval == 0:
                for geom_name in hindleg_3_geom_names:
                    try:
                        force, force_mag = get_contact_force(sim, geom_name)
                        data_logger.record_contact_force(geom_name, current_time, force, force_mag)
                    except Exception:
                        pass

            # 渲染
            if sim_step % RENDER_EVERY == 0:
                viewer.render()

            sim.step()
            current_time += dt
            sim_step += 1

    except KeyboardInterrupt:
        print("\n用户中断仿真")
    except Exception as e:
        print(f"\n仿真出错: {e}")
        import traceback
        traceback.print_exc()

    elapsed_time = time.time() - start_time

    # 计算平均速度
    avg_velocity = data_logger.calculate_average_velocity(HOLD_TIME + 1.0, HOLD_TIME + 4.0)

    # 打印结果
    print("\n" + "=" * 50)
    print("仿真完成")
    print(f"总前进距离: {data_logger.get_total_distance():.4f} m")
    print(f"平均速度 (1s-4s): {avg_velocity:.4f} m/s")
    print(f"仿真耗时: {elapsed_time:.2f} s")
    print("=" * 50)

    # 保存数据
    if save_data:
        csv_filename = generate_csv_filename()
        print(f"数据已保存: {csv_filename}")
        data_logger.save_to_csv(csv_filename)

    return avg_velocity, data_logger


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    avg_speed, data_logger = run_single_simulation_with_viewer(save_data=SAVE_DATA)

    print(f"\n最终结果: 平均速度 = {avg_speed:.4f} m/s")

    print("\n正在启动数据可视化界面...")
    show_visualization(data_logger, JOINT_NAMES, OPTIMAL_PHASE_LAG, TOTAL_TIME,
                      ANALYSIS_START_TIME, ANALYSIS_END_TIME, IMPULSE_START, IMPULSE_END, avg_speed)