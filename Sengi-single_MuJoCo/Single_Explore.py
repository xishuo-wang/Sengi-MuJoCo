"""
批量参数搜索脚本 - 搜索腿部运动参数对速度和高度的影响 (固定线程数并行版本)
搜索参数: a_legH_hip, a_legH_knee, phase_lag
"""

import mujoco_py
import numpy as np
import pandas as pd
import time
import os
from itertools import product
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
# 模型路径
MODEL_PATH = r"D:\Code\Model\Sengi_simple_single\Sengi_simple_single.xml"

# 数据保存路径
CSV_DATA_DIR = r"D:\Code\Sengi-MuJoCo\Sengi-single_MuJoCo\Data_Explore"
CSV_DATA_PREFIX = "Explore_Data"

# 关节名称列表
JOINT_NAMES = [
    "spine_hind_joint", "hindleg_1_joint", "hindleg_2_joint", "hindleg_3_joint",
    "foreleg_1_joint", "foreleg_2_joint", "foreleg_3_joint"
]

# PD控制器参数
KP = 2
KD = 0.1
TOR = 0.5
TOR_SPINE = TOR * 1.5

# 固定运动参数
F = 2.2
A_SPINE = -0.7
A_LEGF_HIP = -0.25

# 仿真参数
INITIAL_PHASE_OFFSET = 0.5 / F  
HOLD_TIME = 0.5
TOTAL_TIME = 4 + HOLD_TIME
HIGH_LEVEL_FREQ = 240
HIGH_LEVEL_DT = 1.0 / HIGH_LEVEL_FREQ

# 并行配置
FIXED_NUM_THREADS = 8

# 参数搜索范围
A_LEGH_HIP_RANGE = np.arange(0, 1.41, 0.1)
A_LEGH_KNEE_RANGE = np.arange(0, 1.01, 0.1)
# PHASE_LAG_RANGE = np.arange(-np.pi, np.pi + np.pi/20, np.pi/20)
PHASE_LAG_RANGE = np.arange(-np.pi/2, np.pi/2 + np.pi/20, np.pi/20)


# ==================== 工具函数 ====================
# 生成带时间戳的CSV文件名
def generate_csv_filename():
    os.makedirs(CSV_DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    filename = f"{CSV_DATA_PREFIX}_{timestamp}.csv"
    return os.path.join(CSV_DATA_DIR, filename)


# 创建Sin轨迹参数矩阵
def create_sin_params(a_legH_hip, a_legH_knee, phase_lag):
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
def PDcontrol(target_pos, target_vel, current_pos, current_vel, joint_indices=None):
    pos_error = target_pos - current_pos
    vel_error = target_vel - current_vel
    torque = KP * pos_error + KD * vel_error

    num_joints = len(torque)
    torque_limits = np.ones(num_joints) * TOR  # 默认力矩限制
    
    back_joint_idx = JOINT_NAMES.index("spine_hind_joint")  # 0
    
    torque_limits[back_joint_idx] = TOR_SPINE

    torque = np.clip(torque, -torque_limits, torque_limits)
    return torque


# ==================== 单次仿真函数 ====================
# 运行单次仿真（供并行调用）
def run_single_simulation(args):
    a_hip_h, a_knee_h, phase_lag, idx = args

    try:
        # 加载模型
        model = mujoco_py.load_model_from_path(MODEL_PATH)
        sim = mujoco_py.MjSim(model)

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

        # 创建Sin参数矩阵
        sin_params = create_sin_params(a_hip_h, a_knee_h, phase_lag)

        # 记录质心位置
        base_x_positions = []
        base_z_positions = []
        times = []
        max_pitch_abs = 0.0  # 记录最大绝对俯仰角

        current_time = 0.0
        last_high_level_time = -HIGH_LEVEL_DT  # 确保首次进入时触发记录
        num_joints = len(JOINT_NAMES)

        # 初始化目标变量
        hold_target_pos = initial_joint_pos.copy()
        target_pos = hold_target_pos
        target_vel = np.zeros(num_joints)
        high_level_target_pos = hold_target_pos
        high_level_target_vel = np.zeros(num_joints)

        while current_time < TOTAL_TIME:
            current_pos = sim.data.qpos[joint_pos_ids].copy()
            current_vel = sim.data.qvel[joint_vel_ids].copy()

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

                    # 获取俯仰角并更新最大值
                    base_rotmat = sim.data.body_xmat[base_link_id].reshape(3, 3)
                    pitch = np.arctan2(-base_rotmat[2, 0],
                                       np.sqrt(base_rotmat[0, 0]**2 + base_rotmat[1, 0]**2))
                    pitch_abs = abs(np.degrees(pitch))
                    if pitch_abs > max_pitch_abs:
                        max_pitch_abs = pitch_abs

            # 应用PD控制
            torque = PDcontrol(target_pos, target_vel, current_pos, current_vel)
            for i in range(min(len(torque), sim.model.nu)):
                sim.data.ctrl[i] = torque[i]

            # 记录基座位置
            current_base_x = sim.data.body_xpos[base_link_id][0]
            current_base_z = sim.data.body_xpos[base_link_id][2]
            base_x_positions.append(current_base_x)
            base_z_positions.append(current_base_z)
            times.append(current_time)

            sim.step()
            current_time += dt

        # 计算平均速度（1s-4s）
        if len(times) > 1:
            idx_1s = np.argmin(np.abs(np.array(times) - 1.0 - HOLD_TIME))
            idx_4s = np.argmin(np.abs(np.array(times) - 4.0 - HOLD_TIME))

            pos_1s = base_x_positions[idx_1s]
            pos_4s = base_x_positions[idx_4s]
            time_diff = times[idx_4s] - times[idx_1s]

            avg_velocity = (pos_4s - pos_1s) / time_diff if time_diff > 0 else 0.0
        else:
            avg_velocity = 0.0

        max_height = max(base_z_positions) if base_z_positions else 0.0

        return {
            'a_legH_hip': a_hip_h,
            'a_legH_knee': a_knee_h,
            'phase_lag': phase_lag,
            'avg_velocity': avg_velocity,
            'max_height': max_height,
            'max_pitch': max_pitch_abs,
            'success': True,
            'index': idx
        }

    except Exception:
        return {
            'a_legH_hip': a_hip_h,
            'a_legH_knee': a_knee_h,
            'phase_lag': phase_lag,
            'avg_velocity': 0.0,
            'max_height': 0.0,
            'max_pitch': 180.0,
            'success': False,
            'index': idx
        }
    

# ==================== 主程序入口 ====================
if __name__ == '__main__':
    mp.freeze_support()

    # 预计算所有参数组合
    param_combinations = list(product(A_LEGH_HIP_RANGE, A_LEGH_KNEE_RANGE, PHASE_LAG_RANGE))
    total_combinations = len(param_combinations)

    print("=" * 60)
    print("参数搜索范围:")
    print(f"a_legH_hip: {len(A_LEGH_HIP_RANGE)}个值, 范围: [{A_LEGH_HIP_RANGE[0]:.2f}, {A_LEGH_HIP_RANGE[-1]:.2f}]")
    print(f"a_legH_knee: {len(A_LEGH_KNEE_RANGE)}个值, 范围: [{A_LEGH_KNEE_RANGE[0]:.2f}, {A_LEGH_KNEE_RANGE[-1]:.2f}]")
    print(f"phase_lag: {len(PHASE_LAG_RANGE)}个值, 范围: [{PHASE_LAG_RANGE[0]:.2f}, {PHASE_LAG_RANGE[-1]:.2f}]")
    print(f"总计参数组合: {total_combinations}")

    # 检查CPU核心数
    cpu_count = mp.cpu_count()
    print(f"系统CPU核心数: {cpu_count}")
    print(f"固定使用线程数: {FIXED_NUM_THREADS}")

    if FIXED_NUM_THREADS > cpu_count:
        print(f"警告: 设置的线程数({FIXED_NUM_THREADS})超过CPU核心数({cpu_count})，可能影响性能")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            exit()

    print("=" * 60)
    print("\n开始参数搜索（固定线程数并行）...")
    print("=" * 60)

    start_time = time.time()

    # 准备参数列表
    param_list = [(a_hip_h, a_knee_h, phase_lag, i)
                  for i, (a_hip_h, a_knee_h, phase_lag) in enumerate(param_combinations)]

    # 创建进程池
    pool = mp.Pool(processes=FIXED_NUM_THREADS)

    all_results = []
    try:
        with tqdm(total=total_combinations, desc="仿真进度", ncols=80) as pbar:
            for result in pool.imap_unordered(run_single_simulation, param_list):
                all_results.append(result)
                pbar.update(1)

    except KeyboardInterrupt:
        print("\n用户中断，正在终止进程...")
        pool.terminate()
        pool.join()
        exit()
    finally:
        pool.close()
        pool.join()

    # 按索引排序结果
    all_results.sort(key=lambda x: x['index'])

    # 过滤成功的结果
    results = [r for r in all_results if r['success']]

    # 筛选俯仰角不超过75°
    PITCH_LIMIT = 75.0
    results_filtered = [r for r in results if r['max_pitch'] <= PITCH_LIMIT]
    filtered_count = len(results) - len(results_filtered)

    # 计算耗时
    elapsed_time = time.time() - start_time

    # 保存结果
    if results:
        df_results = pd.DataFrame(results_filtered)

        if 'index' in df_results.columns:
            df_results = df_results.drop('index', axis=1)

        df_results_sorted_by_speed = df_results.sort_values('avg_velocity', ascending=False)

        # 生成带时间戳的文件名并保存
        output_filename = generate_csv_filename()
        df_results.to_csv(output_filename, index=False)

        print("\n" + "=" * 60)
        print(f"总耗时: {elapsed_time:.2f} 秒")
        print(f"成功完成的仿真数: {len(results)}/{total_combinations}")
        print(f"结果已保存到: {output_filename}")
        print("=" * 60)

        # 显示最佳参数组合
        print("\n" + "=" * 60)
        print("按速度排序的前10名最佳参数组合:")
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(df_results_sorted_by_speed.head(10).to_string(index=False))

        # 性能统计
        print("\n" + "=" * 60)
        print("性能统计:")
        print(f"平均每个仿真耗时: {elapsed_time/len(results):.3f} 秒")
        print(f"平均每秒处理: {len(results)/elapsed_time:.1f} 个仿真")

    else:
        print("\n没有成功完成的仿真！")