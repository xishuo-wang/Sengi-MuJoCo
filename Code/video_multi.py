"""
批量视频录制程序 - 扫描相位参数并录制机器人运动视频
支持慢速录制功能
"""

import mujoco_py
import numpy as np
import cv2
import time
import os
from tqdm import tqdm

# ==================== 配置参数 ====================
# 模型路径
MODEL_PATH = r"D:\Code\Model\Sengi\Sengi-video.xml"

# 视频输出配置
VIDEO_OUTPUT_ROOT = r"D:\Code\Sengi-MuJoCo\Video-Matlab\BatchScan"
VIDEO_FPS = 30                                # 视频帧率
VIDEO_RESOLUTION = (3700, 2080)               # 视频分辨率

# 慢速录制配置
SLOW_MOTION_FACTOR = 0.5                      # 慢速因子：0.5表示半速，1.0表示正常速度
ENABLE_SLOW_MOTION = True                      # 是否启用慢速录制

# 运动参数
TOTAL_TIME = 4                                 # 仿真时间 (秒)
F = 3                                          # 运动频率
A_SPINE = -0.5                                 # 脊柱振幅
A_LEGF_HIP = -0.25                             # 前腿髋关节振幅固定

# PD控制器参数
KP = 2
KD = 0.1
TOR = 0.4

# 相位扫描范围（从 -pi 到 pi，步长 pi/20）
PHASE_LAG_RANGE = np.arange(-np.pi/10, np.pi/10, np.pi/20)
n_phases = len(PHASE_LAG_RANGE)

# 固定振幅参数
A_LEGH_HIP = 0.5    # 后腿髋关节振幅
A_LEGH_KNEE = 0.2   # 后腿膝关节振幅

# 定义关节名称列表
JOINT_NAMES = [
    "back_joint", "hindleg_joint_1", "hindleg_joint_2", "hindleg_joint_3",
    "front_joint", "foreleg_joint_1", "foreleg_joint_2", "foreleg_joint_3"
]

print("=" * 60)
print("批量视频录制 (支持慢速录制)")
print(f"相位扫描范围: -π 到 π, 步长 π/20")
print(f"相位数量: {n_phases}")
print(f"仿真时间: {TOTAL_TIME}秒")
print(f"固定参数: 髋={A_LEGH_HIP}, 膝={A_LEGH_KNEE}")
print(f"慢速录制: {'开启' if ENABLE_SLOW_MOTION else '关闭'}")
if ENABLE_SLOW_MOTION:
    print(f"慢速因子: {SLOW_MOTION_FACTOR}x (视频时长: {TOTAL_TIME/SLOW_MOTION_FACTOR:.1f}秒)")
print(f"视频输出: {VIDEO_OUTPUT_ROOT}")
print("=" * 60)

# 创建输出文件夹（带时间戳）
timestamp = time.strftime('%Y%m%d_%H%M%S')
slow_tag = f"_slow{SLOW_MOTION_FACTOR}" if ENABLE_SLOW_MOTION else ""
output_folder = os.path.join(VIDEO_OUTPUT_ROOT, f'PhaseScan_{timestamp}{slow_tag}')
os.makedirs(output_folder, exist_ok=True)
print(f"输出文件夹: {output_folder}")


# ==================== 颜色配置函数 ====================
def setup_phase_colors():
    """
    设置相位相关的颜色映射（从蓝色到红色渐变）
    蓝色 (#12c2e9) → 红色 (#f64f59)
    """
    # 起始颜色和结束颜色 (RGB 0-1范围)
    startColor = np.array([18, 194, 233]) / 255.0   # 蓝色
    endColor = np.array([246, 79, 89]) / 255.0      # 红色

    # 相位范围从 -pi 到 pi
    phase_range_rad = np.arange(-np.pi, np.pi + np.pi/20, np.pi/20)
    n_phases = len(phase_range_rad)

    # 生成颜色的渐变
    Colors = np.zeros((n_phases, 3))
    for i in range(3):
        Colors[:, i] = np.linspace(startColor[i], endColor[i], n_phases)

    # 创建相位到颜色的映射
    phase_range_deg = phase_range_rad * 180 / np.pi
    phase_colors = {}
    for i in range(n_phases):
        phase_key = int(round(phase_range_deg[i]))
        phase_colors[phase_key] = Colors[i, :]

    return phase_colors


def get_color_from_phase(phase_rad, phase_colors):
    """
    根据相位值获取对应的颜色
    """
    phase_deg = phase_rad * 180 / np.pi
    phase_rounded = int(round(phase_deg))

    if phase_rounded in phase_colors:
        return phase_colors[phase_rounded].copy()
    else:
        # 找最近的相位
        all_phase_keys = np.array(list(phase_colors.keys()))
        nearest_idx = np.argmin(np.abs(all_phase_keys - phase_rounded))
        nearest_key = all_phase_keys[nearest_idx]
        return phase_colors[nearest_key].copy()


def set_robot_color(sim, phase_rad, phase_colors):
    """
    根据相位设置整个机器人的颜色（直接通过几何体名称）
    """
    # 获取当前相位对应的颜色
    base_color = get_color_from_phase(phase_rad, phase_colors)
    rgba = np.array([base_color[0], base_color[1], base_color[2], 1.0])

    # 几何体名称列表
    geom_names = [
        'spine_back_visual_link',
        'spine_front_visual_link',
        'hindleg_1_visual_link',
        'hindleg_2_visual_link',
        'hindleg_3_visual_link',
        'foreleg_1_visual_link',
        'foreleg_2_visual_link',
        'foreleg_3_visual_link',
        'spine_visual_bridge',
        'foreleg_3_foot_sphere',
        'hindleg_3_foot_sphere',
    ]

    # 直接通过几何体名称设置颜色
    for geom_name in geom_names:
        try:
            geom_id = sim.model.geom_name2id(geom_name)
            sim.model.geom_rgba[geom_id] = rgba.copy()
        except ValueError:
            pass


# ==================== 视频录制类 ====================
class VideoRecorder:
    def __init__(self, sim, output_file, fps=30, resolution=(1920, 1080), slow_motion_factor=1.0):
        """
        初始化视频录制器
        
        Args:
            sim: MuJoCo仿真对象
            output_file: 输出文件路径
            fps: 视频帧率
            resolution: 视频分辨率
            slow_motion_factor: 慢速因子 (1.0=正常速度, 0.5=半速)
        """
        self.sim = sim
        self.output_file = output_file
        self.fps = fps
        self.width, self.height = resolution
        self.frame_count = 0
        self.is_recording = False
        self.start_time = None
        self.slow_motion_factor = slow_motion_factor
        
        # 慢速录制时，实际录制的帧率保持不变，但视频会显得更慢
        # 因为仿真时间被拉伸了
        print(f"    慢速因子: {slow_motion_factor}x")
        print(f"    视频帧率: {fps} fps")
        print(f"    视频时长: {TOTAL_TIME/slow_motion_factor:.1f}秒")

        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_file, fourcc, fps, (self.width, self.height))

        if not self.video_writer.isOpened():
            raise Exception(f"无法创建视频文件: {output_file}")

    def start(self):
        """开始录制"""
        self.is_recording = True
        self.frame_count = 0
        self.start_time = time.time()

    def capture_frame(self, sim_time=None, phase_deg=None):
        """捕获当前帧"""
        if not self.is_recording:
            return

        try:
            # 使用sim.render()方法直接渲染
            frame = self.sim.render(
                width=self.width,
                height=self.height,
                mode='offscreen',
                camera_name='camera1'
            )

            if frame is not None:
                # 转换为BGR格式
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame

                # 添加时间戳和相位信息
                if sim_time is not None and phase_deg is not None:
                    # 慢速模式下，显示拉伸后的时间
                    display_time = sim_time / self.slow_motion_factor if self.slow_motion_factor > 0 else sim_time
                    time_text = f"Time: {display_time:.2f}s | Phase: {phase_deg:.1f}°"
                    if self.slow_motion_factor != 1.0:
                        time_text += f" (Slow: {self.slow_motion_factor}x)"
                    cv2.putText(frame_bgr, time_text, (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                # 写入视频
                self.video_writer.write(frame_bgr)
                self.frame_count += 1

        except Exception as e:
            print(f"  帧捕获错误: {e}")

    def stop(self):
        """停止录制"""
        if self.is_recording:
            self.is_recording = False
            self.video_writer.release()
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            return self.frame_count, elapsed_time

    def __del__(self):
        """析构函数"""
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()


# ==================== 运动控制函数 ====================
def create_sin_params(a_legH_hip, a_legH_knee, phase_lag):
    """
    根据给定参数创建Sin轨迹参数矩阵
    """
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


# ==================== 单次仿真录制函数 ====================
def run_single_simulation(phase_lag, phase_colors, output_folder, pbar, slow_motion_factor=1.0):
    """
    运行单次仿真并录制视频
    """
    phase_deg = phase_lag * 180 / np.pi
    
    # 生成视频文件名（添加慢速标记）
    slow_tag = f"_slow{slow_motion_factor}" if slow_motion_factor != 1.0 else ""
    video_filename = f"Sin_Phase_{phase_deg:+03.0f}deg_Hip{A_LEGH_HIP:.1f}_Knee{A_LEGH_KNEE:.1f}{slow_tag}.mp4"
    video_path = os.path.join(output_folder, video_filename)

    try:
        # 加载模型
        model = mujoco_py.load_model_from_path(MODEL_PATH)
        sim = mujoco_py.MjSim(model)
        viewer = mujoco_py.MjViewer(sim)

        # 设置相机
        viewer.cam.fixedcamid = 0
        viewer.cam.type = mujoco_py.generated.const.CAMERA_FIXED

        # 获取关节ID
        joint_pos_ids = []
        joint_vel_ids = []
        for name in JOINT_NAMES:
            joint_id = sim.model.joint_name2id(name)
            joint_pos_ids.append(sim.model.jnt_qposadr[joint_id])
            joint_vel_ids.append(sim.model.jnt_dofadr[joint_id])

        base_link_id = sim.model.body_name2id('base_link')
        dt = sim.model.opt.timestep

        # 设置初始关节位置
        initial_joint_pos = np.array([0, -0.5, -1.2, -1.2, 0, -0.5, -1.2, -1.2])
        sim.data.qpos[joint_pos_ids] = initial_joint_pos

        # 根据相位设置机器人颜色
        set_robot_color(sim, phase_lag, phase_colors)

        # 创建Sin参数矩阵
        sin_params = create_sin_params(A_LEGH_HIP, A_LEGH_KNEE, phase_lag)

        # 创建视频录制器（传入慢速因子）
        video_recorder = VideoRecorder(
            sim, video_path, VIDEO_FPS, VIDEO_RESOLUTION, 
            slow_motion_factor=slow_motion_factor
        )
        video_recorder.start()

        # 高层控制参数
        HIGH_LEVEL_FREQ = 240
        HIGH_LEVEL_DT = 1.0 / HIGH_LEVEL_FREQ

        # 初始化控制变量
        last_high_level_time = 0.0
        num_joints = len(JOINT_NAMES)
        high_level_target_pos, high_level_target_vel = get_all_joint_targets(0, sin_params, num_joints)

        # 主仿真循环
        current_time = 0.0
        frame_interval = 1.0 / VIDEO_FPS
        last_frame_time = -frame_interval

        # 记录位置用于速度计算
        base_x_positions = []
        times = []
        
        # 慢速模式下，仿真实时不变，但视频录制时间拉伸
        # 实际仿真时间仍然是TOTAL_TIME，但录制的视频时长会变长
        target_sim_time = TOTAL_TIME

        while current_time < target_sim_time:
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

            # 按帧率捕获视频（使用慢速因子调整录制时间）
            # 在慢速模式下，我们希望更频繁地捕获帧，以使视频看起来更慢
            # 录制频率保持不变，但仿真实时变慢，所以会录制更多帧
            if current_time - last_frame_time >= frame_interval * slow_motion_factor:
                # 传入显示时间（拉伸后的时间）
                display_time = current_time / slow_motion_factor if slow_motion_factor > 0 else current_time
                video_recorder.capture_frame(display_time, phase_deg)
                last_frame_time = current_time

            # 仿真步进
            sim.step()
            # viewer.render()
            current_time += dt

        # 停止视频录制
        frame_count, elapsed = video_recorder.stop()

        # 计算平均速度（从1s到4s）
        if len(times) > 1:
            idx_1s = np.argmin(np.abs(np.array(times) - 1.0))
            idx_4s = np.argmin(np.abs(np.array(times) - 4.0))

            pos_1s = base_x_positions[idx_1s]
            pos_4s = base_x_positions[idx_4s]
            time_diff = times[idx_4s] - times[idx_1s]

            avg_velocity = (pos_4s - pos_1s) / time_diff if time_diff > 0 else 0.0
            avg_velocity_mms = avg_velocity * 1000  # 转换为mm/s
        else:
            avg_velocity_mms = 0.0

        return True, frame_count, video_filename, avg_velocity_mms

    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, video_filename, 0.0


# ==================== 主程序 ====================
def main():
    # 初始化相位颜色映射
    phase_colors = setup_phase_colors()

    # 用于记录结果的列表
    results = []
    success_count = 0

    # 确定慢速因子
    slow_factor = SLOW_MOTION_FACTOR if ENABLE_SLOW_MOTION else 1.0

    # 创建总体进度条
    with tqdm(total=n_phases, desc="批量视频录制", ncols=100) as pbar:
        for i, phase_lag in enumerate(PHASE_LAG_RANGE):
            phase_deg = phase_lag * 180 / np.pi
            pbar.set_description(f"相位 {phase_deg:+6.1f}°")

            # 运行仿真并录制视频
            success, frame_count, filename, velocity = run_single_simulation(
                phase_lag, phase_colors, output_folder, pbar,
                slow_motion_factor=slow_factor
            )

            if success:
                success_count += 1
                results.append({
                    'phase_rad': phase_lag,
                    'phase_deg': phase_deg,
                    'filename': filename,
                    'frame_count': frame_count,
                    'velocity_mms': velocity
                })
                pbar.set_postfix({'状态': '成功', '速度': f'{velocity:.1f} mm/s'})
            else:
                pbar.set_postfix({'状态': '失败'})

            pbar.update(1)

            # 短暂暂停，确保资源释放
            if i < n_phases - 1:
                time.sleep(8)

    # 生成汇总信息
    print("\n" + "=" * 60)
    print("批量视频录制完成!")
    print(f"输出文件夹: {output_folder}")
    print(f"目标视频数量: {n_phases}")
    print(f"成功生成视频: {success_count}")
    print(f"成功率: {100 * success_count / n_phases:.1f}%")
    
    if ENABLE_SLOW_MOTION:
        print(f"慢速因子: {SLOW_MOTION_FACTOR}x")
        print(f"视频时长: {TOTAL_TIME/SLOW_MOTION_FACTOR:.1f}秒/个")

    # 保存汇总信息到TXT文件
    if success_count > 0:
        # 按速度排序
        results_sorted = sorted(results, key=lambda x: x['velocity_mms'], reverse=True)

        txt_path = os.path.join(output_folder, 'summary.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("批量视频录制信息\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            f.write("【固定参数】\n")
            f.write(f"后腿髋关节振幅: {A_LEGH_HIP:.2f}\n")
            f.write(f"后腿膝关节振幅: {A_LEGH_KNEE:.2f}\n")
            f.write(f"脊柱振幅: {A_SPINE:.2f}\n")
            f.write(f"前腿髋关节振幅: {A_LEGF_HIP:.2f}\n\n")

            f.write("【PD控制器参数】\n")
            f.write(f"Kp: {KP}, Kd: {KD}, Torque limit: {TOR}\n\n")

            f.write("【视频参数】\n")
            f.write(f"仿真时间: {TOTAL_TIME}秒\n")
            f.write(f"视频帧率: {VIDEO_FPS} fps\n")
            f.write(f"视频分辨率: {VIDEO_RESOLUTION[0]}x{VIDEO_RESOLUTION[1]}\n")
            if ENABLE_SLOW_MOTION:
                f.write(f"慢速因子: {SLOW_MOTION_FACTOR}x\n")
                f.write(f"输出视频时长: {TOTAL_TIME/SLOW_MOTION_FACTOR:.1f}秒/个\n\n")
            else:
                f.write("\n")

            f.write("【扫描参数】\n")
            f.write(f"相位范围: -π 到 π\n")
            f.write(f"步长: π/20\n")
            f.write(f"相位数量: {n_phases}\n")
            f.write(f"成功生成视频: {success_count}\n\n")

            f.write("【视频列表（按速度排序）】\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'序号':<6} {'相位(°)':<12} {'速度(mm/s)':<15} {'文件名':<40}\n")
            f.write("-" * 80 + "\n")

            for idx, result in enumerate(results_sorted, 1):
                f.write(f"{idx:<6} {result['phase_deg']:+8.1f}°   {result['velocity_mms']:>8.1f}       {result['filename']:<40}\n")

            f.write("-" * 80 + "\n\n")

            f.write("【统计信息】\n")
            velocities = [r['velocity_mms'] for r in results]
            f.write(f"速度范围: {min(velocities):.1f} - {max(velocities):.1f} mm/s\n")
            f.write(f"平均速度: {np.mean(velocities):.1f} mm/s\n")
            f.write(f"最大速度: {results_sorted[0]['velocity_mms']:.1f} mm/s @ 相位 {results_sorted[0]['phase_deg']:.1f}°\n")
            f.write("=" * 60 + "\n")

        print(f"汇总信息已保存: {txt_path}")

        # 显示最优结果
        best = results_sorted[0]
        print(f"\n最优结果:")
        print(f"  相位: {best['phase_deg']:.1f}°")
        print(f"  速度: {best['velocity_mms']:.1f} mm/s")
        print(f"  文件: {best['filename']}")

    print("\n程序执行完成！")


if __name__ == '__main__':
    main()