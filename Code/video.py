# CPG控制 - 单次仿真视频录制 (带相位颜色处理)
import mujoco_py
import numpy as np
import cv2
import time
import os

# ==================== 配置参数 ====================
# 模型路径
MODEL_PATH = r"D:\Code\Model\Sengi\Sengi-video.xml"

# 视频输出配置
VIDEO_OUTPUT_FOLDER = r"D:\Code\Sengi-MuJoCo\Video-Matlab\SingleRun"
VIDEO_FPS = 30                                # 视频帧率
VIDEO_RESOLUTION = (3700, 2080)               # 视频分辨率

# 运动参数
TOTAL_TIME = 4                                 # 仿真时间 (秒)
f = 3                                          # 运动频率
TRAJ_LEN = 86                                  # CPG轨迹长度

# 当前运行的相位值 (可以修改这个值来改变颜色)
CURRENT_PHASE = -np.pi/20                       # 示例相位: -45°

# PD控制器参数
Kp = 3
Kd = 0.05
tor = 0.3

# 定义关节名称列表
joint_names = [
    "back_joint", "hindleg_joint_1", "hindleg_joint_2", "hindleg_joint_3",
    "front_joint", "foreleg_joint_1", "foreleg_joint_2", "foreleg_joint_3"
]

print("="*60)
print("CPG单次仿真视频录制")
print(f"当前相位: {CURRENT_PHASE*180/np.pi:.1f}°")
print(f"仿真时间: {TOTAL_TIME}秒")
print(f"视频输出: {VIDEO_OUTPUT_FOLDER}")
print("="*60)

# 创建输出文件夹
os.makedirs(VIDEO_OUTPUT_FOLDER, exist_ok=True)

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
    
    # 几何体名称列表（您想要着色的所有可视化几何体）
    geom_names = [
        'spine_back_visual_link',      # 后部脊椎连接杆
        'spine_front_visual_link',      # 前部脊椎连接杆
        'hindleg_1_visual_link',       # 后腿1连接杆
        'hindleg_2_visual_link',       # 后腿2连接杆
        'hindleg_3_visual_link',       # 后腿3连接杆
        'foreleg_1_visual_link',       # 前腿1连接杆
        'foreleg_2_visual_link',       # 前腿2连接杆
        'foreleg_3_visual_link',       # 前腿3连接杆
        'spine_visual_bridge',          # 脊椎连接桥
        'foreleg_3_foot_sphere',
        'hindleg_3_foot_sphere',
    ]
    
    # 直接通过几何体名称设置颜色
    for geom_name in geom_names:
        try:
            geom_id = sim.model.geom_name2id(geom_name)
            sim.model.geom_rgba[geom_id] = rgba.copy()
        except ValueError:
            pass  # 忽略找不到的几何体

# ==================== 视频录制类 ====================
# ==================== 视频录制类（使用sim.render方法）====================
class VideoRecorder:
    def __init__(self, sim, output_file, fps=30, resolution=(1920, 1080)):
        self.sim = sim
        self.output_file = output_file
        self.fps = fps
        self.width, self.height = resolution
        self.frame_count = 0
        self.is_recording = False
        self.start_time = None
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_file, fourcc, fps, (self.width, self.height))
        
        if not self.video_writer.isOpened():
            raise Exception(f"无法创建视频文件: {output_file}")
        
        print(f"视频录制初始化完成")
        print(f"  输出文件: {os.path.basename(output_file)}")
        print(f"  分辨率: {self.width}x{self.height}")
        print(f"  帧率: {fps} FPS")
    
    def start(self):
        """开始录制"""
        self.is_recording = True
        self.frame_count = 0
        self.start_time = time.time()
        print(f"开始录制视频: {self.output_file}")
        
    def capture_frame(self, sim_time=None):
        """捕获当前帧"""
        if not self.is_recording:
            return
            
        try:
            # 使用sim.render()方法直接渲染
            frame = self.sim.render(
                width=self.width,
                height=self.height,
                mode='offscreen',
                camera_name='camera1'  # 使用指定的相机
            )
            
            if frame is not None:
                # 转换为BGR格式
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                # 添加时间戳
                if sim_time is not None:
                    time_text = f"Time: {sim_time:.2f}s | Phase: {CURRENT_PHASE*180/np.pi:.1f}°"
                    cv2.putText(frame_bgr, time_text, (30, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                # 写入视频
                self.video_writer.write(frame_bgr)
                self.frame_count += 1
                
        except Exception as e:
            print(f"帧捕获错误: {e}")
    
    def stop(self):
        """停止录制"""
        if self.is_recording:
            self.is_recording = False
            self.video_writer.release()
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            print(f"视频录制完成: {self.frame_count} 帧")
            if elapsed_time > 0:
                print(f"  实际帧率: {self.frame_count/elapsed_time:.1f} FPS")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()


# ==================== CPG轨迹生成函数 ====================
def cpg_sin(beta, phase, amp, base, length):
    peak_x = round(length * beta)
    shift_num = round(length * phase)
    x1 = np.linspace(0, 0.5, peak_x)
    traj_1 = -np.cos(2 * np.pi * x1)
    x2 = np.linspace(0.5, 1, length - peak_x)
    traj_2 = -np.cos(2 * np.pi * x2)
    traj = amp * np.append(traj_1, traj_2) + base
    return shift_array(traj, shift_num)

def shift_array(arr, n):
    length = arr.shape[0]
    n = n % length 
    if n == 0:
        return arr.copy()  
    else:
        return np.concatenate((arr[-n:], arr[:-n]))  

def generate_cpg_trajectories(phase_lag):
    """
    生成CPG轨迹
    """
    # 生成位置轨迹
    fore_goal_position = cpg_sin(0.5, 0.6, 0.5, -0.4, TRAJ_LEN)
    fore_2_goal_position = cpg_sin(0.5, 0.6, 0.25, -1.25, TRAJ_LEN)
    spine_front_goal_position = -cpg_sin(0.5, 0, 0.5, 0, TRAJ_LEN)
    spine_back_goal_position = -cpg_sin(0.5, 0, 0.5, 0, TRAJ_LEN)
    
    # 使用相位滞后调整后腿轨迹
    hind_goal_position = cpg_sin(0.5, phase_lag, 0.45, -0.4, TRAJ_LEN)
    hind_2_goal_position = cpg_sin(0.5, phase_lag, 0.5, -1.25, TRAJ_LEN)
    
    # 组织关节轨迹
    cpg_joint_pos_data = np.zeros((8, TRAJ_LEN))
    cpg_joint_pos_data[0] = spine_back_goal_position      # back_joint
    cpg_joint_pos_data[1] = hind_goal_position            # hindleg_joint_1
    cpg_joint_pos_data[2] = hind_2_goal_position          # hindleg_joint_2
    cpg_joint_pos_data[3] = hind_2_goal_position          # hindleg_joint_3
    cpg_joint_pos_data[4] = spine_front_goal_position     # front_joint
    cpg_joint_pos_data[5] = fore_goal_position            # foreleg_joint_1
    cpg_joint_pos_data[6] = fore_2_goal_position          # foreleg_joint_2
    cpg_joint_pos_data[7] = fore_2_goal_position          # foreleg_joint_3
    
    # 计算速度轨迹
    cpg_joint_vel_data = np.zeros((8, TRAJ_LEN))
    dt_control = 1.0 / 240.0
    for i in range(8):
        for j in range(TRAJ_LEN):
            next_idx = (j + 1) % TRAJ_LEN
            cpg_joint_vel_data[i, j] = (cpg_joint_pos_data[i, next_idx] - cpg_joint_pos_data[i, j]) / dt_control
    
    return cpg_joint_pos_data, cpg_joint_vel_data

# ==================== 主程序 ====================
def main():
    # 初始化相位颜色映射
    phase_colors = setup_phase_colors()
    
    # 生成视频文件名
    phase_deg = CURRENT_PHASE * 180 / np.pi
    video_filename = f"CPG_Phase_{phase_deg:+03.0f}deg.mp4"
    video_path = os.path.join(VIDEO_OUTPUT_FOLDER, video_filename)
    
    print(f"\n开始录制: {video_filename}")
    
    # 加载模型
    model = mujoco_py.load_model_from_path(MODEL_PATH)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    
    # 获取关节ID
    joint_pos_ids = []
    joint_vel_ids = []
    for name in joint_names:
        joint_id = sim.model.joint_name2id(name)
        joint_pos_ids.append(sim.model.jnt_qposadr[joint_id])
        joint_vel_ids.append(sim.model.jnt_dofadr[joint_id])
    
    base_link_id = sim.model.body_name2id('base_link')
    dt = sim.model.opt.timestep
    
    # 设置初始关节位置
    initial_joint_pos = np.array([0, -0.5, -1.2, -1.2, 0, -0.5, -1.2, -1.2])
    sim.data.qpos[joint_pos_ids] = initial_joint_pos
    
    # 根据相位设置机器人颜色
    set_robot_color(sim, CURRENT_PHASE, phase_colors)
    
    # 生成CPG轨迹
    cpg_joint_pos_data, cpg_joint_vel_data = generate_cpg_trajectories(CURRENT_PHASE)
    
    # 创建视频录制器
    video_recorder = VideoRecorder(sim, video_path, VIDEO_FPS, VIDEO_RESOLUTION)
    video_recorder.start()


        
    viewer.cam.fixedcamid = 0  # 固定使用第一个相机
    viewer.cam.type = mujoco_py.generated.const.CAMERA_FIXED
    
    # 记录每帧之间的时间，用于控制录制帧率
    last_record_time = 0.0
    record_interval = 1.0 / video_recorder.fps
    # 主循环中控制帧率
    frame_interval = 1.0 / VIDEO_FPS
    last_frame_time = -frame_interval
    # 高层控制参数
    HIGH_LEVEL_FREQ = 240
    HIGH_LEVEL_DT = 1.0 / HIGH_LEVEL_FREQ
    
    # PD控制器
    def PDcontrol(target_pos, target_vel, current_pos, current_vel):
        pos_error = target_pos - current_pos
        vel_error = 0 - current_vel
        torque = Kp * pos_error + Kd * vel_error
        return np.clip(torque, -tor, tor)
    
    # 初始化控制变量
    cpg_traj_index = 0
    last_high_level_time = 0.0
    high_level_target_pos = cpg_joint_pos_data[:, 0].copy()
    high_level_target_vel = cpg_joint_vel_data[:, 0].copy()
    
    # 主仿真循环
    current_time = 0.0
    frame_interval = 1.0 / VIDEO_FPS
    last_frame_time = -frame_interval  # 确保第一帧被捕获
    
    print("开始仿真...")
    start_time = time.time()
    
    while current_time < TOTAL_TIME:
        current_pos = sim.data.qpos[joint_pos_ids].copy()
        current_vel = sim.data.qvel[joint_vel_ids].copy()
        
        # 高层控制 (240Hz)
        if current_time - last_high_level_time >= HIGH_LEVEL_DT - 1e-9:
            last_high_level_time = current_time
            
            # 使用CPG轨迹
            target_pos = cpg_joint_pos_data[:, cpg_traj_index % TRAJ_LEN].copy()
            target_vel = cpg_joint_vel_data[:, cpg_traj_index % TRAJ_LEN].copy()
            
            cpg_traj_index += 1
            
            high_level_target_pos = target_pos.copy()
            high_level_target_vel = target_vel.copy()
        
        # 应用PD控制
        torque = PDcontrol(high_level_target_pos, high_level_target_vel, current_pos, current_vel)
        for i in range(min(len(torque), sim.model.nu)):
            sim.data.ctrl[i] = torque[i]
        
        
        if current_time - last_frame_time >= frame_interval:
            video_recorder.capture_frame(current_time)
            last_frame_time = current_time
        # 仿真步进
        sim.step()
        viewer.render()
        current_time += dt
        
        # 显示进度
        if int(current_time * 10) % 10 == 0 and current_time > 0.01:
            progress = int(current_time / TOTAL_TIME * 100)
            print(f"\r进度: {progress}%", end="")
    
    # 停止视频录制
    video_recorder.stop()
    
    # 计算实际运行时间
    elapsed = time.time() - start_time
    print(f"\n仿真完成! 实际运行时间: {elapsed:.2f}秒")
    print(f"视频已保存: {video_path}")

if __name__ == '__main__':
    main()