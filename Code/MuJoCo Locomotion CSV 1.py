# 不区分高层和低层控制
import mujoco_py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
import os
import cv2
import time

# ==================== 配置参数 ====================
ENABLE_DATA_VISUALIZATION = True           # 是否开启数据可视化
ENABLE_DATA_LOGGING = False                 # 是否开启数据记录
ENABLE_VIDEO_RECORDING = False              # 是否开启视频录制
TOTAL_TIME = 20

# 加载模型
model = mujoco_py.load_model_from_path(r"D:\Code\Model\Sengi\Sengi.xml")
csv_path = r"D:\Code\Sengi-Pybullet\Data\OptRef (RL) data 1.csv"
csv_filename = "MuJoCo Locomotion data 6.csv"
video_filename = "MuJoCo Locomotion video 6.mp4"

sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)
dt = sim.model.opt.timestep

# PD控制器参数
Kp = 10
Kd = 1
tor = 10

# 定义关节名称列表
joint_names = [
    "back_joint", "hindleg_joint_1", "hindleg_joint_2", "hindleg_joint_3",
    "front_joint", "foreleg_joint_1", "foreleg_joint_2", "foreleg_joint_3"
]

# 获取关节在 qpos 中的索引
actuator_ids = list(range(sim.model.nu))    # 执行器ID
joint_ids = []                              # 执行器对应关节ID
joint_pos_ids = []                          # 执行器对应位置索引
joint_vel_ids = []                          # 执行器对应速度索引

for name in joint_names:
    joint_id = sim.model.joint_name2id(name)
    joint_ids.append(joint_id)
    joint_pos_ids.append(sim.model.jnt_qposadr[joint_id])
    joint_vel_ids.append(sim.model.jnt_dofadr[joint_id])


# 设置初始关节位置
initial_joint_pos = np.array([0, -0.5, -1.2, -1.2, 0, -0.5, -1.2, -1.2,])
target_joint_pos = initial_joint_pos.copy()
base_link_id = sim.model.body_name2id('base_link')


# 读取CSV文件
if os.path.exists(csv_path):
    csv_data = pd.read_csv(csv_path)
    
    # 映射CSV列到我们的关节位置
    joint_pos_mapping = {
        "back_joint": "back_joint_pos",
        "front_joint": "front_joint_pos",
        "hindleg_joint_1": "hind_leg_pos",
        "hindleg_joint_2": "hind_leg_2_joint_pos", 
        "hindleg_joint_3": "hind_leg_3_joint_pos",
        "foreleg_joint_1": "fore_leg_pos",
        "foreleg_joint_2": "fore_leg_2_joint_pos",
        "foreleg_joint_3": "fore_leg_3_joint_pos"
    }
    
    # 映射CSV列到我们的关节速度
    joint_vel_mapping = {
        "back_joint": "back_joint_vel",
        "front_joint": "front_joint_vel",
        "hindleg_joint_1": "hind_leg_vel",
        "hindleg_joint_2": "hind_leg_2_joint_vel", 
        "hindleg_joint_3": "hind_leg_3_joint_vel",
        "foreleg_joint_1": "fore_leg_vel",
        "foreleg_joint_2": "fore_leg_2_joint_vel",
        "foreleg_joint_3": "fore_leg_3_joint_vel"
    }
    
    # 提取关节位置数据
    csv_joint_pos_data = []
    for i, name in enumerate(joint_names):
        csv_column = joint_pos_mapping.get(name)
        if csv_column and csv_column in csv_data.columns:
            joint_values = csv_data[csv_column].values
            csv_joint_pos_data.append(joint_values)
    
    # 提取关节速度数据
    csv_joint_vel_data = []
    for i, name in enumerate(joint_names):
        csv_column = joint_vel_mapping.get(name)
        if csv_column and csv_column in csv_data.columns:
            joint_values = csv_data[csv_column].values
            csv_joint_vel_data.append(joint_values)
    
    # 将列表转换为数组 (8 x N)
    csv_joint_pos_data = np.array(csv_joint_pos_data)
    csv_joint_vel_data = np.array(csv_joint_vel_data)
    
    # 获取时间数据
    if 'step' in csv_data.columns:
        csv_steps = csv_data['step'].values
        csv_times = csv_steps / 240.0
        csv_total_time = csv_times[-1]
    elif 'sim_time' in csv_data.columns:
        csv_times = csv_data['sim_time'].values
        csv_total_time = csv_times[-1]
        csv_steps = np.arange(len(csv_data))
    else:
        csv_steps = np.arange(len(csv_data))
        csv_times = csv_steps / 240.0
        csv_total_time = csv_times[-1]
    
    

# CSV数据控制参数
CSV_CONTROL_FREQ = 240  # CSV数据频率 (Hz)
CSV_DT = 1.0 / CSV_CONTROL_FREQ
print(f"CSV控制频率: {CSV_CONTROL_FREQ}Hz, 控制间隔: {CSV_DT:.6f}秒")
print(f"仿真频率: {1/dt:.1f}Hz, 仿真步长: {dt:.6f}秒")
print(f"每个CSV步长对应仿真步数: {CSV_DT/dt:.2f}")
print(f"数据记录状态: {'开启' if ENABLE_DATA_LOGGING else '关闭'}")
print(f"视频录制状态: {'开启' if ENABLE_VIDEO_RECORDING else '关闭'}")


# PD控制器
def PDcontrol(target_pos, target_vel, current_pos, current_vel):
    pos_error = target_pos - current_pos
    vel_error = target_vel - current_vel
    torque = Kp * pos_error + Kd * vel_error
    torque = np.clip(torque, -tor, tor)
    return torque

# 对位置进行线性插值
def linear_interpolate_position(t, t_array, data_array):
    if t <= t_array[0]:
        return data_array[:, 0]
    
    if t >= t_array[-1]:
        return data_array[:, -1]
    
    idx = np.searchsorted(t_array, t) - 1
    idx = max(0, min(idx, len(t_array) - 2))
    t_prev = t_array[idx]
    t_next = t_array[idx + 1]
    alpha = (t - t_prev) / (t_next - t_prev) if t_next > t_prev else 0
    
    result = np.zeros(data_array.shape[0])
    for i in range(data_array.shape[0]):
        prev_val = data_array[i, idx]
        next_val = data_array[i, idx + 1]
        result[i] = prev_val + alpha * (next_val - prev_val)
    
    return result

# 获取最近时间点的速度数据（不插值）
def get_nearest_velocity(t, t_array, data_array):
    if t <= t_array[0]:
        return data_array[:, 0]

    if t >= t_array[-1]:
        return data_array[:, -1]
    
    idx = np.argmin(np.abs(t_array - t))
    
    return data_array[:, idx]


# 视频录制设置
class VideoRecorder:
    def __init__(self, sim, output_file=video_filename, fps=30):
        self.sim = sim
        self.output_file = output_file
        self.fps = fps
        
        # 设置视频参数
        self.width = 3700
        self.height = 2080
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_file, fourcc, fps, (self.width, self.height))
        
        # 录制状态
        self.is_recording = False
        self.frame_count = 0
        self.start_time = None
        
    def start(self):
        """开始录制"""
        self.is_recording = True
        self.frame_count = 0
        self.start_time = time.time()
        print(f"开始录制视频到: {self.output_file}")
        print(f"视频参数: {self.width}x{self.height}, {self.fps} FPS")
        
    def capture_frame(self):
        """捕获当前帧"""
        if not self.is_recording:
            return
            
        try:
            # 使用默认相机进行渲染
            frame = self.sim.render(
                width=self.width, 
                height=self.height, 
                mode='offscreen'
            )
            
            # 检查frame的类型和形状
            if frame is not None:
                # 确保frame是numpy数组
                if not isinstance(frame, np.ndarray):
                    frame = np.array(frame)
                
                # 确保frame有正确的形状 (height, width, channels)
                if len(frame.shape) == 3:
                    if frame.shape[2] == 3:  # RGB格式
                        # 转换颜色格式 (RGB -> BGR for OpenCV)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    elif frame.shape[2] == 4:  # RGBA格式
                        # 先转换为RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                        # 再转换为BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # 调整大小到指定尺寸
                if frame.shape[0] != self.height or frame.shape[1] != self.width:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # 添加时间戳
                frame = self.add_timestamp(frame, time.time() - self.start_time)
                
                # 写入视频
                self.video_writer.write(frame)
                self.frame_count += 1
            else:
                print("警告: 渲染返回空帧")
                
        except Exception as e:
            print(f"捕获帧时出错: {e}")
            import traceback
            traceback.print_exc()
        
    def add_timestamp(self, frame, current_time):
        """在帧上添加时间戳信息"""
        if frame is None:
            return frame
            
        # 添加时间信息
        time_text = f"Time: {current_time:.2f}s"
        frame_text = f"Frame: {self.frame_count}"
        
        # 设置字体和颜色
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)  # 白色
        thickness = 2
        
        # 在图像左上角添加文本
        cv2.putText(frame, time_text, (10, 30), font, font_scale, color, thickness)
        cv2.putText(frame, frame_text, (10, 60), font, font_scale, color, thickness)
        
        return frame
        
    def stop(self):
        """停止录制"""
        if self.is_recording:
            self.is_recording = False
            self.video_writer.release()
            elapsed_time = time.time() - self.start_time
            print(f"视频录制完成: {self.output_file}")
            print(f"  录制帧数: {self.frame_count}")
            print(f"  录制时间: {elapsed_time:.2f}秒")
            if elapsed_time > 0:
                print(f"  实际帧率: {self.frame_count/elapsed_time:.1f} FPS")
            else:
                print(f"  实际帧率: N/A")
            
    def __del__(self):
        """析构函数，确保视频写入器被释放"""
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()


# 数据记录类
class DataLogger:
    def __init__(self, joint_names):
        self.joint_names = joint_names
        self.initial_base_x = None
        self.base_link_x = []
        self.base_times = []
        self.position_data = []
        self.position_times = []
        self.control_times = [] 
        self.csv_target_positions = []
        self.csv_target_velocities = []
        self.csv_update_times = []
        self.step_data = []  
        
    def record_base_position(self, time, current_base_x):
        if self.initial_base_x is None:
            self.initial_base_x = current_base_x
        relative_x = current_base_x - self.initial_base_x
        self.base_link_x.append(relative_x)
        self.base_times.append(time)
    
    def record_joint_data(self, time, target_pos, target_vel, current_pos, torque):
        self.position_data.append({
            'time': time,
            'target_pos': target_pos.copy(),
            'target_vel': target_vel.copy(),
            'current_pos': current_pos.copy(),
            'torque': torque.copy(), 
        })
        self.position_times.append(time)
        
    def record_control_time(self, time):
        self.control_times.append(time)
            
    def record_csv_target(self, time, target_pos, target_vel):
        self.csv_target_positions.append(target_pos.copy())
        self.csv_target_velocities.append(target_vel.copy())
        self.csv_update_times.append(time)
    
    def record_step_data(self, step, sim_time, real_time, base_x, base_y, base_z, base_pitch, joint_data_dict):
        if not ENABLE_DATA_LOGGING:
            return
            
        step_record = {
            'step': step,
            'sim_time': sim_time,
            'real_time': real_time,
            'base_x': base_x,
            'base_y': base_y,
            'base_z': base_z,
            'base_pitch': base_pitch,
        }
        
        step_record.update(joint_data_dict)
        self.step_data.append(step_record)
    
    def save_to_csv(self, filename):
        if not ENABLE_DATA_LOGGING:
            print("数据记录未开启，跳过保存")
            return
            
        if not self.step_data:
            print("警告: 没有数据可保存")
            return
        
        df = pd.DataFrame(self.step_data)
        df.to_csv(filename, index=False)
        
# 创建数据记录器
data_logger = DataLogger(joint_names)

# 创建视频录制器（如果开启视频录制）
if ENABLE_VIDEO_RECORDING:
    video_recorder = VideoRecorder(
        sim, 
        output_file=video_filename, 
        fps=30
    )
    
    # 设置相机
    viewer.cam.fixedcamid = 0  # 固定使用第一个相机
    viewer.cam.type = mujoco_py.generated.const.CAMERA_FIXED
    
    # 记录每帧之间的时间，用于控制录制帧率
    last_record_time = 0.0
    record_interval = 1.0 / video_recorder.fps

# 主仿真循环
current_time = 0.0
current_step = 0
start_real_time = time.time()  # 记录真实开始时间


# 开始录制视频（如果开启）
if ENABLE_VIDEO_RECORDING:
    video_recorder.start()

while current_time < TOTAL_TIME:
    current_real_time = time.time() - start_real_time
    current_time_mod = current_time % csv_total_time if csv_total_time > 0 else current_time


    current_pos = sim.data.qpos[joint_pos_ids].copy()
    current_vel = sim.data.qvel[joint_vel_ids].copy()

    target_pos = linear_interpolate_position(current_time_mod, csv_times, csv_joint_pos_data)
    target_vel = get_nearest_velocity(current_time_mod, csv_times, csv_joint_vel_data)
    
    torque = PDcontrol(target_pos, target_vel, current_pos, current_vel)
    for i in range(min(len(torque), sim.model.nu)):
        sim.data.ctrl[i] = torque[i]

    current_base_x = sim.data.body_xpos[base_link_id][0]
    current_base_y = sim.data.body_xpos[base_link_id][1]
    current_base_z = sim.data.body_xpos[base_link_id][2]
    base_quat = sim.data.body_xquat[base_link_id]
    w, x, y, z = base_quat
    base_pitch = np.arcsin(2.0 * (w * y - z * x))  
    
    data_logger.record_csv_target(current_time, target_pos, target_vel)
    data_logger.record_control_time(current_time)
    data_logger.record_base_position(current_time, current_base_x)
    data_logger.record_joint_data(current_time, target_pos, target_vel, current_pos, torque)
    
    joint_data_dict = {
        'front_joint_pos': current_pos[4],  
        'front_joint_vel': current_vel[4],
        'front_joint_torque': torque[4],
        'back_joint_pos': current_pos[0], 
        'back_joint_vel': current_vel[0],
        'back_joint_torque': torque[0],
        'fore_leg_pos': current_pos[5],    
        'fore_leg_vel': current_vel[5],
        'fore_leg_torque': torque[5],
        'hind_leg_pos': current_pos[1],     
        'hind_leg_vel': current_vel[1],
        'hind_leg_torque': torque[1],
        'fore_leg_2_joint_pos': current_pos[6],  
        'fore_leg_2_joint_vel': current_vel[6],
        'fore_leg_2_joint_torque': torque[6],
        'fore_leg_3_joint_pos': current_pos[7],
        'fore_leg_3_joint_vel': current_vel[7],
        'fore_leg_3_joint_torque': torque[7],
        'hind_leg_2_joint_pos': current_pos[2], 
        'hind_leg_2_joint_vel': current_vel[2],
        'hind_leg_2_joint_torque': torque[2],
        'hind_leg_3_joint_pos': current_pos[3], 
        'hind_leg_3_joint_vel': current_vel[3],
        'hind_leg_3_joint_torque': torque[3],
    }
    
    data_logger.record_step_data(
        step=current_step,
        sim_time=current_time,
        real_time=current_real_time,
        base_x=current_base_x,
        base_y=current_base_y,
        base_z=current_base_z,
        base_pitch=base_pitch,
        joint_data_dict=joint_data_dict
    )

    # 定期捕获视频帧（如果开启视频录制）
    if ENABLE_VIDEO_RECORDING and current_time - last_record_time >= record_interval:
        video_recorder.capture_frame()
        last_record_time = current_time

    # 仿真步进
    sim.step()
    viewer.render()
    current_time += dt
    current_step += 1
    
# 停止录制视频（如果开启）
if ENABLE_VIDEO_RECORDING:
    video_recorder.stop()


if ENABLE_DATA_LOGGING:
    data_logger.save_to_csv(csv_filename)

print(f"仿真总步数: {current_step}")
print(f"仿真总时间: {current_time:.3f}秒")
print(f"控制更新次数: {len(data_logger.control_times)}")
print(f"控制频率: {len(data_logger.control_times)/current_time:.1f} Hz (目标: {1/dt:.1f}Hz)")

# 中文字体和负号可视化问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 可视化部分
class VisualizationController:
    PAGES = [
        "基座位置",
        "关节: back_joint 和 front_joint",
        "关节: hindleg_joint_1 和 foreleg_joint_1",
        "关节: hindleg_joint_2 和 foreleg_joint_2",
        "关节: hindleg_joint_3 和 foreleg_joint_3"
    ]
    JOINT_GROUPS = [
        [0, 4],  # back_joint 和 front_joint
        [1, 5],  # hindleg_joint_1 和 foreleg_joint_1
        [2, 6],  # hindleg_joint_2 和 foreleg_joint_2
        [3, 7],  # hindleg_joint_3 和 foreleg_joint_3
    ]
    
    def __init__(self, data_logger, joint_names, csv_data=None, csv_joint_pos_data=None, csv_joint_vel_data=None, csv_times=None, csv_steps=None):
        self.data_logger = data_logger
        self.joint_names = joint_names
        self.csv_data = csv_data
        self.csv_joint_pos_data = csv_joint_pos_data
        self.csv_joint_vel_data = csv_joint_vel_data
        self.csv_times = csv_times
        self.csv_steps = csv_steps
        self.current_page = 0
        self.current_axes = []  
        self.twin_axes = []  
        self.fig = plt.figure(figsize=(16, 9))
        plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.15, hspace=0.3)
        ax_prev = plt.axes([0.30, 0.05, 0.15, 0.04])
        ax_next = plt.axes([0.55, 0.05, 0.15, 0.04])
        self.btn_prev = Button(ax_prev, '上一页')
        self.btn_next = Button(ax_next, '下一页')
        self.btn_prev.on_clicked(self.prev_page)
        self.btn_next.on_clicked(self.next_page)
        self.page_text = self.fig.text(0.5, 0.02, f'页面 {self.current_page+1}/{len(self.PAGES)}: {self.PAGES[self.current_page]}', ha='center', fontsize=12, fontweight='bold')
        
        # 初始绘制
        self.update_plot()

    # 清除所有坐标轴
    def clear_all_axes(self):
        for ax in self.twin_axes:
            if ax in self.fig.axes:
                ax.remove()
        self.twin_axes.clear()
        
        for ax in self.current_axes:
            if ax in self.fig.axes:
                ax.remove()
        self.current_axes.clear()
        
        self.fig.clf()
        
        plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.15, hspace=0.3)
        ax_prev = plt.axes([0.30, 0.05, 0.15, 0.04])
        ax_next = plt.axes([0.55, 0.05, 0.15, 0.04])
        self.btn_prev = Button(ax_prev, '上一页')
        self.btn_next = Button(ax_next, '下一页')
        self.btn_prev.on_clicked(self.prev_page)
        self.btn_next.on_clicked(self.next_page)
        self.page_text = self.fig.text(0.5, 0.02, f'页面 {self.current_page+1}/{len(self.PAGES)}: {self.PAGES[self.current_page]}', ha='center', fontsize=12, fontweight='bold')

    # 更新当前页面的可视化
    def update_plot(self):
        self.clear_all_axes()
        self.page_text.set_text(f'页面 {self.current_page+1}/{len(self.PAGES)}: {self.PAGES[self.current_page]}')
        
        if self.current_page == 0:  # 基座位置
            ax = self.fig.add_subplot(111)
            self.current_axes.append(ax)
            self.show_base_position(ax)
        else:  # 关节详细信息页面
            group_idx = self.current_page - 1
            joint_ids = self.JOINT_GROUPS[group_idx]
            
            ax1 = self.fig.add_subplot(2, 1, 1)
            ax2 = self.fig.add_subplot(2, 1, 2)
            
            self.current_axes.append(ax1)
            self.current_axes.append(ax2)

            times = self.data_logger.position_times

            self.plot_joint(ax1, joint_ids[0], times)
            self.plot_joint(ax2, joint_ids[1], times)
            
            if len(self.current_axes) > 1:
                self.current_axes[1].sharex(self.current_axes[0])
                plt.setp(self.current_axes[0].get_xticklabels(), visible=False)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # 为底部按钮留出空间
        self.fig.canvas.draw_idle()

    def plot_joint(self, ax, joint_idx, times):
        # 获取数据
        target_pos = [data['target_pos'][joint_idx] for data in self.data_logger.position_data]
        target_vel = [data['target_vel'][joint_idx] for data in self.data_logger.position_data]
        current_positions = [data['current_pos'][joint_idx] for data in self.data_logger.position_data]
        torque = [data['torque'][joint_idx] for data in self.data_logger.position_data]
        
        # 绘制位置曲线 - 左侧y轴
        ax.set_title(f'{self.joint_names[joint_idx]}', fontsize=12, pad=8)
        ax.set_ylabel('位置 (rad)', fontsize=10, color='tab:blue')
        
        # 绘制目标位置和实际位置
        line1, = ax.plot(times, target_pos, 'r-', linewidth=1.8, label='期望位置', alpha=0.8)
        line2, = ax.plot(times, current_positions, 'b-', linewidth=1.8, label='实际位置', alpha=0.8)
        
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 创建右侧y轴用于显示速度和力矩
        ax_right = ax.twinx()
        ax_right.set_ylabel('速度/力矩', fontsize=10, color='tab:red')
        
        # 绘制目标速度和力矩
        line3, = ax_right.plot(times, target_vel, 'c-', linewidth=1.5, label='期望速度', alpha=0.7)
        line4, = ax_right.plot(times, torque, 'm-', linewidth=1.5, label='控制力矩', alpha=0.7)
        
        ax_right.tick_params(axis='y', labelcolor='tab:red')
        
        # 保存右侧坐标轴的引用
        self.twin_axes.append(ax_right)
        
        # 合并图例
        lines = [line1, line2, line3, line4]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=8, ncol=2, framealpha=0.8)

    def show_base_position(self, ax):
        times = self.data_logger.base_times
        
        # 绘制基座位置
        ax.plot(times, self.data_logger.base_link_x, 'b-', linewidth=2.0, label='仿真X位置')
        
        # 如果CSV中有基座数据，也绘制
        if self.csv_data is not None and 'base_x' in self.csv_data.columns:
            csv_times = self.csv_times if self.csv_times is not None else np.arange(len(self.csv_data)) / 240.0
            # 计算相对位置
            initial_csv_x = self.csv_data['base_x'].iloc[0] if len(self.csv_data) > 0 else 0
            relative_csv_x = self.csv_data['base_x'].values - initial_csv_x
            ax.plot(csv_times[:len(relative_csv_x)], relative_csv_x, 'r--', linewidth=1.5, label='CSV X位置')
        
        ax.set_title('世界坐标系下的基座位置信息', fontsize=14, pad=15)
        ax.set_xlabel('时间 (秒)', fontsize=12)
        ax.set_ylabel('X位置 (米)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 计算并显示前进距离
        if len(self.data_logger.base_link_x) > 0:
            distance = self.data_logger.base_link_x[-1] - self.data_logger.base_link_x[0]
            ax.text(0.05, 0.95, f'总前进距离: {distance:.3f} 米', 
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 添加速度信息
            if len(self.data_logger.base_link_x) > 1:
                velocity = np.diff(self.data_logger.base_link_x) / np.diff(times)
                avg_velocity = distance / (times[-1] - times[0]) if len(times) > 1 else 0
                max_velocity = np.max(velocity) if len(velocity) > 0 else 0
                min_velocity = np.min(velocity) if len(velocity) > 0 else 0
                
                ax.text(0.05, 0.85, f'平均速度: {avg_velocity:.3f} 米/秒', 
                        transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                ax.text(0.05, 0.75, f'最大速度: {max_velocity:.3f} 米/秒', 
                        transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
                
                ax.text(0.05, 0.65, f'最小速度: {min_velocity:.3f} 米/秒', 
                        transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 添加图例
        ax.legend(loc='upper right', fontsize=10)

    # 切换到下一页
    def next_page(self, event):
        self.current_page = (self.current_page + 1) % len(self.PAGES)
        self.update_plot()
    
    # 切换到上一页
    def prev_page(self, event):
        self.current_page = (self.current_page - 1) % len(self.PAGES)
        self.update_plot()


if (ENABLE_DATA_VISUALIZATION):
    visualizer = VisualizationController(
        data_logger=data_logger,
        joint_names=joint_names,
        csv_data=csv_data,
        csv_joint_pos_data=csv_joint_pos_data,
        csv_joint_vel_data=csv_joint_vel_data,
        csv_times=csv_times,
        csv_steps=csv_steps
    )

    plt.show()