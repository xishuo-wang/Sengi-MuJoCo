# 分层控制版本 - CPG生成轨迹
import mujoco_py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
import os
import cv2
import time
from scipy import signal  # 用于数据平滑

# ==================== 配置参数 ====================
ENABLE_DATA_VISUALIZATION = True           # 是否开启数据可视化
ENABLE_DATA_LOGGING = True                 # 是否开启数据记录
ENABLE_VIDEO_RECORDING = False              # 是否开启视频录制
TOTAL_TIME = 5

# 加载模型
model = mujoco_py.load_model_from_path(r"D:\Code\Model\Sengi\Sengi.xml")
csv_filename = "MuJoCo OptRef (RL) data 33.csv"
video_filename = "MuJoCo OptRef (RL) video 33.mp4"

sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)
dt = sim.model.opt.timestep

# PD控制器参数
Kp = 3
Kd = 0.05
tor = 0.3

# 定义关节名称列表
joint_names = [
    "back_joint", "hindleg_joint_1", "hindleg_joint_2", "hindleg_joint_3",
    "front_joint", "foreleg_joint_1", "foreleg_joint_2", "foreleg_joint_3"
]

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
initial_joint_pos = np.array([0, -0.5, -1.2, -1.2, 0, -0.5, -1.2, -1.2])
target_joint_pos = initial_joint_pos.copy()
base_link_id = sim.model.body_name2id('base_link')

# CPG轨迹生成参数
TRAJ_LEN = 86  # 轨迹长度，与PyBullet版本一致

# 定义CPG正弦轨迹生成函数
def cpg_sin(beta, phase, amp, base, len):
    peak_x = round(len * beta)
    shift_num = round(len * phase)
    x1 = np.linspace(0, 0.5, peak_x)
    traj_1 = -np.cos(2 * np.pi * x1)
    x2 = np.linspace(0.5, 1, len - peak_x)
    traj_2 = -np.cos(2 * np.pi * x2)
    traj = amp * np.append(traj_1, traj_2) + base
    return shift_array(traj, shift_num)

def shift_array(arr, n):
    length = arr.shape[0]
    n = n % length 

    if n == 0:
        shifted_array = arr.copy()  
    else:
        shifted_array = np.concatenate((arr[-n:], arr[:-n]))  

    return shifted_array

# 生成各关节的CPG轨迹（与PyBullet版本参数一致）
print("生成CPG轨迹...")

# 生成位置轨迹（与PyBullet版本一致的参数）
fore_goal_position = cpg_sin(0.5, 0.6, 0.5, -0.4, TRAJ_LEN)
fore_2_goal_position = cpg_sin(0.5, 0.6, 0.25, -1.25, TRAJ_LEN)
spine_front_goal_position = -cpg_sin(0.5, 0, 0.5, 0, TRAJ_LEN)
spine_back_goal_position = -cpg_sin(0.5, 0, 0.5, 0, TRAJ_LEN)
hind_goal_position = cpg_sin(0.5, 0.05, 0.45, -0.4, TRAJ_LEN)
hind_2_goal_position = cpg_sin(0.5, 0.1, 0.5, -1.25, TRAJ_LEN)

# 组织关节轨迹到我们的关节顺序
# 关节顺序: ["back_joint", "hindleg_joint_1", "hindleg_joint_2", "hindleg_joint_3",
#            "front_joint", "foreleg_joint_1", "foreleg_joint_2", "foreleg_joint_3"]
cpg_joint_pos_data = np.zeros((8, TRAJ_LEN))

# back_joint (索引0) -> spine_back_goal_position
cpg_joint_pos_data[0] = spine_back_goal_position

# hindleg_joint_1 (索引1) -> hind_goal_position
cpg_joint_pos_data[1] = hind_goal_position

# hindleg_joint_2 (索引2) -> hind_2_goal_position
cpg_joint_pos_data[2] = hind_2_goal_position

# hindleg_joint_3 (索引3) -> hind_2_goal_position (与hindleg_joint_2相同)
cpg_joint_pos_data[3] = hind_2_goal_position

# front_joint (索引4) -> spine_front_goal_position
cpg_joint_pos_data[4] = spine_front_goal_position

# foreleg_joint_1 (索引5) -> fore_goal_position
cpg_joint_pos_data[5] = fore_goal_position

# foreleg_joint_2 (索引6) -> fore_2_goal_position
cpg_joint_pos_data[6] = fore_2_goal_position

# foreleg_joint_3 (索引7) -> fore_2_goal_position (与foreleg_joint_2相同)
cpg_joint_pos_data[7] = fore_2_goal_position

# 通过差分计算速度轨迹
cpg_joint_vel_data = np.zeros((8, TRAJ_LEN))
for i in range(8):
    for j in range(TRAJ_LEN):
        next_idx = (j + 1) % TRAJ_LEN
        cpg_joint_vel_data[i, j] = (cpg_joint_pos_data[i, next_idx] - cpg_joint_pos_data[i, j]) / (1.0/240.0)

print(f"CPG轨迹生成完成，轨迹长度: {TRAJ_LEN}")
print(f"back_joint位置范围: [{np.min(cpg_joint_pos_data[0]):.3f}, {np.max(cpg_joint_pos_data[0]):.3f}]")
print(f"front_joint位置范围: [{np.min(cpg_joint_pos_data[4]):.3f}, {np.max(cpg_joint_pos_data[4]):.3f}]")

# 高层控制参数 - 与PyBullet版本同步
HIGH_LEVEL_FREQ = 240  # Hz
HIGH_LEVEL_DT = 1.0 / HIGH_LEVEL_FREQ
LOW_LEVEL_DT = dt       # MuJoCo仿真步长 (通常1000Hz)

print(f"CPG控制频率: {HIGH_LEVEL_FREQ}Hz, 控制间隔: {HIGH_LEVEL_DT:.6f}秒")
print(f"仿真频率: {1/dt:.1f}Hz, 仿真步长: {dt:.6f}秒")
print(f"高层控制频率: {1/HIGH_LEVEL_DT:.1f}Hz, 底层控制频率: {1/LOW_LEVEL_DT:.1f}Hz")
print(f"数据记录状态: {'开启' if ENABLE_DATA_LOGGING else '关闭'}")
print(f"视频录制状态: {'开启' if ENABLE_VIDEO_RECORDING else '关闭'}")


# PD控制器
def PDcontrol(target_pos, target_vel, current_pos, current_vel):
    pos_error = target_pos - current_pos
    vel_error = 0 - current_vel
    torque = Kp * pos_error + Kd * vel_error
    torque = np.clip(torque, -tor, tor)
    return torque


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


# ==================== 数据记录类（增强版，包含接触力记录） ====================
class DataLogger:
    def __init__(self, joint_names):
        self.joint_names = joint_names
        self.initial_base_x = None
        self.base_link_x = []
        self.base_times = []
        self.position_data = []
        self.position_times = []
        self.control_times = [] 
        self.cpg_target_positions = []
        self.cpg_target_velocities = []
        self.cpg_update_times = []
        self.step_data = []
        
        # 新增：接触力记录
        self.contact_forces = []  # 存储所有接触力
        self.contact_times = []   # 存储对应时间
        self.total_force_x = []   # X轴总力（前进方向）
        self.total_force_z = []   # Z轴总力（垂直方向）
        
        # 分层控制统计
        self.high_level_updates = 0
        self.low_level_updates = 0
        
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
        
    def record_control_time(self, time, is_high_level):
        self.control_times.append((time, is_high_level))
        if is_high_level:
            self.high_level_updates += 1
        else:
            self.low_level_updates += 1
            
    def record_cpg_target(self, time, target_pos, target_vel):
        self.cpg_target_positions.append(target_pos.copy())
        self.cpg_target_velocities.append(target_vel.copy())
        self.cpg_update_times.append(time)
    
    def record_contact_forces(self, time, sim):
        """记录接触力 - 只读取后腿与地面的接触力"""
        total_fx = 0.0  # 世界坐标系X轴力（前进方向）
        total_fz = 0.0  # 世界坐标系Z轴力（垂直方向）
        
        # 后腿相关的几何体名称（根据你的模型调整）
        hind_leg_geoms = ['hindleg_3_collision']
        
        # 遍历所有接触
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            
            # 获取接触的geom ID
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # 获取geom名称
            geom1_name = sim.model.geom_id2name(geom1) if geom1 < sim.model.ngeom else None
            geom2_name = sim.model.geom_id2name(geom2) if geom2 < sim.model.ngeom else None
            
            # 检查是否涉及地面和后腿
            is_hind_leg_contact = False
            leg_name = None
            
            # 判断是否是后腿接触
            if geom1_name == 'floor':
                for hind_geom in hind_leg_geoms:
                    if geom2_name and hind_geom in geom2_name.lower():
                        is_hind_leg_contact = True
                        leg_name = geom2_name
                        break
            elif geom2_name == 'floor':
                for hind_geom in hind_leg_geoms:
                    if geom1_name and hind_geom in geom1_name.lower():
                        is_hind_leg_contact = True
                        leg_name = geom1_name
                        break
            
            if is_hind_leg_contact:
                # 获取接触力（在接触局部坐标系中）
                force_local = np.zeros(6)
                mujoco_py.functions.mj_contactForce(sim.model, sim.data, i, force_local)
                
                # 接触局部坐标系
                # contact.frame 是一个9元素的数组，reshape成3x3矩阵
                # 每一列是一个轴: [x_axis, y_axis, z_axis]
                # 其中z轴是接触法向
                contact_frame = contact.frame.reshape(3, 3)
                
                # 将力从接触坐标系转换到世界坐标系
                # 力向量在接触坐标系中为 [fx_local, fy_local, fz_local]
                force_local_3d = force_local[:3]
                
                # 转换: force_world = frame @ force_local
                # 因为frame的每一列是接触坐标系的轴在世界坐标系中的表示
                force_world = contact_frame @ force_local_3d
                
                fx_world = force_world[0]  # 世界X轴力（前进方向）
                fz_world = force_world[2]  # 世界Z轴力（垂直方向）
                
                # 存储详细信息
                contact_info = {
                    'time': time,
                    'leg': leg_name,
                    'force_x_world': fx_world,
                    'force_z_world': fz_world,
                    'force_magnitude': np.sqrt(fx_world**2 + fz_world**2),
                    'geom1': geom1_name,
                    'geom2': geom2_name,
                    'normal_force': force_world[2],  # 法向力（支撑力）
                    'friction_force': np.sqrt(force_world[0]**2 + force_world[1]**2)  # 摩擦力大小
                }
                self.contact_forces.append(contact_info)
                
                total_fx += fx_world
                total_fz += abs(fz_world)  # 支撑力取正值
        
        self.total_force_x.append(total_fx)
        self.total_force_z.append(total_fz)
        self.contact_times.append(time)
    
    def record_step_data(self, step, sim_time, real_time, base_x, base_y, base_z, base_pitch, joint_data_dict, sim):
        if not ENABLE_DATA_LOGGING:
            return
            
        # 记录接触力
        self.record_contact_forces(sim_time, sim)
            
        step_record = {
            'step': step,
            'sim_time': sim_time,
            'real_time': real_time,
            'base_x': base_x,
            'base_y': base_y,
            'base_z': base_z,
            'base_pitch': base_pitch,
            'total_contact_force_x': self.total_force_x[-1] if self.total_force_x else 0,
            'total_contact_force_z': self.total_force_z[-1] if self.total_force_z else 0,
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
        
        # 保存主要数据
        df = pd.DataFrame(self.step_data)
        df.to_csv(filename, index=False)
        
        # 保存接触力详细信息
        if self.contact_forces:
            contact_df = pd.DataFrame(self.contact_forces)
            contact_filename = filename.replace('.csv', '_contact_forces.csv')
            contact_df.to_csv(contact_filename, index=False)
            print(f"接触力数据已保存到: {contact_filename}")
    
    def get_contact_force_stats(self):
        """获取接触力统计信息"""
        if not self.contact_times:
            return None
        
        stats = {
            'avg_force_x': np.mean(self.total_force_x),
            'max_force_x': np.max(self.total_force_x),
            'min_force_x': np.min(self.total_force_x),
            'avg_force_z': np.mean(self.total_force_z),
            'max_force_z': np.max(self.total_force_z),
            'min_force_z': np.min(self.total_force_z),
            'avg_magnitude': np.mean(np.sqrt(np.array(self.total_force_x)**2 + np.array(self.total_force_z)**2))
        }
        return stats


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

# 准备CPG轨迹索引
cpg_traj_index = 0
print(f"将使用CPG轨迹控制机器人，轨迹长度: {TRAJ_LEN}")

# 设置初始上层控制目标
last_high_level_time = 0.0
high_level_target_pos = initial_joint_pos.copy()
high_level_target_vel = np.zeros(len(joint_names))

# 使用CPG第一帧数据初始化目标
high_level_target_pos = cpg_joint_pos_data[:, 0].copy()
high_level_target_vel = cpg_joint_vel_data[:, 0].copy()
print(f"初始目标位置: {high_level_target_pos}")
print(f"初始目标速度: {high_level_target_vel}")

# 主仿真循环
current_time = 0.0
current_step = 0
start_real_time = time.time()  # 记录真实开始时间

# 开始录制视频（如果开启）
if ENABLE_VIDEO_RECORDING:
    video_recorder.start()

while current_time < TOTAL_TIME:
    current_real_time = time.time() - start_real_time
    
    current_pos = sim.data.qpos[joint_pos_ids].copy()
    current_vel = sim.data.qvel[joint_vel_ids].copy()
    
    # ==================== 分层控制逻辑 ====================
    # 高层控制 (240Hz) - 与CPG轨迹频率同步
    if current_time - last_high_level_time >= HIGH_LEVEL_DT - 1e-9 or current_time == 0:
        last_high_level_time = current_time
        data_logger.record_control_time(current_time, True)
        
        # 使用CPG轨迹数据
        target_pos = cpg_joint_pos_data[:, cpg_traj_index % TRAJ_LEN].copy()
        target_vel = cpg_joint_vel_data[:, cpg_traj_index % TRAJ_LEN].copy()
        data_logger.record_cpg_target(current_time, target_pos, target_vel)
        
        # 更新CPG索引
        cpg_traj_index += 1
        
        # 如果到达轨迹末尾，从头开始
        if cpg_traj_index >= TRAJ_LEN and cpg_traj_index % 10 == 0:
            print(f"CPG轨迹循环: {cpg_traj_index//TRAJ_LEN}次 (当前时间: {current_time:.3f}s)")
        
        # 更新上层目标位置和速度
        high_level_target_pos = target_pos.copy()
        high_level_target_vel = target_vel.copy()
    
    # 底层控制 (MuJoCo仿真频率) - 每个仿真步都执行
    data_logger.record_control_time(current_time, False)
    
    # ==================== 应用PD控制 ====================
    torque = PDcontrol(high_level_target_pos, high_level_target_vel, current_pos, current_vel)
    for i in range(min(len(torque), sim.model.nu)):
        sim.data.ctrl[i] = torque[i]
    
    # ==================== 记录数据 ====================
    current_base_x = sim.data.body_xpos[base_link_id][0]
    current_base_y = sim.data.body_xpos[base_link_id][1]
    current_base_z = sim.data.body_xpos[base_link_id][2]
    base_quat = sim.data.body_xquat[base_link_id]
    w, x, y, z = base_quat
    base_pitch = np.arcsin(2.0 * (w * y - z * x))  
    
    data_logger.record_base_position(current_time, current_base_x)
    data_logger.record_joint_data(current_time, high_level_target_pos, high_level_target_vel, current_pos, torque)
    
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
        joint_data_dict=joint_data_dict,
        sim=sim  # 添加sim参数
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
    
    # 输出接触力统计
    force_stats = data_logger.get_contact_force_stats()
    if force_stats:
        print("\n" + "="*50)
        print("接触力统计信息:")
        print(f"  平均前进方向力 (X轴): {force_stats['avg_force_x']:.3f} N")
        print(f"  最大前进方向力 (X轴): {force_stats['max_force_x']:.3f} N")
        print(f"  最小前进方向力 (X轴): {force_stats['min_force_x']:.3f} N")
        print(f"  平均垂直方向力 (Z轴): {force_stats['avg_force_z']:.3f} N")
        print(f"  最大垂直方向力 (Z轴): {force_stats['max_force_z']:.3f} N")
        print(f"  最小垂直方向力 (Z轴): {force_stats['min_force_z']:.3f} N")
        print(f"  平均合力大小: {force_stats['avg_magnitude']:.3f} N")
        print("="*50)

print(f"\n仿真完成!")
print(f"仿真总步数: {current_step}")
print(f"仿真总时间: {current_time:.3f}秒")
print(f"CPG轨迹循环次数: {cpg_traj_index//TRAJ_LEN}")
print(f"高层控制更新次数: {data_logger.high_level_updates}")
print(f"底层控制更新次数: {data_logger.low_level_updates}")
print(f"高层控制频率: {data_logger.high_level_updates/current_time:.1f} Hz (目标: {HIGH_LEVEL_FREQ}Hz)")
print(f"底层控制频率: {data_logger.low_level_updates/current_time:.1f} Hz (目标: {1/dt:.1f}Hz)")

# 中文字体和负号可视化问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 可视化控制器（增强版，包含接触力页面） ====================
class VisualizationController:
    PAGES = [
        "基座位置",
        "接触力分析",  # 新增页面
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
    
    def __init__(self, data_logger, joint_names, cpg_pos_data=None, cpg_vel_data=None):
        self.data_logger = data_logger
        self.joint_names = joint_names
        self.cpg_pos_data = cpg_pos_data
        self.cpg_vel_data = cpg_vel_data
        self.current_page = 0
        self.current_axes = []  
        self.twin_axes = []  
        self.fig = plt.figure(figsize=(16, 12))  # 增加高度以容纳更多内容
        plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.15, hspace=0.4)
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
        
        plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.15, hspace=0.4)
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
        elif self.current_page == 1:  # 接触力分析
            self.show_contact_forces()
        else:  # 关节详细信息页面
            group_idx = self.current_page - 2  # 调整索引
            joint_indices = self.JOINT_GROUPS[group_idx]
            
            ax1 = self.fig.add_subplot(2, 1, 1)
            ax2 = self.fig.add_subplot(2, 1, 2)
            
            self.current_axes.append(ax1)
            self.current_axes.append(ax2)

            times = self.data_logger.position_times

            self.plot_joint(ax1, joint_indices[0], times)
            self.plot_joint(ax2, joint_indices[1], times)
            
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
        line1, = ax.plot(times, target_pos, 'r-', linewidth=1.8, label='期望位置(CPG)', alpha=0.8)
        line2, = ax.plot(times, current_positions, 'b-', linewidth=1.8, label='实际位置', alpha=0.8)
        
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 创建右侧y轴用于显示速度和力矩
        ax_right = ax.twinx()
        ax_right.set_ylabel('速度/力矩', fontsize=10, color='tab:red')
        
        # 绘制目标速度和力矩
        line3, = ax_right.plot(times, target_vel, 'c-', linewidth=1.5, label='期望速度(CPG)', alpha=0.7)
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
        
        ax.set_title('世界坐标系下的基座位置信息 (CPG控制)', fontsize=14, pad=15)
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

    def show_contact_forces(self):
        """显示接触力分析"""
        times = self.data_logger.contact_times
        
        if not times:
            ax = self.fig.add_subplot(111)
            self.current_axes.append(ax)
            ax.text(0.5, 0.5, '无接触力数据', ha='center', va='center', fontsize=16)
            return
        
        # 创建3个子图
        ax1 = self.fig.add_subplot(3, 1, 1)
        ax2 = self.fig.add_subplot(3, 1, 2)
        ax3 = self.fig.add_subplot(3, 1, 3)
        
        self.current_axes.extend([ax1, ax2, ax3])
        
        # 1. 总接触力（X轴和Z轴）
        force_x = self.data_logger.total_force_x
        force_z = self.data_logger.total_force_z
        
        # 数据平滑处理
        if len(force_x) > 10:
            window_size = min(11, len(force_x) // 10)
            if window_size % 2 == 0:
                window_size += 1
            force_x_smooth = signal.savgol_filter(force_x, window_size, 2)
            force_z_smooth = signal.savgol_filter(force_z, window_size, 2)
        else:
            force_x_smooth = force_x
            force_z_smooth = force_z
        
        ax1.plot(times, force_x, 'b-', alpha=0.3, linewidth=1, label='原始X轴力')
        ax1.plot(times, force_x_smooth, 'b-', linewidth=2, label='平滑X轴力')
        ax1.plot(times, force_z, 'r-', alpha=0.3, linewidth=1, label='原始Z轴力')
        ax1.plot(times, force_z_smooth, 'r-', linewidth=2, label='平滑Z轴力')
        ax1.set_ylabel('力 (N)', fontsize=10)
        ax1.set_title('地面接触力 (X轴:前进方向, Z轴:垂直方向)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)
        
        # 2. 合力大小和方向
        force_magnitude = np.sqrt(np.array(force_x)**2 + np.array(force_z)**2)
        force_angle = np.arctan2(force_z, force_x) * 180 / np.pi  # 角度（度）
        
        ax2.plot(times, force_magnitude, 'g-', linewidth=2, label='合力大小')
        ax2.set_ylabel('合力大小 (N)', fontsize=10, color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.grid(True, alpha=0.3)
        
        # 添加第二Y轴显示角度
        ax2_twin = ax2.twinx()
        ax2_twin.plot(times, force_angle, 'm-', linewidth=1.5, label='合力方向', alpha=0.7)
        ax2_twin.set_ylabel('合力方向 (度)', fontsize=10, color='m')
        ax2_twin.tick_params(axis='y', labelcolor='m')
        ax2.set_title('接触合力分析', fontsize=12)
        
        # 保存右侧坐标轴引用
        self.twin_axes.append(ax2_twin)
        
        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        # 3. 各条腿的接触力（如果记录了详细信息）
        if self.data_logger.contact_forces:
            # 按腿分组
            leg_forces = {}
            for contact in self.data_logger.contact_forces:
                leg = contact['leg']
                if leg not in leg_forces:
                    leg_forces[leg] = {'times': [], 'forces': []}
                leg_forces[leg]['times'].append(contact['time'])
                leg_forces[leg]['forces'].append(contact['force_magnitude'])
            
            colors = ['b', 'r', 'g', 'm', 'c', 'y', 'orange', 'purple']
            for i, (leg, data) in enumerate(leg_forces.items()):
                color = colors[i % len(colors)]
                # 数据平滑
                if len(data['forces']) > 5:
                    window = min(5, len(data['forces'])//3)
                    if window % 2 == 0:
                        window += 1
                    if window >= 3:
                        forces_smooth = signal.savgol_filter(data['forces'], window, 2)
                    else:
                        forces_smooth = data['forces']
                else:
                    forces_smooth = data['forces']
                
                ax3.plot(data['times'], forces_smooth, color=color, linewidth=1.5, 
                        label=f'{leg}', alpha=0.8)
            
            ax3.set_xlabel('时间 (秒)', fontsize=10)
            ax3.set_ylabel('接触力大小 (N)', fontsize=10)
            ax3.set_title('各腿接触力（平滑后）', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper right', fontsize=7, ncol=2)
        
        # 添加统计信息
        stats = self.data_logger.get_contact_force_stats()
        if stats:
            info_text = (
                f"平均X轴力: {stats['avg_force_x']:.2f} N\n"
                f"最大X轴力: {stats['max_force_x']:.2f} N\n"
                f"平均Z轴力: {stats['avg_force_z']:.2f} N\n"
                f"最大Z轴力: {stats['max_force_z']:.2f} N\n"
                f"平均合力: {stats['avg_magnitude']:.2f} N"
            )
            ax1.text(0.02, 0.95, info_text, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

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
        cpg_pos_data=cpg_joint_pos_data,
        cpg_vel_data=cpg_joint_vel_data
    )

    plt.show()