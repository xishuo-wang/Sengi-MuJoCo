"""
数据可视化模块 - 仿真数据分析和展示
包含DataLogger类和交互式可视化界面
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import csv
import os


# ==================== 配置参数 ====================
# 数据文件路径
CSV_DATA_FILE = "simulation_data.csv"

# 可视化默认参数
DEFAULT_PHASE_LAG = -np.pi / 20
DEFAULT_TOTAL_TIME = 5.0
DEFAULT_ANALYSIS_START_TIME = 0.0
DEFAULT_ANALYSIS_END_TIME = 5.0
DEFAULT_IMPULSE_START = 0.12
DEFAULT_IMPULSE_END = 0.2

# 关节名称列表
JOINT_NAMES = [
    "back_joint", "hindleg_joint_1", "hindleg_joint_2", "hindleg_joint_3",
    "front_joint", "foreleg_joint_1", "foreleg_joint_2", "foreleg_joint_3"
]

# 图表显示配置
FIGURE_SIZE = (14, 10)
FONT_FAMILY = 'SimHei'


class DataLogger:
    # 初始化数据记录器
    def __init__(self, joint_names):
        self.joint_names = joint_names
        self.num_joints = len(joint_names)

        # 基座位置数据
        self.base_times = []
        self.base_x_positions = []
        self.initial_base_x = None

        # 关节状态数据（高层控制频率记录）
        self.joint_times = []
        self.target_positions = []
        self.actual_positions = []
        self.actual_velocities = []

        # 关节力矩数据（仿真步频率记录）
        self.torque_times = []
        self.joint_torques = []

        # 接触力数据
        self.contact_forces = {}

    # 记录基座位置
    def record_base_pos(self, time, base_x):
        if self.initial_base_x is None:
            self.initial_base_x = base_x

        relative_x = base_x - self.initial_base_x
        self.base_times.append(time)
        self.base_x_positions.append(relative_x)

    # 记录高层控制数据
    def record_high_level_data(self, time, target_pos, current_pos, current_vel):
        self.joint_times.append(time)
        self.target_positions.append(target_pos.copy())
        self.actual_positions.append(current_pos.copy())
        self.actual_velocities.append(current_vel.copy())

    # 记录关节力矩数据
    def record_torque_data(self, time, torque):
        self.torque_times.append(time)
        self.joint_torques.append(torque.copy())

    # 记录接触力数据
    def record_contact_force(self, geom_name, time, force, magnitude):
        if geom_name not in self.contact_forces:
            self.contact_forces[geom_name] = {
                'time': [], 'fx': [], 'fy': [], 'fz': [], 'magnitude': []
            }

        self.contact_forces[geom_name]['time'].append(time)
        self.contact_forces[geom_name]['fx'].append(force[0])
        self.contact_forces[geom_name]['fy'].append(force[1])
        self.contact_forces[geom_name]['fz'].append(force[2])
        self.contact_forces[geom_name]['magnitude'].append(magnitude)

    # 获取总前进距离
    def get_total_distance(self):
        if len(self.base_x_positions) > 0:
            return self.base_x_positions[-1] - self.base_x_positions[0]
        return 0.0

    # 计算指定时间段平均速度
    def calculate_average_velocity(self, start_time, end_time):
        if len(self.base_times) < 2:
            return 0.0

        start_idx = np.argmin(np.abs(np.array(self.base_times) - start_time))
        end_idx = np.argmin(np.abs(np.array(self.base_times) - end_time))

        if start_idx >= end_idx:
            return 0.0

        pos_start = self.base_x_positions[start_idx]
        pos_end = self.base_x_positions[end_idx]
        time_diff = self.base_times[end_idx] - self.base_times[start_idx]

        if time_diff > 0:
            return (pos_end - pos_start) / time_diff
        return 0.0

    # 保存数据到CSV文件
    def save_to_csv(self, filename):
        max_len = max(
            len(self.base_times),
            len(self.joint_times),
            len(self.torque_times)
        )

        rows = []
        for i in range(max_len):
            row = {}

            # 基座数据
            if i < len(self.base_times):
                row['base_time'] = self.base_times[i]
                row['base_x'] = self.base_x_positions[i]
            else:
                row['base_time'] = ''
                row['base_x'] = ''

            # 关节状态数据
            if i < len(self.joint_times):
                row['joint_time'] = self.joint_times[i]
                for j in range(self.num_joints):
                    row[f'target_pos_{j}'] = self.target_positions[i][j]
                    row[f'actual_pos_{j}'] = self.actual_positions[i][j]
                    row[f'actual_vel_{j}'] = self.actual_velocities[i][j]
            else:
                row['joint_time'] = ''
                for j in range(self.num_joints):
                    row[f'target_pos_{j}'] = ''
                    row[f'actual_pos_{j}'] = ''
                    row[f'actual_vel_{j}'] = ''

            # 力矩数据
            if i < len(self.torque_times):
                row['torque_time'] = self.torque_times[i]
                for j in range(self.num_joints):
                    row[f'torque_{j}'] = self.joint_torques[i][j]
            else:
                row['torque_time'] = ''
                for j in range(self.num_joints):
                    row[f'torque_{j}'] = ''

            # 接触力数据（取第一个几何体）
            if self.contact_forces:
                first_geom = list(self.contact_forces.keys())[0]
                cf = self.contact_forces[first_geom]
                if i < len(cf['time']):
                    row['contact_time'] = cf['time'][i]
                    row['contact_fx'] = cf['fx'][i]
                    row['contact_fy'] = cf['fy'][i]
                    row['contact_fz'] = cf['fz'][i]
                    row['contact_magnitude'] = cf['magnitude'][i]
                else:
                    row['contact_time'] = ''
                    row['contact_fx'] = ''
                    row['contact_fy'] = ''
                    row['contact_fz'] = ''
                    row['contact_magnitude'] = ''
            else:
                row['contact_time'] = ''
                row['contact_fx'] = ''
                row['contact_fy'] = ''
                row['contact_fz'] = ''
                row['contact_magnitude'] = ''

            rows.append(row)

        # 构建CSV列名
        fieldnames = ['base_time', 'base_x', 'joint_time']
        for j in range(self.num_joints):
            fieldnames.extend([f'target_pos_{j}', f'actual_pos_{j}', f'actual_vel_{j}'])
        fieldnames.append('torque_time')
        for j in range(self.num_joints):
            fieldnames.append(f'torque_{j}')
        fieldnames.extend(['contact_time', 'contact_fx', 'contact_fy', 'contact_fz', 'contact_magnitude'])

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"数据已保存到CSV: {filename}")

    # 从CSV文件加载数据
    @classmethod
    def load_from_csv(cls, filename, joint_names):
        logger = cls(joint_names)

        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 基座数据
                if row['base_time']:
                    logger.base_times.append(float(row['base_time']))
                    logger.base_x_positions.append(float(row['base_x']))

                # 关节状态数据
                if row['joint_time']:
                    logger.joint_times.append(float(row['joint_time']))
                    target_pos = [float(row[f'target_pos_{j}']) for j in range(len(joint_names))]
                    actual_pos = [float(row[f'actual_pos_{j}']) for j in range(len(joint_names))]
                    actual_vel = [float(row[f'actual_vel_{j}']) for j in range(len(joint_names))]
                    logger.target_positions.append(target_pos)
                    logger.actual_positions.append(actual_pos)
                    logger.actual_velocities.append(actual_vel)

                # 力矩数据
                if row['torque_time']:
                    logger.torque_times.append(float(row['torque_time']))
                    torque = [float(row[f'torque_{j}']) for j in range(len(joint_names))]
                    logger.joint_torques.append(torque)

                # 接触力数据
                if row['contact_time']:
                    if 'default_geom' not in logger.contact_forces:
                        logger.contact_forces['default_geom'] = {
                            'time': [], 'fx': [], 'fy': [], 'fz': [], 'magnitude': []
                        }
                    logger.contact_forces['default_geom']['time'].append(float(row['contact_time']))
                    logger.contact_forces['default_geom']['fx'].append(float(row['contact_fx']))
                    logger.contact_forces['default_geom']['fy'].append(float(row['contact_fy']))
                    logger.contact_forces['default_geom']['fz'].append(float(row['contact_fz']))
                    logger.contact_forces['default_geom']['magnitude'].append(float(row['contact_magnitude']))

        return logger


class VisualizationController:
    # 页面名称定义
    PAGES = [
        "运动性能",
        "关节0-3 (后肢+脊柱)",
        "关节4-7 (前肢+脊柱)",
        "足端接触力 (XZ方向)",
    ]

    # 初始化可视化控制器
    def __init__(self, data_logger, joint_names, phase_lag, total_time,
                 analysis_start_time, analysis_end_time, impulse_start, impulse_end, avg_velocity):
        self.data_logger = data_logger
        self.joint_names = joint_names
        self.phase_lag = phase_lag
        self.total_time = total_time
        self.analysis_start_time = analysis_start_time
        self.analysis_end_time = analysis_end_time
        self.impulse_start = impulse_start
        self.impulse_end = impulse_end
        self.avg_velocity = avg_velocity
        self.current_page = 0

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = [FONT_FAMILY]
        plt.rcParams['axes.unicode_minus'] = False

        # 创建图形界面
        self.fig = plt.figure(figsize=FIGURE_SIZE)

        # 添加导航按钮
        ax_prev = plt.axes([0.30, 0.02, 0.15, 0.04])
        ax_next = plt.axes([0.55, 0.02, 0.15, 0.04])
        self.btn_prev = Button(ax_prev, '上一页')
        self.btn_next = Button(ax_next, '下一页')
        self.btn_prev.on_clicked(self.prev_page)
        self.btn_next.on_clicked(self.next_page)

        # 页面指示器
        self.page_text = self.fig.text(
            0.5, 0.03,
            f'页面 {self.current_page+1}/{len(self.PAGES)}: {self.PAGES[self.current_page]}',
            ha='center', fontsize=12, fontweight='bold'
        )

        # 初始化显示
        self.update_plot()

    # 清除所有子图（保留按钮）
    def clear_all_axes(self):
        for ax in self.fig.axes:
            if ax not in [self.btn_prev.ax, self.btn_next.ax]:
                ax.remove()

    # 更新当前页面显示
    def update_plot(self):
        self.clear_all_axes()

        self.page_text.set_text(
            f'页面 {self.current_page+1}/{len(self.PAGES)}: {self.PAGES[self.current_page]}'
        )

        if self.current_page == 0:
            self.show_motion_performance()
        elif self.current_page == 1:
            self.show_joints(0, 4, "后肢+脊柱关节状态")
        elif self.current_page == 2:
            self.show_joints(4, 8, "前肢+脊柱关节状态")
        elif self.current_page == 3:
            self.show_contact_force_analysis()

        plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.12, hspace=0.4)
        self.fig.canvas.draw_idle()

    # 切换到下一页
    def next_page(self, event):
        self.current_page = (self.current_page + 1) % len(self.PAGES)
        self.update_plot()

    # 切换到上一页
    def prev_page(self, event):
        self.current_page = (self.current_page - 1) % len(self.PAGES)
        self.update_plot()

    # 显示运动性能页面
    def show_motion_performance(self):
        ax = self.fig.add_subplot(111)

        times = np.array(self.data_logger.base_times)
        positions = np.array(self.data_logger.base_x_positions)

        # 计算速度
        if len(times) > 1:
            velocities = np.diff(positions) / np.diff(times)
            velocity_times = (times[:-1] + times[1:]) / 2
        else:
            velocities = []
            velocity_times = []

        ax2 = ax.twinx()

        # 绘制位置
        line1, = ax.plot(times, positions, 'b-', linewidth=2.5, label='X位置')

        # 绘制速度
        if len(velocities) > 0:
            line2, = ax2.plot(velocity_times, velocities, 'r-', linewidth=1.5, alpha=0.7, label='速度')

        ax.set_title('机器人运动性能', fontsize=16, pad=20)
        ax.set_xlabel('时间 (秒)', fontsize=14)
        ax.set_ylabel('X位置 (m)', fontsize=14, color='b')
        ax2.set_ylabel('速度 (m/s)', fontsize=14, color='r')
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')

        # 统计信息
        total_distance = self.data_logger.get_total_distance()
        max_velocity = np.max(np.abs(velocities)) if len(velocities) > 0 else 0

        info_text = (
            f'总前进距离: {total_distance:.3f} m\n'
            f'平均速度: {self.avg_velocity:.3f} m/s\n'
            f'最大速度: {max_velocity:.3f} m/s\n'
            f'相位滞后: {self.phase_lag:.4f} rad'
        )

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        lines = [line1]
        labels = ['X位置 (m)']
        if len(velocities) > 0:
            lines.append(line2)
            labels.append('速度 (m/s)')

        ax.legend(lines, labels, loc='upper right', fontsize=11)

    # 显示关节状态页面
    def show_joints(self, start_idx, end_idx, title):
        num_joints = end_idx - start_idx

        joint_times = np.array(self.data_logger.joint_times)

        if len(joint_times) == 0:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                   fontsize=14, color='gray', transform=ax.transAxes)
            ax.set_title(title, fontsize=16)
            return

        torque_times = np.array(self.data_logger.torque_times)

        # 动态创建子图
        axs = []
        for i in range(num_joints):
            if i == 0:
                ax = self.fig.add_subplot(num_joints, 1, i+1)
            else:
                ax = self.fig.add_subplot(num_joints, 1, i+1, sharex=axs[0])
            axs.append(ax)

        # 绘制每个关节的数据
        for i, ax in enumerate(axs):
            joint_idx = start_idx + i

            target_pos = np.array([data[joint_idx] for data in self.data_logger.target_positions])
            actual_pos = np.array([data[joint_idx] for data in self.data_logger.actual_positions])

            ax2 = ax.twinx()

            # 绘制期望和实际角度（左轴）
            line1, = ax.plot(joint_times, target_pos, 'r--', linewidth=1.5,
                            label='期望角度', alpha=0.8)
            line2, = ax.plot(joint_times, actual_pos, 'b-', linewidth=2.0,
                            label='实际角度', alpha=0.8)

            # 绘制力矩（右轴）
            if len(torque_times) > 0:
                torques = np.array([data[joint_idx] for data in self.data_logger.joint_torques])
                if len(torque_times) > 5000:
                    step = len(torque_times) // 5000
                    line3, = ax2.plot(torque_times[::step], torques[::step],
                                     'g-', linewidth=1.0, alpha=0.5, label='力矩')
                else:
                    line3, = ax2.plot(torque_times, torques,
                                     'g-', linewidth=1.0, alpha=0.5, label='力矩')

            ax.set_title(f'关节 {joint_idx}: {self.joint_names[joint_idx]}',
                        fontsize=11, pad=12)
            ax.set_ylabel('角度 (rad)', fontsize=9, color='b')
            ax2.set_ylabel('力矩 (N·m)', fontsize=9, color='g')
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax2.tick_params(axis='y', labelcolor='g')
            ax.grid(True, linestyle='--', alpha=0.6)

            if i == num_joints - 1:
                ax.set_xlabel('时间 (秒)', fontsize=9)

            # 计算误差统计
            if len(actual_pos) > 0:
                tracking_error = np.mean(np.abs(actual_pos - target_pos))
                max_torque = np.max(np.abs(torques)) if len(torque_times) > 0 else 0

                ax.text(0.02, 0.98,
                       f'跟踪误差: {tracking_error:.4f} rad\n最大力矩: {max_torque:.4f} N·m',
                       transform=ax.transAxes, fontsize=7,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

            # 图例（仅第一个子图显示）
            if i == 0:
                lines = [line1, line2]
                labels = ['期望角度', '实际角度']
                if len(torque_times) > 0:
                    lines.append(line3)
                    labels.append('力矩')
                ax.legend(lines, labels, loc='upper right', fontsize=8, ncol=3)

    # 显示足端接触力分析页面
    def show_contact_force_analysis(self):
        if not self.data_logger.contact_forces:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, '无接触力数据', ha='center', va='center',
                   fontsize=14, color='gray', transform=ax.transAxes)
            return

        # 取第一个几何体的数据
        for geom_name, forces in self.data_logger.contact_forces.items():
            if not forces['time']:
                continue

            times_array = np.array(forces['time'])
            fx_array = np.array(forces['fx'])
            fz_array = np.array(forces['fz'])

            # 选择分析时间段
            mask = (times_array >= self.analysis_start_time) & (times_array <= self.analysis_end_time)
            if np.sum(mask) == 0:
                continue

            analysis_times = times_array[mask] - self.analysis_start_time
            analysis_fx = fx_array[mask]
            analysis_fz = -fz_array[mask]

            # 计算XZ合力
            analysis_fxz_magnitude = np.sqrt(analysis_fx**2 + analysis_fz**2)
            analysis_fxz_angle = np.degrees(np.arctan2(analysis_fz, analysis_fx))

            max_fx = np.max(np.abs(analysis_fx))
            max_fz = np.max(np.abs(analysis_fz))
            max_fxz = np.max(analysis_fxz_magnitude)
            mean_fx = np.mean(analysis_fx)
            mean_fz = np.mean(analysis_fz)
            mean_fxz = np.mean(analysis_fxz_magnitude)

            # 计算冲量
            impulse_mask = (times_array >= self.impulse_start) & (times_array <= self.impulse_end)
            if np.sum(impulse_mask) >= 2:
                impulse_times = times_array[impulse_mask]
                impulse_fx = fx_array[impulse_mask]
                impulse_fz = -fz_array[impulse_mask]

                time_steps = np.diff(impulse_times)
                fx_mid = (impulse_fx[:-1] + impulse_fx[1:]) / 2
                fz_mid = (impulse_fz[:-1] + impulse_fz[1:]) / 2

                impulse_x = np.sum(fx_mid * time_steps)
                impulse_z = np.sum(fz_mid * time_steps)
                impulse_magnitude = np.sqrt(impulse_x**2 + impulse_z**2)
            else:
                impulse_x = impulse_z = impulse_magnitude = 0

            # 子图1: X和Z方向力随时间变化
            ax1 = self.fig.add_subplot(3, 1, 1)

            ax1.plot(analysis_times, analysis_fx, 'b-',
                    label='Fx (水平力)', linewidth=2, alpha=0.8)
            ax1.plot(analysis_times, analysis_fz, 'r-',
                    label='Fz (垂直力)', linewidth=2, alpha=0.8)

            impulse_start_rel = self.impulse_start - self.analysis_start_time
            impulse_end_rel = self.impulse_end - self.analysis_start_time
            ax1.axvspan(impulse_start_rel, impulse_end_rel, alpha=0.2, color='yellow',
                       label=f'冲量区间 ({self.impulse_start:.2f}-{self.impulse_end:.2f}s)')

            ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

            ax1.set_ylabel('力 (N)', fontsize=12)
            ax1.set_title(f'X和Z方向地面反作用力 ({geom_name})', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3)

            ax1.text(0.02, 0.98,
                    f'Max |Fx|: {max_fx:.2f}N\nMax |Fz|: {max_fz:.2f}N\n'
                    f'Ix = {impulse_x:.4f} N·s\nIz = {impulse_z:.4f} N·s',
                    transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # 子图2: XZ合力方向随时间变化
            ax2 = self.fig.add_subplot(3, 1, 2)

            ax2.plot(analysis_times, analysis_fxz_angle, 'g-',
                    label='合力方向角度', linewidth=2, alpha=0.8)

            ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='0° (纯水平)')
            ax2.axhline(y=90, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='90° (垂直向上)')
            ax2.axhline(y=-90, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='-90° (垂直向下)')

            angle_range = max(abs(np.min(analysis_fxz_angle)), abs(np.max(analysis_fxz_angle)))
            y_limit = max(angle_range * 1.1, 180)
            ax2.set_ylim(-y_limit, y_limit)

            ax2.set_ylabel('角度 (度)', fontsize=12)
            ax2.set_title('XZ合力方向', fontsize=14, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=9, ncol=2)
            ax2.grid(True, alpha=0.3)

            ax2.fill_between(analysis_times, 0, analysis_fxz_angle,
                           where=(analysis_fxz_angle > 0),
                           color='red', alpha=0.2)
            ax2.fill_between(analysis_times, 0, analysis_fxz_angle,
                           where=(analysis_fxz_angle < 0),
                           color='blue', alpha=0.2)

            mean_angle = np.mean(analysis_fxz_angle)
            std_angle = np.std(analysis_fxz_angle)
            ax2.text(0.02, 0.98, f'平均角度: {mean_angle:.1f}°\n标准差: {std_angle:.1f}°',
                    transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

            # 子图3: XZ合力大小随时间变化
            ax3 = self.fig.add_subplot(3, 1, 3)

            ax3.plot(analysis_times, analysis_fxz_magnitude, 'm-',
                    label='XZ合力大小', linewidth=2.5, alpha=0.8)

            ax3.axhline(y=mean_fxz, color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'平均力: {mean_fxz:.2f}N')
            ax3.axhline(y=max_fxz, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'最大力: {max_fxz:.2f}N')

            ax3.fill_between(analysis_times, 0, analysis_fxz_magnitude,
                           color='purple', alpha=0.15)

            ax3.set_xlabel('时间 (s)', fontsize=12)
            ax3.set_ylabel('合力大小 (N)', fontsize=12)
            ax3.set_title(f'XZ合力大小 (平均速度: {self.avg_velocity:.3f} m/s)',
                         fontsize=14, fontweight='bold')
            ax3.legend(loc='upper right', fontsize=10)
            ax3.grid(True, alpha=0.3)

            ax3.text(0.02, 0.98,
                    f'峰值力: {max_fxz:.2f}N\n平均力: {mean_fxz:.2f}N\n'
                    f'力波动: {np.std(analysis_fxz_magnitude):.2f}N\n'
                    f'合冲量: {impulse_magnitude:.4f} N·s',
                    transform=ax3.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

            # 只处理第一个几何体
            break


# 启动可视化界面
def show_visualization(data_logger, joint_names, phase_lag, total_time,
                      analysis_start_time=DEFAULT_ANALYSIS_START_TIME,
                      analysis_end_time=DEFAULT_ANALYSIS_END_TIME,
                      impulse_start=DEFAULT_IMPULSE_START,
                      impulse_end=DEFAULT_IMPULSE_END,
                      avg_velocity=0.0):
    controller = VisualizationController(
        data_logger, joint_names, phase_lag, total_time,
        analysis_start_time, analysis_end_time,
        impulse_start, impulse_end, avg_velocity
    )
    plt.show()


# 从CSV文件加载并可视化
def load_and_visualize(data_file, joint_names, phase_lag=DEFAULT_PHASE_LAG,
                       total_time=DEFAULT_TOTAL_TIME,
                       analysis_start_time=DEFAULT_ANALYSIS_START_TIME,
                       analysis_end_time=DEFAULT_ANALYSIS_END_TIME,
                       impulse_start=DEFAULT_IMPULSE_START,
                       impulse_end=DEFAULT_IMPULSE_END):
    print(f"从文件加载数据: {data_file}")
    data_logger = DataLogger.load_from_csv(data_file, joint_names)

    avg_velocity = data_logger.calculate_average_velocity(1.0, 4.0)
    print(f"总前进距离: {data_logger.get_total_distance():.4f} m")
    print(f"平均速度 (1s-4s): {avg_velocity:.4f} m/s")

    show_visualization(
        data_logger, joint_names, phase_lag, total_time,
        analysis_start_time, analysis_end_time,
        impulse_start, impulse_end, avg_velocity
    )


# 主程序入口
if __name__ == '__main__':
    if os.path.exists(CSV_DATA_FILE):
        load_and_visualize(CSV_DATA_FILE, JOINT_NAMES)
    else:
        print(f"未找到数据文件 '{CSV_DATA_FILE}'")
        print("请先运行 Locomotion.py 生成数据文件")