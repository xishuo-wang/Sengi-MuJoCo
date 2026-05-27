"""
冲量分析模块 - 独立的地面接触力冲量分析工具
可以从CSV文件加载数据并生成XZ方向力分析图
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os


# ==================== 配置参数 ====================
# 数据文件路径
CSV_DATA_FILE = r"D:\Code\Sengi-MuJoCo\Sengi-double-MuJoCo\Data_Locomotion\Simulation_Data_260527_1.csv"

# 分析时间段配置（秒）
ANALYSIS_START_TIME = 0.0
ANALYSIS_END_TIME = 5.0

# 冲量计算区间（秒）
IMPULSE_START = 0.10
IMPULSE_END = 0.20

# 平均速度计算区间（秒）
VELOCITY_START_TIME = 1.0
VELOCITY_END_TIME = 4.0

# 图表显示配置
FIGURE_SIZE = (16, 12)
FONT_FAMILY = 'SimHei'


# ==================== 数据加载函数 ====================
# 从CSV加载接触力数据
def load_contact_data_from_csv(filename):
    times = []
    fx = []
    fy = []
    fz = []
    magnitude = []
    base_times = []
    base_x = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('contact_time'):
                times.append(float(row['contact_time']))
                fx.append(float(row['contact_fx']))
                fy.append(float(row['contact_fy']))
                fz.append(float(row['contact_fz']))
                magnitude.append(float(row['contact_magnitude']))

            if row.get('base_time'):
                base_times.append(float(row['base_time']))
                base_x.append(float(row['base_x']))

    return {
        'times': np.array(times),
        'fx': np.array(fx),
        'fy': np.array(fy),
        'fz': np.array(fz),
        'magnitude': np.array(magnitude),
        'base_times': np.array(base_times),
        'base_x': np.array(base_x)
    }


# 计算平均速度
def calculate_average_velocity(base_times, base_x, start_time, end_time):
    if len(base_times) < 2:
        return 0.0

    start_idx = np.argmin(np.abs(base_times - start_time))
    end_idx = np.argmin(np.abs(base_times - end_time))

    if start_idx >= end_idx:
        return 0.0

    pos_start = base_x[start_idx]
    pos_end = base_x[end_idx]
    time_diff = base_times[end_idx] - base_times[start_idx]

    if time_diff > 0:
        return (pos_end - pos_start) / time_diff
    return 0.0


# ==================== 冲量计算函数 ====================
# 使用梯形法则计算冲量
def calculate_impulse(times, fx, fz, start_time, end_time):
    mask = (times >= start_time) & (times <= end_time)

    if np.sum(mask) < 2:
        return 0.0, 0.0, 0.0

    impulse_times = times[mask]
    impulse_fx = fx[mask]
    impulse_fz = fz[mask]

    time_steps = np.diff(impulse_times)
    fx_mid = (impulse_fx[:-1] + impulse_fx[1:]) / 2
    fz_mid = (impulse_fz[:-1] + impulse_fz[1:]) / 2

    impulse_x = np.sum(fx_mid * time_steps)
    impulse_z = np.sum(fz_mid * time_steps)
    impulse_magnitude = np.sqrt(impulse_x**2 + impulse_z**2)

    return impulse_x, impulse_z, impulse_magnitude


# ==================== 可视化函数 ====================
# 绘制XZ方向力分析图
def plot_impulse_analysis(data, avg_velocity, analysis_start, analysis_end,
                         impulse_start, impulse_end):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = [FONT_FAMILY]
    plt.rcParams['axes.unicode_minus'] = False

    # 提取分析时间段的数据
    mask = (data['times'] >= analysis_start) & (data['times'] <= analysis_end)
    analysis_times = data['times'][mask] - analysis_start
    analysis_fx = data['fx'][mask]
    analysis_fz = -data['fz'][mask]

    # 计算XZ合力
    analysis_fxz_magnitude = np.sqrt(analysis_fx**2 + analysis_fz**2)
    analysis_fxz_angle = np.degrees(np.arctan2(analysis_fz, analysis_fx))

    # 统计信息
    max_fx = np.max(np.abs(analysis_fx))
    max_fz = np.max(np.abs(analysis_fz))
    max_fxz = np.max(analysis_fxz_magnitude)
    mean_fx = np.mean(analysis_fx)
    mean_fz = np.mean(analysis_fz)
    mean_fxz = np.mean(analysis_fxz_magnitude)

    # 计算冲量
    impulse_x, impulse_z, impulse_magnitude = calculate_impulse(
        data['times'], data['fx'], -data['fz'], impulse_start, impulse_end
    )

    # 计算区间最大力
    interval_mask = (data['times'] >= impulse_start) & (data['times'] <= impulse_end)
    interval_fx = data['fx'][interval_mask]
    interval_fz = -data['fz'][interval_mask]
    interval_fxz = np.sqrt(interval_fx**2 + interval_fz**2)
    interval_max_fx = np.max(np.abs(interval_fx))
    interval_max_fz = np.max(np.abs(interval_fz))
    interval_max_fxz = np.max(interval_fxz)

    # 创建图表
    fig = plt.figure(figsize=FIGURE_SIZE)

    # 子图1: X和Z方向力随时间变化
    ax1 = fig.add_subplot(3, 1, 1)

    ax1.plot(analysis_times, analysis_fx, 'b-',
            label='Fx (水平力)', linewidth=2, alpha=0.8)
    ax1.plot(analysis_times, analysis_fz, 'r-',
            label='Fz (垂直力)', linewidth=2, alpha=0.8)

    # 高亮冲量计算区间
    impulse_start_rel = impulse_start - analysis_start
    impulse_end_rel = impulse_end - analysis_start
    ax1.axvspan(impulse_start_rel, impulse_end_rel, alpha=0.2, color='yellow',
               label=f'冲量区间 ({impulse_start:.2f}-{impulse_end:.2f}s)')

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    ax1.set_ylabel('力 (N)', fontsize=12)
    ax1.set_title(f'X和Z方向地面反作用力', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax1.text(0.02, 0.98,
            f'Max |Fx|: {max_fx:.2f}N\nMax |Fz|: {max_fz:.2f}N\n'
            f'Ix = {impulse_x:.4f} N·s\nIz = {impulse_z:.4f} N·s',
            transform=ax1.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 子图2: XZ合力方向随时间变化
    ax2 = fig.add_subplot(3, 1, 2)

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
                   color='red', alpha=0.2, label='向上力区域')
    ax2.fill_between(analysis_times, 0, analysis_fxz_angle,
                   where=(analysis_fxz_angle < 0),
                   color='blue', alpha=0.2, label='向下力区域')

    mean_angle = np.mean(analysis_fxz_angle)
    std_angle = np.std(analysis_fxz_angle)
    ax2.text(0.02, 0.98, f'平均角度: {mean_angle:.1f}°\n标准差: {std_angle:.1f}°',
            transform=ax2.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 子图3: XZ合力大小随时间变化
    ax3 = fig.add_subplot(3, 1, 3)

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
    ax3.set_title(f'XZ合力大小 (平均速度: {avg_velocity:.3f} m/s)',
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

    plt.tight_layout()
    plt.show()


# ==================== 主函数 ====================
# 运行冲量分析
def run_impulse_analysis(csv_file, analysis_start, analysis_end,
                        impulse_start, impulse_end, velocity_start, velocity_end):
    print(f"加载数据文件: {csv_file}")

    # 加载数据
    data = load_contact_data_from_csv(csv_file)

    if len(data['times']) == 0:
        print("错误: 未找到接触力数据")
        return

    print(f"加载了 {len(data['times'])} 个接触力数据点")

    # 计算平均速度
    avg_velocity = calculate_average_velocity(
        data['base_times'], data['base_x'], velocity_start, velocity_end
    )

    # 计算冲量（矢量和）
    impulse_x, impulse_z, impulse_magnitude = calculate_impulse(
        data['times'], data['fx'], -data['fz'], impulse_start, impulse_end
    )

    mask = (data['times'] >= impulse_start) & (data['times'] <= impulse_end)
    impulse_fx = data['fx'][mask]
    impulse_fz = -data['fz'][mask]

    max_fx = np.max(np.abs(impulse_fx))
    max_fz = np.max(np.abs(impulse_fz))
    max_fxz = np.max(np.sqrt(impulse_fx**2 + impulse_fz**2))

    # 打印简化结果
    print("\n" + "=" * 50)
    print("冲量分析结果")
    print("=" * 50)
    print(f"区间持续时间: {impulse_end - impulse_start:.4f}s")
    print(f"平均速度: {avg_velocity:.4f} m/s")
    print(f"冲量计算区间: {impulse_start:.2f}s - {impulse_end:.2f}s")
    print(f"Ix (水平): {impulse_x:.6f} N·s")
    print(f"Iz (垂直): {impulse_z:.6f} N·s")
    print(f"|I| (矢量和): {impulse_magnitude:.6f} N·s")
    print(f"Fx区间最大力: {max_fx:.4f} N")
    print(f"Fz区间最大力: {max_fz:.4f} N")
    print(f"合力最大力: {max_fxz:.4f} N")
    print("=" * 50)

    # 生成图表（不保存）
    plot_impulse_analysis(
        data, avg_velocity, analysis_start, analysis_end,
        impulse_start, impulse_end
    )


# 主程序入口
if __name__ == '__main__':
    if os.path.exists(CSV_DATA_FILE):
        run_impulse_analysis(
            CSV_DATA_FILE,
            ANALYSIS_START_TIME,
            ANALYSIS_END_TIME,
            IMPULSE_START,
            IMPULSE_END,
            VELOCITY_START_TIME,
            VELOCITY_END_TIME
        )
    else:
        print(f"未找到数据文件 '{CSV_DATA_FILE}'")
        print("请确认文件路径是否正确")