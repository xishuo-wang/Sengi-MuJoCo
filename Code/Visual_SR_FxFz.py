import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ==========================================
# 1. 全局字体与样式设置 (对齐 MATLAB 视觉风格)
# ==========================================
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'

# 读取CSV文件
csv_path = r"D:\Software\测力台软件\Data\pi_10_260522_14.csv"
df = pd.read_csv(csv_path)

# 将时间戳转换为相对于第一个时间的秒数
df['m_Time'] = pd.to_datetime(df['m_Time'])
df['Time_sec'] = (df['m_Time'] - df['m_Time'].iloc[0]).dt.total_seconds()

# ==========================================
# 2. 数据提取与预处理
# ==========================================
time = df['Time_sec']

# 提取 Sensor 17 的数据 (假设为左侧)
fx_17 = df.get('Fx17', pd.Series(np.zeros(len(df))))
fz_17 = df.get('Fz17', pd.Series(np.zeros(len(df))))
fy_17 = df.get('Fy17', pd.Series(np.zeros(len(df)))) # 用于计算合力
mag_17 = np.sqrt(fx_17**2 + fy_17**2 + fz_17**2)

# 提取 Sensor 23 的数据 (假设为右侧)
fx_23 = df.get('Fx23', pd.Series(np.zeros(len(df))))
fz_23 = df.get('Fz23', pd.Series(np.zeros(len(df))))
fy_23 = df.get('Fy23', pd.Series(np.zeros(len(df))))
mag_23 = np.sqrt(fx_23**2 + fy_23**2 + fz_23**2)

# 计算两条腿的平均支反力
fx_avg = (fx_17 + fx_23) / 2
fz_avg = -(fz_17 + fz_23) / 2

# 设置颜色变量 (严格对应 MATLAB 的 RGB 色值)
color_s17 = (0.40, 0.60, 0.90)  # MATLAB: colorlf
color_s23 = (0.75, 0.50, 0.50)  # MATLAB: colorrf
color_avg_fx = '#77AC30'
color_avg_fz = '#D95319'

# 设置固定的时间轴显示范围
t_min, t_max = 3.5, 4.4

# ==========================================
# 3. 绘制图 1: 左右两侧受力独立分量分析
# ==========================================
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
fig1.canvas.manager.set_window_title('Ground Reaction Forces Component Analysis')

# Subplot 1: Total Magnitude (总合力)
ax1.plot(time, mag_17, color=color_s17, linewidth=2.5, label='Sensor 17')
ax1.plot(time, mag_23, color=color_s23, linewidth=2.5, label='Sensor 23')
ax1.set_title('Total Ground Reaction Force Magnitude', fontsize=18)
ax1.set_ylabel('Total Force (N)', fontsize=16)
ax1.legend(fontsize=14, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(t_min, t_max)
ax1.tick_params(axis='both', labelsize=14)

# Subplot 2: Fx (水平分力)
ax2.plot(time, fx_17, color=color_s17, linewidth=2.5, label='Sensor 17')
ax2.plot(time, fx_23, color=color_s23, linewidth=2.5, label='Sensor 23')
ax2.set_title('GRF X-axis Component ($F_x$)', fontsize=18)
ax2.set_ylabel('Force $F_x$ (N)', fontsize=16)
ax2.legend(fontsize=14, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(t_min, t_max)
ax2.tick_params(axis='both', labelsize=14)

# Subplot 3: Fz (竖直分力)
ax3.plot(time, fz_17, color=color_s17, linewidth=2.5, label='Sensor 17')
ax3.plot(time, fz_23, color=color_s23, linewidth=2.5, label='Sensor 23')
ax3.set_title('GRF Z-axis Component ($F_z$)', fontsize=18)
ax3.set_xlabel('Time (s)', fontsize=16)
ax3.set_ylabel('Force $F_z$ (N)', fontsize=16)
ax3.legend(fontsize=14, loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(t_min, t_max)
ax3.tick_params(axis='both', labelsize=14)

plt.tight_layout()

# ==========================================
# 4. 绘制图 2: 后腿平均支反力 (水平与竖直对比)
# ==========================================
fig2, ax_avg = plt.subplots(figsize=(10, 6))
fig2.canvas.manager.set_window_title('Average Ground Reaction Forces (Fx & Fz)')

# 绘制平均曲线
ax_avg.plot(time, fx_avg, color=color_avg_fx, linewidth=3, label='Average $F_x$ (Horizontal)')
ax_avg.plot(time, fz_avg, color=color_avg_fz, linewidth=3, label='Average $F_z$ (Vertical)')

# 样式设置
ax_avg.set_title('Average Ground Reaction Force (Sensor 17 & 23)', fontsize=22)
ax_avg.set_xlabel('Time (s)', fontsize=18)
ax_avg.set_ylabel('Average Force (N)', fontsize=18)
ax_avg.legend(fontsize=16, loc='best')
ax_avg.grid(True, alpha=0.3)
ax_avg.set_xlim(t_min, t_max)
ax_avg.tick_params(axis='both', labelsize=16)

plt.tight_layout()

# ==========================================
# 5. 绘制图 3: 最大平均合力向量朝向可视化 (新增)
# ==========================================
# 在 3.2s - 4.1s 区间内寻找最大合力
mask = (time >= t_min) & (time <= t_max)
fx_avg_window = fx_avg[mask]
fz_avg_window = fz_avg[mask]
mag_avg_window = np.sqrt(fx_avg_window**2 + fz_avg_window**2)

# 找到最大值的索引
if not mag_avg_window.empty:
    max_idx = mag_avg_window.idxmax()
    peak_fx = fx_avg[max_idx]
    peak_fz = fz_avg[max_idx]
    peak_mag = mag_avg_window[max_idx]
    # 计算朝向角度
    angle_deg = np.degrees(np.arctan2(peak_fz, peak_fx))

    # 创建独立图形
    fig3, ax_vec = plt.subplots(figsize=(8, 8))
    fig3.canvas.manager.set_window_title('Maximum Force Vector Orientation')

    # 绘制基础坐标轴十字线
    ax_vec.axhline(0, color='black', linewidth=1.5)
    ax_vec.axvline(0, color='black', linewidth=1.5)

    # 绘制向量箭头 (使用 quiver，比例 1:1)
    ax_vec.quiver(0, 0, peak_fx, peak_fz, angles='xy', scale_units='xy', scale=1,
                  color='#EDB120', width=0.015, headwidth=4, headlength=6, zorder=3)

    # 动态计算坐标轴范围以保证完美物理比例
    max_val = max(abs(peak_fx), abs(peak_fz)) * 1.2
    x_min = -max_val * 0.2 if peak_fx > 0 else -max_val
    x_max = max_val if peak_fx > 0 else max_val * 0.2
    ax_vec.set_xlim(x_min, x_max)
    ax_vec.set_ylim(-max_val * 0.1, max_val)
    ax_vec.set_aspect('equal') # 强制 1:1 比例，角度绝对真实

    # 在图上标注数值和角度信息
    ax_vec.text(peak_fx/2, peak_fz/2 + max_val*0.05,
                f'Mag: {peak_mag:.2f} N\nAngle: {angle_deg:.1f}°',
                color='#D95319', fontsize=16, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 样式设置
    ax_vec.set_title('Direction of Maximum Average Resultant Force', fontsize=20)
    ax_vec.set_xlabel('Horizontal Force $F_x$ (N)', fontsize=16)
    ax_vec.set_ylabel('Vertical Force $F_z$ (N)', fontsize=16)
    ax_vec.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax_vec.tick_params(axis='both', labelsize=14)

    plt.tight_layout()

# 统一显示所有图形
plt.show()