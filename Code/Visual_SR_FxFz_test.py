import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ==========================================
# 1. 全局字体与样式设置
# ==========================================
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'

# 读取CSV文件
csv_path = r"D:\Software\测力台软件\Data\pi_10_260520_2.csv"
df = pd.read_csv(csv_path)

# 将时间戳转换为相对于第一个时间的秒数
df['m_Time'] = pd.to_datetime(df['m_Time'])
df['Time_sec'] = (df['m_Time'] - df['m_Time'].iloc[0]).dt.total_seconds()

# ==========================================
# 2. 数据提取
# ==========================================
time = df['Time_sec']

# 提取 Sensor 17 的数据 (左腿)
fx_17 = df.get('Fx17', pd.Series(np.zeros(len(df))))
fz_17 = df.get('Fz17', pd.Series(np.zeros(len(df))))

# 提取 Sensor 23 的数据 (右腿)
fx_23 = df.get('Fx23', pd.Series(np.zeros(len(df))))
fz_23 = df.get('Fz23', pd.Series(np.zeros(len(df))))

# 设置颜色
color_s17 = (0.40, 0.60, 0.90)  # 左腿蓝色
color_s23 = (0.75, 0.50, 0.50)  # 右腿红褐色

# 设置固定的时间轴显示范围
t_min, t_max = 2.9, 3.8

# ==========================================
# 3. 绘制原始支反力曲线 (不再求平均)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.canvas.manager.set_window_title('Individual Leg Ground Reaction Forces')

# ---- 上图：水平分力 Fx ----
ax1.plot(time, fx_17, color=color_s17, linewidth=2.5, label='Sensor 17 (Left Leg)')
ax1.plot(time, fx_23, color=color_s23, linewidth=2.5, label='Sensor 23 (Right Leg)')
ax1.set_title('Horizontal Ground Reaction Force ($F_x$)', fontsize=18)
ax1.set_ylabel('Force $F_x$ (N)', fontsize=16)
ax1.legend(fontsize=14, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(t_min, t_max)
ax1.tick_params(axis='both', labelsize=14)

# ---- 下图：竖直分力 Fz ----
ax2.plot(time, fz_17, color=color_s17, linewidth=2.5, label='Sensor 17 (Left Leg)')
ax2.plot(time, fz_23, color=color_s23, linewidth=2.5, label='Sensor 23 (Right Leg)')
ax2.set_title('Vertical Ground Reaction Force ($F_z$)', fontsize=18)
ax2.set_xlabel('Time (s)', fontsize=16)
ax2.set_ylabel('Force $F_z$ (N)', fontsize=16)
ax2.legend(fontsize=14, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(t_min, t_max)
ax2.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.show()