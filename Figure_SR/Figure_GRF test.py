import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap  


# ==================== 配置参数 ====================
# 文件路径
COLORMAP_PATH = r"D:\Code\Sengi-MuJoCo\Figure_SR\mycolora.mat"

CSV_PATH = r"D:\Software\WeiChat\xwechat_files\wxid_vbfmp2aqmane22_098e\msg\file\2026-05\Robot_Dynamics_Data.csv"
TIME_START = 1.3  # 修改为实际需要的时间范围
TIME_END = TIME_START+0.25

# 扇形计算阈值（过滤掉幅值过小的点）
SECTOR_THRESHOLD = 0.1

# 刻度间隔
TICK_INTERVAL = 0.25

# 图形尺寸（英寸）
FIGURE_SIZE = (4.6, 4.0)

# 轨迹线宽度
LINE_WIDTH = 4

# 扇形样式
SECTOR_FACECOLOR = 'lightgray'
SECTOR_EDGECOLOR = 'gray'
SECTOR_ALPHA = 0.6
SECTOR_LINEWIDTH = 0

# 参考线样式
REFERENCE_LINE_COLOR = 'white'
REFERENCE_LINE_STYLE = '--'
REFERENCE_LINE_WIDTH = 4
REFERENCE_LINE_ALPHA = 1

# 坐标轴样式
AXIS_LINEWIDTH = 3
TICK_PADDING = 8  # 刻度标签与轴线的距离（点）

SAVE_FIGURE = True  # 是否自动保存图片
SAVE_DIRECTORY = r"D:\Software\测力台软件\Data\Figure" 
# ==================== 数据加载 ====================
df = pd.read_csv(CSV_PATH)

# 映射CSV列到变量（新格式：不考虑Y方向）
forceX = df['Fx_Left_N'].values     # 左腿X方向力
forceZ = df['Fz_Left_N'].values     # 左腿Z方向力
forceXr = df['Fx_Right_N'].values   # 右腿X方向力
forceZr = df['Fz_Right_N'].values   # 右腿Z方向力

# 创建时间步力数据数组（只包含X和Z方向，右腿用于绘图）
# 注意：这里不再需要Y方向，所以数组只有2列
timeStepForcer = np.column_stack([forceXr, forceZr])  # 右腿 [Fx, Fz]

print(f"数据加载完成: {len(timeStepForcer)} 行")

# ==================== 时间窗口选择 ====================
# 时间列已经是数值格式，直接使用
time_array = df['Time_s'].values
startpoint = np.argmax(time_array >= TIME_START)
endpoint = np.argmax(time_array >= TIME_END)

print(f"时间范围: {time_array[startpoint]:.3f}s - {time_array[endpoint]:.3f}s")
print(f"数据点范围: {startpoint} - {endpoint} (共{endpoint - startpoint}个点)")

# ==================== 数据归一化与方向翻转 ====================
# 提取选定时间窗口的力数据（右腿）
Fx_raw = -timeStepForcer[startpoint:endpoint, 0]
Fz_raw = timeStepForcer[startpoint:endpoint, 1]  # Fz取反，使方向向上为正

# 使用Fz最大值进行归一化（X轴和Z轴使用相同的归一化因子）
fz_max = np.max(np.abs(Fz_raw))
if fz_max > 0:
    xx = Fx_raw / fz_max
    yy = Fz_raw / fz_max
else:
    xx = Fx_raw
    yy = Fz_raw

print(f"归一化方式: Z轴最大值归一化")
print(f"Fz最大值（归一化因子）: {fz_max:.2f}")
print(f"全部数据点数: {len(xx)}")
print(f"归一化Fx范围: {xx.min():.3f} 到 {xx.max():.3f}")
print(f"归一化Fz范围: {yy.min():.3f} 到 {yy.max():.3f}")

# ==================== 数据过滤（仅用于扇形计算） ====================
mask = (np.abs(xx) > SECTOR_THRESHOLD) & (np.abs(yy) > SECTOR_THRESHOLD)
xx_filtered = xx[mask]
yy_filtered = yy[mask]

removed_count = len(xx) - len(xx_filtered)
print(f"\n扇形计算数据过滤结果 (阈值={SECTOR_THRESHOLD}):")
print(f"全部数据点: {len(xx)}")
print(f"用于扇形计算的数据点: {len(xx_filtered)}")
print(f"过滤掉的数据点: {removed_count}")

# ==================== 自适应扇形计算 ====================
if len(xx_filtered) < 2:
    raise ValueError("错误：过滤后数据点不足，无法计算扇形！")

# 计算每个数据点与X轴正半轴的夹角，转换为与Z轴正半轴的夹角
angles_x_axis = np.arctan2(yy_filtered, xx_filtered)
angles_y_axis = angles_x_axis - np.pi / 2
angles_y_axis = np.arctan2(np.sin(angles_y_axis), np.cos(angles_y_axis))  # 标准化到[-π, π]

# 找到与Z轴正半轴的最大和最小夹角
min_angle_y = np.min(angles_y_axis)
max_angle_y = np.max(angles_y_axis)

# 转换为matplotlib Wedge所需的角度（从X轴正半轴逆时针）
theta1_deg = np.degrees(np.pi / 2 + min_angle_y)
theta2_deg = np.degrees(np.pi / 2 + max_angle_y)

# 确保theta1 < theta2
if theta1_deg > theta2_deg:
    theta1_deg, theta2_deg = theta2_deg, theta1_deg

sector_angle = theta2_deg - theta1_deg

# 计算扇形半径（过滤后数据的最大合力）
resultant_force_filtered = np.sqrt(xx_filtered**2 + yy_filtered**2)
Fmaxradius = np.max(resultant_force_filtered)

print("\n" + "=" * 60)
print("自适应扇形统计信息（以Z轴正半轴为参考）:")
print(f"与Z轴最小夹角: {np.degrees(min_angle_y):.2f}°")
print(f"与Z轴最大夹角: {np.degrees(max_angle_y):.2f}°")
print(f"扇形总角度: {sector_angle:.2f}°")
print(f"扇形绘制范围: {theta1_deg:.2f}° 到 {theta2_deg:.2f}°")
print(f"扇形半径（最大合力）: {Fmaxradius:.3f}")

# ==================== 计算归一化数据统计 ====================
fx_min, fx_max = xx.min(), xx.max()
fy_min, fy_max = yy.min(), yy.max()
resultant_force_all = np.sqrt(xx**2 + yy**2)
max_radius = max(np.max(resultant_force_all), Fmaxradius)

print(f"\n归一化数据范围:")
print(f"  Fx: [{fx_min:.3f}, {fx_max:.3f}]")
print(f"  Fz: [{fy_min:.3f}, {fy_max:.3f}]")
print(f"  最大合力半径: {max_radius:.3f}")

# ==================== 创建图形 ====================
fig = plt.figure(figsize=FIGURE_SIZE)
ax = fig.add_subplot(111)

# ==================== 绘制彩色轨迹线（使用全部数据） ====================
da = np.arange(startpoint, endpoint)

# 创建线段集合
points = np.array([xx, yy]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 加载自定义颜色映射
try:
    mycolora = loadmat(COLORMAP_PATH)
    if 'mycolora' in mycolora:
        cmap_data = mycolora['mycolora']
        mycolormap = LinearSegmentedColormap.from_list('mycolora', cmap_data)
    else:
        print("警告: mycolora.mat中未找到'mycolora'变量，使用默认colormap")
        mycolormap = plt.cm.viridis
except Exception as e:
    print(f"警告: 无法加载mycolora.mat ({e})，使用默认colormap")
    mycolormap = plt.cm.viridis

norm = plt.Normalize(startpoint, endpoint)
lc = LineCollection(segments, cmap=mycolormap, norm=norm, linewidth=LINE_WIDTH)
lc.set_array(da[:-1])
ax.add_collection(lc)

# ==================== 绘制自适应扇形 ====================
wedge = Wedge(
    center=(0, 0),
    r=Fmaxradius,
    theta1=theta1_deg,
    theta2=theta2_deg,
    facecolor=SECTOR_FACECOLOR,
    edgecolor=SECTOR_EDGECOLOR,
    alpha=SECTOR_ALPHA,
    linewidth=SECTOR_LINEWIDTH
)
ax.add_patch(wedge)

# ==================== 绘制参考线：从原点向Z轴正方向的白色虚线 ====================
# 计算虚线长度（使用Z轴最大范围）
y_max_limit = max(fy_max, Fmaxradius) + 0.05
y_max_limit = np.ceil(y_max_limit / TICK_INTERVAL) * TICK_INTERVAL

ax.plot([0, 0], [0, y_max_limit], 
        color=REFERENCE_LINE_COLOR, 
        linestyle=REFERENCE_LINE_STYLE, 
        linewidth=REFERENCE_LINE_WIDTH, 
        alpha=REFERENCE_LINE_ALPHA,
        zorder=4)  # zorder确保虚线在轨迹线上方

# ==================== 坐标轴设置 ====================
# 使用MultipleLocator强制设置刻度间隔
ax.xaxis.set_major_locator(MultipleLocator(TICK_INTERVAL))
ax.yaxis.set_major_locator(MultipleLocator(TICK_INTERVAL))
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

# 自适应坐标轴范围
# X轴范围基于数据，并向外扩展到最近的刻度
x_margin = max(abs(fx_min), abs(fx_max)) * 0.1 + 0.05
x_min_plot = fx_min - x_margin
x_max_plot = fx_max + x_margin

# 调整到刻度点
x_min_plot = np.floor(x_min_plot / TICK_INTERVAL) * TICK_INTERVAL
x_max_plot = np.ceil(x_max_plot / TICK_INTERVAL) * TICK_INTERVAL

# 确保0在范围内
if x_min_plot > 0:
    x_min_plot = -TICK_INTERVAL
if x_max_plot < 0:
    x_max_plot = TICK_INTERVAL

# Y轴（Z轴）范围基于数据，并向外扩展到最近的刻度
y_min_plot = min(0, fy_min) - 0.05
y_max_plot = max(fy_max, Fmaxradius) + 0.05

y_min_plot = np.floor(y_min_plot / TICK_INTERVAL) * TICK_INTERVAL
y_max_plot = np.ceil(y_max_plot / TICK_INTERVAL) * TICK_INTERVAL

ax.set_xlim([x_min_plot, x_max_plot])
ax.set_ylim([-0.1, y_max_plot])

print(f"\n最终坐标轴设置（间隔{TICK_INTERVAL}）:")
print(f"  X轴范围: [{x_min_plot:.2f}, {x_max_plot:.2f}]")
print(f"  Z轴范围: [{y_min_plot:.2f}, {y_max_plot:.2f}]")

# ==================== 图形样式 ====================
ax.grid(False)
ax.tick_params(labelsize=20)
ax.tick_params(axis='both', which='major', pad=TICK_PADDING)
# 设置坐标轴样式：只显示左边和下边
ax.spines['top'].set_visible(False)    # 隐藏上边框
ax.spines['right'].set_visible(False)  # 隐藏右边框

# 设置左边和下边坐标轴的线宽
ax.spines['left'].set_linewidth(AXIS_LINEWIDTH)
ax.spines['bottom'].set_linewidth(AXIS_LINEWIDTH)

# 设置坐标轴颜色
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

# 颜色条（可选，已注释）
# cbar = plt.colorbar(lc, ax=ax, shrink=0.8, pad=0.15)
# cbar.ax.tick_params(labelsize=10)

# ==================== 自动保存 ====================
csv_filename = os.path.splitext(os.path.basename(CSV_PATH))[0]  # 提取不含扩展名的文件名
SAVE_FILENAME = csv_filename + '.png'  # 或 '.svg', '.pdf' 等
SAVE_PATH = os.path.join(SAVE_DIRECTORY, SAVE_FILENAME)

# 自动保存图片
if SAVE_FIGURE:
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)  # 确保文件夹存在
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\n图片已保存至: {SAVE_PATH}")

plt.tight_layout()
plt.show()

# ==================== 输出总结 ====================
print("\n" + "=" * 60)
print("绘制总结:")
print(f"归一化方式: Fz最大值归一化 (归一化因子 = {fz_max:.2f})")
print(f"轨迹线使用全部 {len(xx)} 个数据点")
print(f"扇形计算使用 {len(xx_filtered)} 个数据点（过滤掉 {removed_count} 个接近0的点）")
print(f"扇形角度: {sector_angle:.2f}°")
print(f"扇形半径: {Fmaxradius:.3f}")
print(f"Fz归一化后最大值: {yy.max():.3f}（应为1.0）")