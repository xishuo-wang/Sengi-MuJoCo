import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# ==================== 仿真参数 ====================
n_spine = 3                     # 脊柱节数
freq_Hz = 3                     # 频率(Hz)
f = 2 * np.pi * freq_Hz         # 角频率(rad/s)
base_body_length = 0.11         # 基准体长(m)

# ==================== 数据处理参数 ====================
percent_threshold = 0.07        # 取前7%性能最好的参数组
smooth_method = 'lowess'        # 平滑方法
smooth_window = 8               # 平滑窗口大小
verbose = True                  # 是否打印详细信息

# ==================== 路径参数 ====================
RootPath = 'D:/Code/Sengi-Matlab'           # 根目录
figure_save_path = 'D:/Code/Sengi-Matlab/Figure'  # 图片保存路径
csv_folder = 'D:/Code/Sengi-MuJoCo'          # CSV文件所在文件夹

# ==================== 构建扭矩文件字典 ====================
# 方法1：自动扫描文件夹构建字典
torque_dict = {}

if os.path.exists(csv_folder):
    for filename in os.listdir(csv_folder):
        if re.match(r'torque_.*\.csv', filename):
            # 从文件名中提取扭矩值，例如 torque_0.4.csv -> 0.4
            match = re.search(r'torque_([0-9.]+)\.csv', filename)
            if match:
                torque_value = float(match.group(1))
                # 构建展示名称，例如 "tor_0.4"
                display_name = f"tor_{torque_value:.1f}"
                # 完整的文件路径
                full_path = os.path.join(csv_folder, filename)
                torque_dict[display_name] = full_path

# 对字典按扭矩值排序
torque_dict = dict(sorted(torque_dict.items(), 
                         key=lambda x: float(x[0].split('_')[1])))

# 如果自动扫描不符合需求，可以使用手动构建的字典：
torque_dict = {
    "tor_0.35": r"D:\Code\Sengi-MuJoCo\Double_Tor_0.35.csv",
}

# ==================== 图形参数 ====================
# 颜色方案（橙黄-紫罗兰-青绿）
color1 = np.array([0.996, 0.675, 0.369])   # 橙黄色
color2 = np.array([0.781, 0.475, 0.816])   # 紫罗兰色
color3 = np.array([0.294, 0.753, 0.784])   # 青绿色

# 图形尺寸设置
fig_width = 24                   # 图形宽度(cm)
fig_height = 16                  # 图形高度(cm)

# 线条设置
line_width = 2                   # 线宽
alpha_region = 0.15              # 区域透明度

# 坐标轴设置
font_size = 15                   # 字体大小
font_name = 'Arial'              # 字体类型
axis_line_width = 2              # 坐标轴线宽

# ==================== 生成渐变色 ====================
def generate_gradient_colors(color1, color2, color3, n_color=256):
    """生成渐变颜色"""
    colors_1_2 = np.array([
        np.linspace(color1[0], color2[0], n_color),
        np.linspace(color1[1], color2[1], n_color),
        np.linspace(color1[2], color2[2], n_color)
    ]).T
    
    colors_2_3 = np.array([
        np.linspace(color2[0], color3[0], n_color),
        np.linspace(color2[1], color3[1], n_color),
        np.linspace(color2[2], color3[2], n_color)
    ]).T
    
    return np.vstack([colors_1_2, colors_2_3])

# ==================== 数据处理函数 ====================
def process_data_simple(a_legH_hip, a_legH_knee, phase_lag, avg_velocity, 
                        body_length, percent_threshold, smooth_method, smooth_window, verbose):
    """
    简化的数据处理函数
    """
    # 第1步：按参数组合分组
    a_legB_R = a_legH_hip.values if hasattr(a_legH_hip, 'values') else np.array(a_legH_hip)
    a_legB_P = a_legH_knee.values if hasattr(a_legH_knee, 'values') else np.array(a_legH_knee)
    vx = avg_velocity.values if hasattr(avg_velocity, 'values') else np.array(avg_velocity)
    phase_lag_arr = phase_lag.values if hasattr(phase_lag, 'values') else np.array(phase_lag)
    phase_lag_arr = -phase_lag_arr
    # 找出唯一的参数组合
    combo_array = np.column_stack([a_legB_R, a_legB_P])
    unique_combo, group_idx = np.unique(combo_array, axis=0, return_inverse=True)
    n_groups = len(unique_combo)
    
    # 找出唯一的相位滞后值
    phases_sorted = np.sort(np.unique(phase_lag_arr))
    
    if verbose:
        print(f'  数据分组: 共{n_groups}个参数组合')
    
    # 提取每组的速度最大值
    group_max_vx = np.full(n_groups, np.nan)
    for g in range(n_groups):
        mask = group_idx == g
        group_vx = vx[mask]
        
        if len(group_vx) > 0:
            group_max_vx[g] = np.max(group_vx)
    
    # 选择前percent_threshold%性能最好的组
    valid_group_mask = ~np.isnan(group_max_vx)
    valid_max_vx = group_max_vx[valid_group_mask]
    valid_group_indices = np.where(valid_group_mask)[0]
    
    # 按速度降序排序
    sort_idx = np.argsort(valid_max_vx)[::-1]
    sorted_groups = valid_group_indices[sort_idx]
    
    # 计算要选取的组数
    num_selected_groups = max(1, round(percent_threshold * n_groups))
    selected_groups = sorted_groups[:min(num_selected_groups, len(sorted_groups))]
    
    if verbose:
        print(f'  性能筛选: 选中{len(selected_groups)}组(前{percent_threshold*100:.1f}%)')
    
    # 构建VxbyPhaselag矩阵
    Vx_matrix = np.full((len(phases_sorted), len(selected_groups)), np.nan)
    
    for g_idx, g in enumerate(selected_groups):
        mask = group_idx == g
        group_phases = phase_lag_arr[mask]
        group_vx = vx[mask]
        
        # 将Vx值按相位对齐
        for k in range(len(group_phases)):
            phase_idx = np.where(np.abs(phases_sorted - group_phases[k]) < 1e-6)[0]
            if len(phase_idx) > 0:
                Vx_matrix[phase_idx[0], g_idx] = group_vx[k]
    
    # 存储选中的参数组合信息
    selected_groups_info = []
    for g_idx, g in enumerate(selected_groups):
        if g < len(unique_combo):
            info = {
                'a_legH_hip': unique_combo[g, 0],
                'a_legH_knee': unique_combo[g, 1],
            }
            
            # 获取该组的phase_lag范围
            mask = group_idx == g
            group_phases = phase_lag_arr[mask]
            if len(group_phases) > 0:
                info['phase_min'] = np.min(group_phases)
                info['phase_max'] = np.max(group_phases)
            else:
                info['phase_min'] = np.nan
                info['phase_max'] = np.nan
            
            selected_groups_info.append(info)
    
    # 第2步：处理每个相位的数据
    y_max = np.nanmax(Vx_matrix, axis=1)
    y_min = np.nanmin(Vx_matrix, axis=1)
    y_mean = np.nanmean(Vx_matrix, axis=1)
    
    # 转换为体长/秒
    vx_max_BL = y_max / body_length
    vx_min_BL = y_min / body_length
    vx_mean_BL = y_mean / body_length
    
    # 第3步：数据平滑与插值
    valid_idx = ~np.isnan(y_max)
    
    if np.sum(valid_idx) > 3:
        y_min_smooth = np.full_like(y_min, np.nan)
        y_max_smooth = np.full_like(y_max, np.nan)
        y_mean_smooth = np.full_like(y_mean, np.nan)
        
        if smooth_method.lower() in ['rlowess', 'lowess']:
            window = min(smooth_window, np.sum(valid_idx) - 2)
            if window % 2 == 0:
                window += 1
            
            y_min_smooth[valid_idx] = savgol_filter(y_min[valid_idx], window, 2)
            y_max_smooth[valid_idx] = savgol_filter(y_max[valid_idx], window, 2)
            y_mean_smooth[valid_idx] = savgol_filter(y_mean[valid_idx], window, 2)
        else:
            y_min_smooth[valid_idx] = y_min[valid_idx]
            y_max_smooth[valid_idx] = y_max[valid_idx]
            y_mean_smooth[valid_idx] = y_mean[valid_idx]
        
        valid_phases = phases_sorted[valid_idx]
        valid_min = y_min_smooth[valid_idx]
        valid_max = y_max_smooth[valid_idx]
        valid_mean = y_mean_smooth[valid_idx]
        
        phases_fine = np.linspace(-np.pi, np.pi, 201)
        
        try:
            interp_min = interp1d(valid_phases, valid_min, kind='cubic', 
                                  bounds_error=False, fill_value=np.nan)
            interp_max = interp1d(valid_phases, valid_max, kind='cubic', 
                                  bounds_error=False, fill_value=np.nan)
            interp_mean = interp1d(valid_phases, valid_mean, kind='cubic', 
                                   bounds_error=False, fill_value=np.nan)
            
            y_min_q = interp_min(phases_fine)
            y_max_q = interp_max(phases_fine)
            y_mean_q = interp_mean(phases_fine)
        except:
            interp_min = interp1d(valid_phases, valid_min, kind='linear', 
                                  bounds_error=False, fill_value=np.nan)
            interp_max = interp1d(valid_phases, valid_max, kind='linear', 
                                  bounds_error=False, fill_value=np.nan)
            interp_mean = interp1d(valid_phases, valid_mean, kind='linear', 
                                   bounds_error=False, fill_value=np.nan)
            
            y_min_q = interp_min(phases_fine)
            y_max_q = interp_max(phases_fine)
            y_mean_q = interp_mean(phases_fine)
    else:
        phases_fine = np.linspace(-np.pi, np.pi, 201)
        y_min_q = np.full_like(phases_fine, np.nan)
        y_max_q = np.full_like(phases_fine, np.nan)
        y_mean_q = np.full_like(phases_fine, np.nan)
    
    vx_min_q = y_min_q / body_length
    vx_max_q = y_max_q / body_length
    vx_mean_q = y_mean_q / body_length
    
    result = {
        'phases_sorted': phases_sorted,
        'phases_fine': phases_fine,
        'vx_max_BL': vx_max_BL,
        'vx_min_BL': vx_min_BL,
        'vx_mean_BL': vx_mean_BL,
        'vx_max_q': vx_max_q,
        'vx_min_q': vx_min_q,
        'vx_mean_q': vx_mean_q,
        'global_max_vx': np.nanmax(vx_max_BL),
        'n_groups': n_groups,
        'n_selected_groups': len(selected_groups),
        'selected_groups': selected_groups,
        'selected_groups_info': selected_groups_info,
        'group_max_vx': group_max_vx
    }
    
    return result


# ==================== 主程序 ====================
def main():
    all_results = []
    valid_indices = []
    
    # 检查是否有文件
    if len(torque_dict) == 0:
        raise Exception(f'未找到任何torque_*.csv文件在文件夹: {csv_folder}')
    
    if verbose:
        print('\n=== 数据处理程序启动 ===')
        print(f'CSV文件夹: {csv_folder}')
        print(f'找到 {len(torque_dict)} 个扭矩文件:')
        for i, (display_name, csv_path) in enumerate(torque_dict.items()):
            torque_value = float(display_name.split('_')[1])
            print(f'  {i+1}. {display_name} (扭矩值: {torque_value:.1f}) -> {os.path.basename(csv_path)}')
        print(f'图片保存路径: {figure_save_path}')
    
    # ==================== 主处理循环 ====================
    for display_name, csv_path in torque_dict.items():
        # 从显示名称中提取扭矩值
        torque_value = float(display_name.split('_')[1])
        
        if verbose:
            print(f'\n========== 处理文件: {display_name} (扭矩值 = {torque_value:.1f}) ==========')
            print(f'文件路径: {csv_path}')
        
        # 根据扭矩值调整体长
        scale_factor = torque_value
        body_length = base_body_length * scale_factor
        
        if verbose:
            print(f'加载数据文件: {csv_path}')
        
        # 读取CSV文件
        try:
            data_table = pd.read_csv(csv_path)
        except Exception as e:
            print(f'错误：无法读取文件 {csv_path}: {e}')
            continue
        
        # 提取数据列
        a_legH_hip = data_table['a_legH_hip']
        a_legH_knee = data_table['a_legH_knee']
        phase_lag = data_table['phase_lag']
        avg_velocity = data_table['avg_velocity']
        
        if verbose:
            print(f'数据加载完成，共 {len(data_table)} 组数据')
        
        # 处理数据
        result = process_data_simple(a_legH_hip, a_legH_knee, phase_lag, avg_velocity,
                                    body_length, percent_threshold, smooth_method, 
                                    smooth_window, verbose)
        
        # 输出被选中的参数组合
        if verbose:
            print(f'\n被选中的参数组合（前{percent_threshold*100:.1f}%性能最好的组）:')
            print(f'{"组号":<8} {"a_legH_hip":<12} {"a_legH_knee":<12} {"phase_lag范围":<20}')
            print('-' * 56)
            
            for g, info in enumerate(result['selected_groups_info']):
                min_phase = info.get('phase_min', np.nan)
                max_phase = info.get('phase_max', np.nan)
                
                if not np.isnan(min_phase) and not np.isnan(max_phase):
                    print(f'组合 {g+1:<4d}: {info["a_legH_hip"]:<12.3f} '
                          f'{info["a_legH_knee"]:<12.3f} [{min_phase:.3f}, {max_phase:.3f}] rad')
            print('-' * 56)
        
        # 存储结果，使用display_name作为标识
        result_dict = {
            'display_name': display_name,  # 添加显示名称
            'torque_value': torque_value,
            'source': 'CSV',
            'csv_path': csv_path,  # 添加文件路径
            'phases_sorted': result['phases_sorted'],
            'phases_fine': result['phases_fine'],
            'vx_min': result['vx_min_BL'],
            'vx_max': result['vx_max_BL'],
            'vx_mean': result['vx_mean_BL'],
            'vx_min_q': result['vx_min_q'],
            'vx_max_q': result['vx_max_q'],
            'vx_mean_q': result['vx_mean_q'],
            'global_max_vx': result['global_max_vx'],
            'selected_groups_info': result['selected_groups_info']
        }
        
        all_results.append(result_dict)
        valid_indices.append(len(all_results) - 1)
        
        if verbose:
            print(f'\n  ✓ {display_name} 处理完成: '
                  f'最大速度={result["global_max_vx"]:.2f} BL/s, '
                  f'选中参数组合数={len(result["selected_groups"])}')
    
    # ==================== 分配渐变色 ====================
    # 按扭矩值排序
    torque_order = np.argsort([all_results[i]['torque_value'] for i in valid_indices])
    sorted_valid_indices = [valid_indices[i] for i in torque_order]
    
    # 生成渐变色
    Colors_sim = generate_gradient_colors(color1, color2, color3)
    
    # 为不同扭矩值分配颜色
    total_results = len(sorted_valid_indices)
    color_indices = np.linspace(0, len(Colors_sim) - 1, total_results, dtype=int)
    
    # 将颜色分配给对应的结果
    for i, idx in enumerate(sorted_valid_indices):
        all_results[idx]['color'] = Colors_sim[color_indices[i]]
    
    # ==================== 打印统计信息 ====================
    if verbose:
        print('\n========== 所有数据处理完成统计 ==========')
        print(f'{"显示名称":<12} {"扭矩值":<10} {"最大速度(BL/s)":<16} {"数据点数":<12} {"选中组合数":<12}')
        print('-' * 62)
        
        for idx in sorted_valid_indices:
            result = all_results[idx]
            print(f'{result["display_name"]:<12} '
                  f'{result["torque_value"]:<10.1f} '
                  f'{result["global_max_vx"]:<16.2f} '
                  f'{len(result["vx_max"]):<12d} '
                  f'{len(result["selected_groups_info"]):<12d}')
        print('-' * 62)
    
    # ==================== 绘制结果图像 ====================
    plt.rcParams['font.family'] = font_name
    plt.rcParams['font.size'] = font_size
    
    # 图1：原始数据曲线
    fig1, ax1 = plt.subplots(figsize=(fig_width/2.54, fig_height/2.54))
    
    for idx in sorted_valid_indices:
        result = all_results[idx]
        
        phases_rad = result['phases_sorted']
        vx_min = result['vx_min']
        vx_max = result['vx_max']
        vx_mean = result['vx_mean']
        
        valid_idx_arr = ~np.isnan(vx_max)
        
        if np.sum(valid_idx_arr) > 1:
            valid_phases = phases_rad[valid_idx_arr]
            valid_min = vx_min[valid_idx_arr]
            valid_max = vx_max[valid_idx_arr]
            valid_mean = vx_mean[valid_idx_arr]
            
            # 绘制速度区域
            ax1.fill_between(valid_phases, valid_min, valid_max, 
                           color=result['color'], alpha=alpha_region)
            
            # 绘制平均线，使用display_name作为标签
            ax1.plot(valid_phases, valid_mean, '-', 
                   color=result['color'], linewidth=line_width,
                   label=f'{result["display_name"]}')
    
    # 设置图1格式
    ax1.set_xlabel('Phase lag (rad)', fontsize=font_size)
    ax1.set_ylabel('Forward velocity (BL/s)', fontsize=font_size)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax1.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax1.set_xlim([-np.pi, np.pi])
    ax1.set_ylim([-1, 7])
    ax1.set_yticks(np.arange(-1, 8, 1))
    ax1.legend(loc='upper left', title='扭矩值', fontsize=10)
    ax1.tick_params(width=axis_line_width)
    
    plt.tight_layout()
    
    # 图2：平滑+插值后的数据曲线
    fig2, ax2 = plt.subplots(figsize=(fig_width/2.54, fig_height/2.54))
    
    for idx in sorted_valid_indices:
        result = all_results[idx]
        
        phases_fine = result['phases_fine']
        vx_min_q = result['vx_min_q']
        vx_max_q = result['vx_max_q']
        vx_mean_q = result['vx_mean_q']
        
        valid_fine = ~np.isnan(vx_max_q)
        
        if np.sum(valid_fine) > 1:
            valid_phases_fine = phases_fine[valid_fine]
            valid_min_q = vx_min_q[valid_fine]
            valid_max_q = vx_max_q[valid_fine]
            valid_mean_q = vx_mean_q[valid_fine]
            
            # 绘制速度区域
            ax2.fill_between(valid_phases_fine, valid_min_q, valid_max_q, 
                           color=result['color'], alpha=alpha_region)
            
            # 绘制平均线，使用display_name作为标签
            ax2.plot(valid_phases_fine, valid_mean_q, '-', 
                   color=result['color'], linewidth=line_width,
                   label=f'{result["display_name"]} (τ={result["torque_value"]:.1f})')
    
    # 设置图2格式
    ax2.set_xlabel('Phase lag (rad)', fontsize=font_size)
    ax2.set_ylabel('Forward velocity (BL/s)', fontsize=font_size)
    ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax2.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax2.set_xlim([-np.pi, np.pi])
    ax2.set_ylim([-2, 7])
    ax2.set_yticks(np.arange(-2, 8, 1))
    ax2.legend(loc='upper left', fontsize=10)
    ax2.tick_params(width=axis_line_width)
    ax2.set_box_aspect(1)
    
    plt.tight_layout()
    
    # ==================== 保存图片 ====================
    os.makedirs(figure_save_path, exist_ok=True)
    if verbose:
        print(f'\n创建/确认文件夹: {figure_save_path}')
    
    # 保存图2
    filename_base = f'Vx-Phase_lag-Torques_nSpine={n_spine}'
    png_file = os.path.join(figure_save_path, filename_base + '.png')
    fig2.savefig(png_file, dpi=300, bbox_inches='tight')
    
    if verbose:
        print(f'\n对比图已保存: {png_file}')
        print('\n处理完成！')
    
    plt.show()


if __name__ == '__main__':
    main()