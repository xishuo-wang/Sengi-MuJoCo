"""
参数搜索结果查看脚本 - 加载搜索CSV文件并显示最佳参数组合
"""

import pandas as pd
import os


# ==================== 配置参数 ====================
CSV_DATA_FILE = r"D:\Code\Sengi-MuJoCo\Sengi-single_MuJoCo\Data_Explore\Explore_Data_260527_2125.csv"


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    if not os.path.exists(CSV_DATA_FILE):
        print(f"未找到数据文件 '{CSV_DATA_FILE}'")
        exit()

    # 加载数据
    df = pd.read_csv(CSV_DATA_FILE)

    print("=" * 60)
    print(f"数据文件: {CSV_DATA_FILE}")
    print(f"总参数组合数: {len(df)}")
    print("=" * 60)

    # 按速度排序
    df_sorted = df.sort_values('avg_velocity', ascending=False)

    print("\n按速度排序的前10名最佳参数组合:")
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(df_sorted.head(10).to_string(index=False))

    # 统计信息
    print("\n" + "=" * 60)
    print("统计信息:")
    print(f"速度 - 最大: {df['avg_velocity'].max():.4f} m/s")
    print(f"速度 - 最小: {df['avg_velocity'].min():.4f} m/s")
    print(f"速度 - 平均: {df['avg_velocity'].mean():.4f} m/s")
    print(f"高度 - 最大: {df['max_height'].max():.4f} m")
    print(f"高度 - 最小: {df['max_height'].min():.4f} m")
    print(f"俯仰角 - 最大: {df['max_pitch'].max():.1f}°")
    print(f"俯仰角 - 平均: {df['max_pitch'].mean():.1f}°")
    print("=" * 60)