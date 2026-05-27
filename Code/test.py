# exact_pybullet_terrain.py
import numpy as np
from PIL import Image

def generate_exact_pybullet_terrain():
    """生成与PyBullet脚本完全相同的随机地形"""
    # 完全相同的参数
    np.random.seed(307)
    length = 100
    width = 100
    height_min = -0.2
    height_max = 0.1
    
    # 生成随机高度场
    heightfield = np.random.uniform(height_min, height_max, 
                                   size=(length, width))
    
    # 不进行任何平滑（smooth_iterations=0）
    
    # 计算实际缩放
    # PyBullet中：meshScale=[terrain_scale_x, terrain_scale_y, height_scale]
    # terrain_scale_x = 10.0 / length = 0.1
    # terrain_scale_y = 10.0 / width = 0.1
    # height_scale = 0.1
    
    # 转换为0-255范围保存为PNG
    # 注意：需要正确处理负值
    height_normalized = (heightfield - height_min) / (height_max - height_min) * 255
    height_normalized = height_normalized.astype(np.uint8)
    
    # 保存
    img = Image.fromarray(height_normalized, mode='L')
    img.save('exact_pybullet_random.png')
    
    print(f"已生成完全匹配的地形")
    print(f"尺寸: {length}x{width}")
    print(f"实际高度范围: {heightfield.min():.3f} 到 {heightfield.max():.3f}")
    print(f"高度标准差: {heightfield.std():.3f}")
    
    return heightfield

if __name__ == "__main__":
    terrain = generate_exact_pybullet_terrain()