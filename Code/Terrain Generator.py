# terrain_generator.py
import numpy as np
from PIL import Image
import os
import cv2

def smooth_heightfield(heightfield, iterations=1):
    """平滑高度场"""
    if iterations == 0:
        return heightfield
    
    result = heightfield.copy()
    
    for _ in range(iterations):
        padded = np.pad(result, ((1, 1), (1, 1)), mode='edge')
        
        for i in range(1, padded.shape[0]-1):
            for j in range(1, padded.shape[1]-1):
                result[i-1, j-1] = np.mean(padded[i-1:i+2, j-1:j+2])
    
    return result

def generate_pybullet_style_terrain(length=100, width=100, seed=307, 
                                   height_range=(-0.2, 0.1), smooth_iterations=0):
    """生成与PyBullet代码相同的地形"""
    np.random.seed(seed)
    
    # 生成随机高度场
    height_min, height_max = height_range
    heightfield = np.random.uniform(height_min, height_max, size=(length, width))
    
    # 应用平滑
    heightfield = smooth_heightfield(heightfield, iterations=smooth_iterations)
    
    return heightfield

def save_as_png(heightfield, output_path, contrast_enhance=True):
    """保存高度场为PNG文件"""
    # 归一化到0-255
    normalized = (heightfield - heightfield.min()) / (heightfield.max() - heightfield.min()) * 255
    
    # 可选：增强对比度
    if contrast_enhance:
        normalized = np.clip(normalized * 1.5 - 25, 0, 255)
    
    # 转换为uint8
    img_data = normalized.astype(np.uint8)
    
    # 使用PIL保存
    img = Image.fromarray(img_data, mode='L')
    img.save(output_path)
    
    print(f"地形已保存到: {output_path}")
    print(f"尺寸: {heightfield.shape}, 高度范围: [{heightfield.min():.3f}, {heightfield.max():.3f}]")
    
    return output_path

def generate_terrain_variations(base_dir):
    """生成多种地形变体"""
    terrains = {}
    
    # 确保目录存在
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. 平坦地形（用于对比）
    flat = np.zeros((100, 100)) * 0.01
    terrains['flat'] = save_as_png(flat, os.path.join(base_dir, 'flat.png'))
    
    # 2. 轻微起伏地形
    terrain1 = generate_pybullet_style_terrain(
        length=100, width=100, seed=307,
        height_range=(-0.1, 0.1), smooth_iterations=1
    )
    terrains['gentle'] = save_as_png(terrain1, os.path.join(base_dir, 'gentle_hills.png'))
    
    # 3. 中等起伏地形（匹配原始参数）
    terrain2 = generate_pybullet_style_terrain(
        length=100, width=100, seed=307,
        height_range=(-0.2, 0.1), smooth_iterations=0
    )
    terrains['medium'] = save_as_png(terrain2, os.path.join(base_dir, 'medium_terrain.png'))
    
    # 4. 陡峭地形
    terrain3 = generate_pybullet_style_terrain(
        length=100, width=100, seed=42,
        height_range=(-0.3, 0.2), smooth_iterations=2
    )
    terrains['steep'] = save_as_png(terrain3, os.path.join(base_dir, 'steep_mountains.png'))
    
    # 5. 波浪地形
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = np.linspace(-2*np.pi, 2*np.pi, 100)
    X, Y = np.meshgrid(x, y)
    wave = np.sin(X) * np.cos(Y) * 0.15
    terrains['wave'] = save_as_png(wave, os.path.join(base_dir, 'wave_pattern.png'))
    
    return terrains

def visualize_terrain(heightfield, title="地形预览"):
    """可视化地形"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 4))
    
    # 2D视图
    ax1 = fig.add_subplot(131)
    im = ax1.imshow(heightfield, cmap='terrain')
    plt.colorbar(im, ax=ax1, label='高度')
    ax1.set_title(f'{title} - 2D视图')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # 3D视图
    ax2 = fig.add_subplot(132, projection='3d')
    x, y = np.meshgrid(range(heightfield.shape[1]), range(heightfield.shape[0]))
    surf = ax2.plot_surface(x, y, heightfield, cmap='terrain', 
                           linewidth=0, antialiased=False, alpha=0.8)
    ax2.set_title(f'{title} - 3D视图')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('高度')
    
    # 高度分布直方图
    ax3 = fig.add_subplot(133)
    ax3.hist(heightfield.flatten(), bins=30, edgecolor='black', alpha=0.7)
    ax3.set_title('高度分布')
    ax3.set_xlabel('高度值')
    ax3.set_ylabel('频率')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 生成地形
    terrain_dir = r"D:\Code\Model\Sengi\terrain"
    
    # 生成所有地形变体
    print("正在生成地形...")
    terrains = generate_terrain_variations(terrain_dir)
    
    # 生成主地形（与PyBullet完全相同）
    main_terrain = generate_pybullet_style_terrain(
        length=100, width=100, seed=307,
        height_range=(-0.2, 0.1), smooth_iterations=0
    )
    
    main_path = os.path.join(terrain_dir, "pybullet_style.png")
    save_as_png(main_terrain, main_path)
    
    # 可视化主地形
    print("\n主地形统计信息:")
    print(f"最小高度: {main_terrain.min():.4f}")
    print(f"最大高度: {main_terrain.max():.4f}")
    print(f"平均高度: {main_terrain.mean():.4f}")
    print(f"高度标准差: {main_terrain.std():.4f}")
    
    # 显示可视化（可选）
    visualize_terrain(main_terrain, "PyBullet风格地形")
    
    print(f"\n所有地形已生成到: {terrain_dir}")