"""
视频背景处理程序 - 将视频中的发黄背景替换为纯白色
直接处理已合成的视频，保持主体颜色不变
"""

import cv2
import numpy as np
import os
from tqdm import tqdm


class VideoBackgroundProcessor:
    def __init__(self, input_video_path, output_video_path=None):
        """
        初始化视频背景处理器
        
        Args:
            input_video_path: 输入视频文件路径
            output_video_path: 输出视频文件路径（如果不指定，自动生成）
        """
        self.input_video_path = input_video_path
        
        if output_video_path is None:
            # 自动生成输出文件名
            dir_name = os.path.dirname(input_video_path)
            base_name = os.path.basename(input_video_path)
            name, ext = os.path.splitext(base_name)
            self.output_video_path = os.path.join(dir_name, f"{name}_white_bg{ext}")
        else:
            self.output_video_path = output_video_path
        
        # 默认参数
        self.white_threshold = 200  # 白色阈值
        self.feather_radius = 3     # 羽化半径
        self.smooth_edges = True    # 是否平滑边缘
        self.background_color = [255, 255, 255]  # 目标背景颜色（纯白）
        
        print("=" * 60)
        print("视频背景处理程序初始化")
        print("=" * 60)
        print(f"输入视频: {self.input_video_path}")
        print(f"输出视频: {self.output_video_path}")
    
    def set_parameters(self, **kwargs):
        """设置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"参数 {key} 设置为: {value}")
    
    def create_background_mask(self, frame, white_threshold=200, feather_radius=3):
        """
        创建背景掩码
        
        Args:
            frame: 输入帧
            white_threshold: 白色阈值
            feather_radius: 羽化半径
        
        Returns:
            bg_mask: 背景掩码（背景为1，主体为0）
            fg_mask: 前景掩码（主体为1，背景为0）
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 方法1：基于灰度阈值
        mask_gray = (gray > white_threshold).astype(np.uint8) * 255
        
        # 方法2：基于RGB三通道（更精确）
        # 判断为背景的条件：三个通道都接近白色
        is_background = (
            (frame[:, :, 0] > white_threshold) &  # B通道
            (frame[:, :, 1] > white_threshold) &  # G通道
            (frame[:, :, 2] > white_threshold)     # R通道
        )
        
        # 也可以使用HSV色彩空间（可能更准确）
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 白色在HSV中：H任意，S较低，V较高
        # is_background = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 200)
        
        mask_rgb = is_background.astype(np.uint8) * 255
        
        # 结合两种方法（可根据需要选择）
        mask = mask_rgb  # 使用RGB方法
        
        # 形态学处理，优化边缘
        kernel = np.ones((3, 3), np.uint8)
        
        # 先腐蚀再膨胀，去除噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 边缘平滑
        if self.smooth_edges and feather_radius > 0:
            # 对掩码进行高斯模糊，实现边缘羽化
            ksize = feather_radius * 2 + 1
            mask_float = mask.astype(np.float32) / 255.0
            mask_float = cv2.GaussianBlur(mask_float, (ksize, ksize), 0)
            mask = (mask_float * 255).astype(np.uint8)
        
        # 创建前景掩码（主体区域）
        fg_mask = 255 - mask
        
        return mask, fg_mask
    
    def process_frame(self, frame):
        """
        处理单帧图像，将背景替换为纯白色
        
        Args:
            frame: 输入帧 (BGR格式)
        
        Returns:
            processed_frame: 处理后的帧
        """
        height, width = frame.shape[:2]
        
        # 创建背景掩码
        bg_mask, fg_mask = self.create_background_mask(
            frame, 
            self.white_threshold, 
            self.feather_radius
        )
        
        # 创建纯白背景
        white_bg = np.ones_like(frame) * 255
        
        # 转换为浮点数进行混合
        frame_float = frame.astype(np.float32)
        white_bg_float = white_bg.astype(np.float32)
        
        # 将掩码扩展到3通道
        if len(bg_mask.shape) == 2:
            bg_mask_3d = np.stack([bg_mask, bg_mask, bg_mask], axis=2)
            fg_mask_3d = np.stack([fg_mask, fg_mask, fg_mask], axis=2)
        else:
            bg_mask_3d = bg_mask
            fg_mask_3d = fg_mask
        
        # 归一化到0-1
        bg_mask_norm = bg_mask_3d.astype(np.float32) / 255.0
        fg_mask_norm = fg_mask_3d.astype(np.float32) / 255.0
        
        if self.smooth_edges:
            # 使用带羽化的混合
            # 背景区域用白色，前景区域用原图
            processed = white_bg_float * bg_mask_norm + frame_float * fg_mask_norm
        else:
            # 简单替换
            processed = np.where(bg_mask_3d > 128, white_bg_float, frame_float)
        
        # 确保值在有效范围内
        processed = np.clip(processed, 0, 255).astype(np.uint8)
        
        return processed
    
    def process_video(self):
        """处理整个视频"""
        print(f"\n========== 开始处理视频 ==========")
        
        # 打开输入视频
        cap = cv2.VideoCapture(self.input_video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {self.input_video_path}")
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"视频信息:")
        print(f"  时长: {duration:.2f}秒")
        print(f"  分辨率: {width}×{height}")
        print(f"  帧率: {fps:.2f}fps")
        print(f"  总帧数: {total_frames}")
        
        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            fps,
            (width, height)
        )
        
        if not out.isOpened():
            raise ValueError(f"无法创建输出视频: {self.output_video_path}")
        
        # 进度条
        pbar = tqdm(total=total_frames, desc="处理视频", unit="帧")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理当前帧
            processed_frame = self.process_frame(frame)
            
            # 写入输出
            out.write(processed_frame)
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        
        # 释放资源
        cap.release()
        out.release()
        
        print(f"\n========== 视频处理完成 ==========")
        print(f"输入视频: {self.input_video_path}")
        print(f"输出视频: {self.output_video_path}")
        print(f"处理帧数: {frame_count}")
        print(f"参数设置:")
        print(f"  白色阈值: {self.white_threshold}")
        print(f"  羽化半径: {self.feather_radius}")
        print(f"  边缘平滑: {self.smooth_edges}")
        print(f"  背景颜色: 纯白 (RGB: {self.background_color[2]},{self.background_color[1]},{self.background_color[0]})")
        print("=" * 50)
    
    def preview_frame(self, frame_number=0):
        """
        预览单帧处理效果
        
        Args:
            frame_number: 要预览的帧编号
        """
        cap = cv2.VideoCapture(self.input_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"无法读取第 {frame_number} 帧")
            return
        
        processed = self.process_frame(frame)
        
        # 并排显示原图和处理后的图
        comparison = np.hstack([frame, processed])
        
        cv2.imshow('Original (Left) vs Processed (Right) - Press any key to continue', comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def run(self, preview=False, preview_frame=0):
        """
        运行处理
        
        Args:
            preview: 是否先预览效果
            preview_frame: 预览的帧编号
        """
        if preview:
            print("\n预览处理效果...")
            self.preview_frame(preview_frame)
            
            response = input("\n是否继续处理整个视频? (y/n): ")
            if response.lower() != 'y':
                print("处理已取消")
                return
        
        self.process_video()


if __name__ == "__main__":
    # 设置输入视频路径
    input_video = r"D:\Code\Sengi-MuJoCo\Video-Matlab\BatchScan\torque_0.36_1.mp4"
    
    # 创建处理器
    processor = VideoBackgroundProcessor(
        input_video_path=input_video,
        output_video_path=None  # 自动生成输出文件名
    )
    
    # 设置参数
    processor.set_parameters(
        white_threshold=200,      # 白色阈值（调整这个值来更好地识别背景）
        feather_radius=3,         # 羽化半径（越大边缘越柔和）
        smooth_edges=True,        # 是否平滑边缘
        background_color=[255, 255, 255]  # 纯白背景
    )
    
    # 运行处理
    # 先预览一帧（第100帧），确认效果后再处理整个视频
    processor.run(preview=True, preview_frame=100)