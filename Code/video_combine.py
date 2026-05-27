"""
视频合成程序 - 将文件夹中的所有视频直接合成为一个
不解析文件名，不进行排序和颜色处理，直接合并所有视频
"""

import cv2
import numpy as np
import os
import glob
from tqdm import tqdm


class VideoSynthesizer:
    def __init__(self, video_folder, output_video_name='combined_video.mp4'):
        """
        初始化视频合成器
        
        Args:
            video_folder: 包含视频文件的文件夹路径
            output_video_name: 输出视频文件名
        """
        self.video_folder = video_folder
        self.output_video_name = output_video_name
        self.output_video_path = os.path.join(video_folder, output_video_name)
        
        # 默认参数
        self.frame_rate = 30
        self.white_threshold = 200
        self.feather_radius = 2
        
        print("=" * 60)
        print("视频合成程序初始化")
        print("=" * 60)
    
    def set_parameters(self, **kwargs):
        """设置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"参数 {key} 设置为: {value}")
    
    def get_all_videos(self):
        """获取文件夹中所有MP4文件"""
        pattern = os.path.join(self.video_folder, "*.mp4")
        all_videos = glob.glob(pattern)
        
        # 排除已生成的合成视频
        self.video_files = [f for f in all_videos 
                           if os.path.basename(f) != self.output_video_name]
        
        print(f"\n找到 {len(self.video_files)} 个视频文件")
        for i, video in enumerate(self.video_files):
            print(f"  {i+1}. {os.path.basename(video)}")
        
        return self.video_files
    
    def create_mask(self, fg_frame, white_threshold=200, feather_radius=2):
        # 转换为灰度图
        gray = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2GRAY)
        
        # 创建掩码（白色背景为0，主体为1）
        mask = (gray < white_threshold).astype(np.uint8) * 255
        
        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)  # 腐蚀锐化边缘
        mask = cv2.dilate(mask, kernel, iterations=1)  # 膨胀恢复主体
        
        # 高斯模糊实现羽化
        if feather_radius > 0:
            ksize = feather_radius * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        
        # 归一化到0-1
        mask = mask.astype(np.float32) / 255.0
        
        # 扩展为3通道
        mask_3d = np.stack([mask, mask, mask], axis=2)
        
        return mask_3d
    
    def open_videos(self):
        """打开所有视频文件"""
        print(f"\n========== 打开视频文件 ==========")
        
        # 使用第一个视频作为背景
        first_video = self.video_files[0]
        self.bg_cap = cv2.VideoCapture(first_video)
        
        if not self.bg_cap.isOpened():
            raise ValueError(f"无法打开背景视频: {first_video}")
        
        self.bg_fps = self.bg_cap.get(cv2.CAP_PROP_FPS)
        self.bg_width = int(self.bg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.bg_height = int(self.bg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.bg_total_frames = int(self.bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.bg_duration = self.bg_total_frames / self.bg_fps
        
        print(f"✓ 背景视频: {os.path.basename(first_video)}")
        print(f"  {self.bg_duration:.2f}秒, {self.bg_width}×{self.bg_height}, {self.bg_fps:.1f}fps")
        
        # 打开其他视频作为前景
        self.fg_caps = []
        
        for i, video_path in enumerate(self.video_files[1:], 1):
            cap = cv2.VideoCapture(video_path)
            
            if cap.isOpened():
                self.fg_caps.append(cap)
                
                fg_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                fg_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fg_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fg_duration = fg_frames / cap.get(cv2.CAP_PROP_FPS)
                
                print(f"✓ 前景视频 {i}: {os.path.basename(video_path)}")
                print(f"  {fg_duration:.2f}秒, {fg_width}×{fg_height}")
                
                if fg_width != self.bg_width or fg_height != self.bg_height:
                    print(f"  ⚠ 分辨率不匹配!")
            else:
                print(f"⚠ 无法打开前景视频: {os.path.basename(video_path)}")
        
        print(f"\n成功加载 {len(self.fg_caps)} 个前景视频")
        
        # 总帧数以背景视频为准
        self.total_frames = self.bg_total_frames
        self.total_duration = self.bg_duration
        
        print(f"合成视频时长: {self.total_duration:.2f}秒")
        print(f"总帧数: {self.total_frames}帧")
    
    def synthesize(self):
        """执行视频合成"""
        print(f"\n========== 开始视频合成 ==========")
        
        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            self.frame_rate,
            (self.bg_width, self.bg_height)
        )
        
        if not out.isOpened():
            raise ValueError(f"无法创建输出视频: {self.output_video_path}")
        
        # 重置所有视频到第一帧
        self.bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for cap in self.fg_caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 进度条
        pbar = tqdm(total=self.total_frames, desc="视频合成", unit="帧")
        
        frame_count = 0
        
        while True:
            # 读取背景帧
            ret_bg, bg_frame = self.bg_cap.read()
            if not ret_bg:
                break
            
            # 初始化合成帧为背景帧
            combined_frame = bg_frame.copy().astype(np.float32)
            
            # 处理每个前景视频
            for cap in self.fg_caps:
                # 读取前景帧
                ret_fg, fg_frame = cap.read()
                if not ret_fg:
                    # 如果前景视频结束，重置到第一帧继续循环
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_fg, fg_frame = cap.read()
                
                if ret_fg:
                    # 创建掩码并融合
                    mask = self.create_mask(
                        fg_frame,
                        self.white_threshold,
                        self.feather_radius
                    )
                    
                    combined_frame = combined_frame * (1 - mask) + fg_frame.astype(np.float32) * mask
            
            # 转换为uint8并写入
            combined_frame = np.clip(combined_frame, 0, 255).astype(np.uint8)
            out.write(combined_frame)
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        
        # 释放资源
        out.release()
        self.bg_cap.release()
        for cap in self.fg_caps:
            cap.release()
        
        print(f"\n========== 视频合成完成 ==========")
        print(f"输出视频: {self.output_video_path}")
        print(f"总视频数: {len(self.video_files)}")
        print(f"视频时长: {self.total_duration:.2f}秒")
        print(f"总帧数: {frame_count}帧")
        print(f"分辨率: {self.bg_width}×{self.bg_height}")
        print(f"帧率: {self.frame_rate} fps")
        print("=" * 50)
    
    def run(self):
        print(f"视频文件夹: {self.video_folder}")
        
        # 执行各个步骤
        self.get_all_videos()
        self.open_videos()
        self.synthesize()
        
        print("\n程序执行完成！")


if __name__ == "__main__":
    video_folder = r"D:\Code\Sengi-MuJoCo\Video-Matlab\BatchScan\torque_0.4"
    synthesizer = VideoSynthesizer(video_folder=video_folder, output_video_name='combined_video.mp4')
    
    synthesizer.set_parameters(
        frame_rate=30,           # 输出视频帧率
        white_threshold=200,      # 白色背景阈值
        feather_radius=2          # 羽化半径
    )
    
    # 运行合成
    synthesizer.run()