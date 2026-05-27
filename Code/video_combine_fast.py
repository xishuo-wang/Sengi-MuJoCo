"""
视频合成程序 - 将文件夹中的所有视频直接合成为一个
优化版本：支持多线程处理和性能优化
"""

import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import time


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
        
        # 性能优化参数
        self.use_multithreading = True          # 是否使用多线程
        self.num_threads = 4       # 线程数（默认CPU核心数）
        self.frame_buffer_size = 30              # 帧缓冲区大小
        self.skip_frames_if_needed = False       # 是否在必要时跳帧
        self.target_fps = 30                      # 目标帧率
        
        print("=" * 60)
        print("视频合成程序 (优化版)")
        print(f"CPU核心数: {self.num_threads}")
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
        return self.video_files
    
    def create_mask_fast(self, fg_frame):
        """
        快速创建前景掩码（优化版本）
        """
        # 转换为灰度图
        gray = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2GRAY)
        
        # 创建掩码（白色背景为0，主体为1）- 使用更快的阈值方法
        _, mask = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # 简化的形态学处理（使用更小的核）
        if self.feather_radius > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 使用更快的模糊方法
            ksize = self.feather_radius * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        
        # 归一化到0-1
        mask = mask.astype(np.float32) / 255.0
        
        # 扩展为3通道
        return np.stack([mask, mask, mask], axis=2)
    
    def create_mask_super_fast(self, fg_frame):
        """
        超快速掩码创建（牺牲一点质量换取速度）
        """
        # 转换为灰度图
        gray = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2GRAY)
        
        # 快速阈值
        mask = (gray < self.white_threshold).astype(np.float32)
        
        # 如果不需要羽化，直接返回
        if self.feather_radius <= 0:
            return np.stack([mask, mask, mask], axis=2)
        
        # 使用简单的均值滤波代替高斯滤波（更快）
        ksize = self.feather_radius * 2 + 1
        mask = cv2.blur(mask, (ksize, ksize))
        
        return np.stack([mask, mask, mask], axis=2)
    
    def open_videos_fast(self):
        """快速打开视频文件（使用优化设置）"""
        print(f"\n========== 打开视频文件 ==========")
        
        # 使用第一个视频作为背景
        first_video = self.video_files[0]
        self.bg_cap = cv2.VideoCapture(first_video)
        
        if not self.bg_cap.isOpened():
            raise ValueError(f"无法打开背景视频: {first_video}")
        
        # 获取视频属性
        self.bg_fps = self.bg_cap.get(cv2.CAP_PROP_FPS)
        self.bg_width = int(self.bg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.bg_height = int(self.bg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.bg_total_frames = int(self.bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.bg_duration = self.bg_total_frames / self.bg_fps
        
        print(f"✓ 背景视频: {os.path.basename(first_video)}")
        print(f"  {self.bg_duration:.2f}秒, {self.bg_width}×{self.bg_height}")
        
        # 打开其他视频作为前景
        self.fg_caps = []
        self.fg_frame_counts = []
        
        for i, video_path in enumerate(self.video_files[1:], 1):
            cap = cv2.VideoCapture(video_path)
            
            if cap.isOpened():
                # 设置缓冲区大小以优化读取性能
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self.frame_buffer_size)
                
                self.fg_caps.append(cap)
                fg_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fg_frame_counts.append(fg_frames)
                
                print(f"✓ 前景视频 {i}: {os.path.basename(video_path)} ({fg_frames}帧)")
            else:
                print(f"⚠ 无法打开前景视频: {os.path.basename(video_path)}")
        
        print(f"\n成功加载 {len(self.fg_caps)} 个前景视频")
        
        # 总帧数以背景视频为准
        self.total_frames = self.bg_total_frames
        self.total_duration = self.bg_duration
        
        print(f"合成视频时长: {self.total_duration:.2f}秒")
        print(f"总帧数: {self.total_frames}帧")
    
    def process_frame_batch(self, frame_batch):
        """
        批量处理帧（用于多线程）
        """
        results = []
        for bg_frame, fg_frames_list in frame_batch:
            # 初始化合成帧
            combined = bg_frame.astype(np.float32)
            
            # 处理每个前景
            for fg_frame in fg_frames_list:
                if fg_frame is not None:
                    # 使用快速掩码创建
                    mask = self.create_mask_super_fast(fg_frame)
                    combined = combined * (1 - mask) + fg_frame.astype(np.float32) * mask
            
            # 转换为uint8
            results.append(np.clip(combined, 0, 255).astype(np.uint8))
        
        return results
    
    def synthesize_optimized(self):
        """优化的视频合成（使用多线程）"""
        print(f"\n========== 开始视频合成 (优化模式) ==========")
        print(f"多线程: {'开启' if self.use_multithreading else '关闭'}")
        print(f"线程数: {self.num_threads}")
        print(f"帧缓冲区: {self.frame_buffer_size}")
        
        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # 如果目标帧率与源视频不同，进行相应设置
        output_fps = self.target_fps if self.target_fps else self.bg_fps
        
        out = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            output_fps,
            (self.bg_width, self.bg_height)
        )
        
        if not out.isOpened():
            raise ValueError(f"无法创建输出视频: {self.output_video_path}")
        
        # 重置所有视频到第一帧
        self.bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for cap in self.fg_caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 计算帧率调整因子
        fps_ratio = self.bg_fps / output_fps if output_fps > 0 else 1.0
        
        # 进度条
        pbar = tqdm(total=self.total_frames, desc="视频合成", unit="帧")
        
        frame_count = 0
        frames_to_skip = max(1, int(fps_ratio)) if self.skip_frames_if_needed else 1
        
        # 如果使用多线程，准备批处理
        if self.use_multithreading and self.num_threads > 1:
            batch_size = self.num_threads * 2
            frame_batches = []
            current_batch = []
            
            while True:
                # 读取背景帧
                ret_bg, bg_frame = self.bg_cap.read()
                if not ret_bg:
                    break
                
                # 如果需要跳帧
                if frame_count % frames_to_skip != 0 and self.skip_frames_if_needed:
                    frame_count += 1
                    continue
                
                # 读取所有前景帧
                fg_frames = []
                for cap in self.fg_caps:
                    ret_fg, fg_frame = cap.read()
                    if not ret_fg:
                        # 循环播放
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_fg, fg_frame = cap.read()
                    fg_frames.append(fg_frame if ret_fg else None)
                
                current_batch.append((bg_frame, fg_frames))
                
                # 当批次达到大小时处理
                if len(current_batch) >= batch_size:
                    frame_batches.append(current_batch)
                    current_batch = []
                
                frame_count += 1
                
                # 进度更新（每10帧更新一次以减少开销）
                if frame_count % 10 == 0:
                    pbar.update(10)
            
            # 处理剩余的批次
            if current_batch:
                frame_batches.append(current_batch)
            
            pbar.close()
            
            # 使用线程池并行处理批次
            print(f"\n开始并行处理 {len(frame_batches)} 个批次...")
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # 提交所有批次任务
                futures = [executor.submit(self.process_frame_batch, batch) 
                          for batch in frame_batches]
                
                # 收集结果并写入
                write_pbar = tqdm(total=len(frame_batches), desc="写入视频", unit="批次")
                for future in futures:
                    batch_results = future.result()
                    for frame in batch_results:
                        out.write(frame)
                    write_pbar.update(1)
                write_pbar.close()
        
        else:
            # 单线程处理
            while True:
                # 读取背景帧
                ret_bg, bg_frame = self.bg_cap.read()
                if not ret_bg:
                    break
                
                # 如果需要跳帧
                if frame_count % frames_to_skip != 0 and self.skip_frames_if_needed:
                    frame_count += 1
                    continue
                
                # 初始化合成帧
                combined_frame = bg_frame.copy().astype(np.float32)
                
                # 处理每个前景视频
                for cap in self.fg_caps:
                    ret_fg, fg_frame = cap.read()
                    if not ret_fg:
                        # 循环播放
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_fg, fg_frame = cap.read()
                    
                    if ret_fg:
                        # 使用超快速掩码
                        mask = self.create_mask_super_fast(fg_frame)
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
        
        actual_frames = frame_count
        actual_duration = actual_frames / output_fps
        
        print(f"\n========== 视频合成完成 ==========")
        print(f"输出视频: {self.output_video_path}")
        print(f"总视频数: {len(self.video_files)}")
        print(f"处理帧数: {actual_frames}帧")
        print(f"视频时长: {actual_duration:.2f}秒")
        print(f"分辨率: {self.bg_width}×{self.bg_height}")
        print(f"输出帧率: {output_fps} fps")
        print("=" * 50)
    
    def synthesize_single_thread(self):
        """单线程合成（保留原始版本作为备选）"""
        print(f"\n========== 开始视频合成 (单线程模式) ==========")
        
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
            
            # 初始化合成帧
            combined_frame = bg_frame.copy().astype(np.float32)
            
            # 处理每个前景视频
            for cap in self.fg_caps:
                ret_fg, fg_frame = cap.read()
                if not ret_fg:
                    # 循环播放
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_fg, fg_frame = cap.read()
                
                if ret_fg:
                    # 使用超快速掩码
                    mask = self.create_mask_super_fast(fg_frame)
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
        print(f"处理帧数: {frame_count}帧")
        print(f"分辨率: {self.bg_width}×{self.bg_height}")
        print("=" * 50)
    
    def run(self, use_optimized=True):
        """
        运行完整的视频合成流程
        
        Args:
            use_optimized: 是否使用优化版本
        """
        print(f"视频文件夹: {self.video_folder}")
        
        start_time = time.time()
        
        # 执行各个步骤
        self.get_all_videos()
        self.open_videos_fast()
        
        if use_optimized:
            self.synthesize_optimized()
        else:
            self.synthesize_single_thread()
        
        elapsed_time = time.time() - start_time
        print(f"\n总耗时: {elapsed_time:.2f}秒")
        print(f"处理速度: {self.total_frames/elapsed_time:.1f} 帧/秒")
        print("\n程序执行完成！")


# ==================== 极速模式版本 ====================
class VideoSynthesizerUltraFast(VideoSynthesizer):
    """超快速视频合成器 - 牺牲更多质量换取极致速度"""
    
    def __init__(self, video_folder, output_video_name='combined_video_ultra.mp4'):
        super().__init__(video_folder, output_video_name)
        
        # 极致优化参数
        self.use_ultra_fast_mask = True        # 使用超快速掩码
        self.reduce_resolution_factor = 1.0     # 分辨率降低因子 (1.0=原分辨率, 0.5=半分辨率)
        self.use_fast_video_codec = True        # 使用更快的视频编码器
    
    def create_mask_ultra_fast(self, fg_frame):
        """
        极致快速掩码创建（大幅简化）
        """
        # 如果启用了分辨率降低，先缩小图像
        if self.reduce_resolution_factor < 1.0:
            new_width = int(fg_frame.shape[1] * self.reduce_resolution_factor)
            new_height = int(fg_frame.shape[0] * self.reduce_resolution_factor)
            fg_frame_small = cv2.resize(fg_frame, (new_width, new_height))
        else:
            fg_frame_small = fg_frame
        
        # 转换为灰度图
        gray = cv2.cvtColor(fg_frame_small, cv2.COLOR_BGR2GRAY)
        
        # 简单阈值
        mask = (gray < self.white_threshold).astype(np.float32)
        
        # 如果需要羽化，使用简单的均值滤波
        if self.feather_radius > 0:
            ksize = self.feather_radius * 2 + 1
            mask = cv2.blur(mask, (ksize, ksize))
        
        # 扩展为3通道
        mask_3d = np.stack([mask, mask, mask], axis=2)
        
        # 如果需要，恢复原始大小
        if self.reduce_resolution_factor < 1.0:
            mask_3d = cv2.resize(mask_3d, (fg_frame.shape[1], fg_frame.shape[0]))
        
        return mask_3d
    
    def process_frame_batch_ultra(self, frame_batch):
        """
        超快速批处理
        """
        results = []
        for bg_frame, fg_frames_list in frame_batch:
            # 如果需要降低背景分辨率
            if self.reduce_resolution_factor < 1.0:
                bg_small = cv2.resize(bg_frame, 
                                     (int(bg_frame.shape[1] * self.reduce_resolution_factor),
                                      int(bg_frame.shape[0] * self.reduce_resolution_factor)))
                combined = bg_small.astype(np.float32)
            else:
                combined = bg_frame.astype(np.float32)
            
            # 处理每个前景
            for fg_frame in fg_frames_list:
                if fg_frame is not None:
                    mask = self.create_mask_ultra_fast(fg_frame)
                    
                    # 如果降低了分辨率，需要调整掩码大小
                    if self.reduce_resolution_factor < 1.0:
                        fg_small = cv2.resize(fg_frame, 
                                             (combined.shape[1], combined.shape[0]))
                        combined = combined * (1 - mask) + fg_small.astype(np.float32) * mask
                    else:
                        combined = combined * (1 - mask) + fg_frame.astype(np.float32) * mask
            
            # 转换并恢复原始大小
            result = np.clip(combined, 0, 255).astype(np.uint8)
            if self.reduce_resolution_factor < 1.0:
                result = cv2.resize(result, (bg_frame.shape[1], bg_frame.shape[0]))
            
            results.append(result)
        
        return results
    
    def synthesize_ultra_fast(self):
        """极致快速合成"""
        print(f"\n========== 开始视频合成 (极致速度模式) ==========")
        print(f"分辨率降低因子: {self.reduce_resolution_factor}")
        
        # 选择更快的编码器
        if self.use_fast_video_codec:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID通常比MP4V快
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            self.frame_rate,
            (self.bg_width, self.bg_height)
        )
        
        if not out.isOpened():
            raise ValueError(f"无法创建输出视频: {self.output_video_path}")
        
        # 重置视频
        self.bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for cap in self.fg_caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 使用更大的批次
        batch_size = self.num_threads * 4
        frame_batches = []
        current_batch = []
        
        frame_count = 0
        pbar = tqdm(total=self.total_frames, desc="读取帧", unit="帧")
        
        while True:
            ret_bg, bg_frame = self.bg_cap.read()
            if not ret_bg:
                break
            
            fg_frames = []
            for cap in self.fg_caps:
                ret_fg, fg_frame = cap.read()
                if not ret_fg:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_fg, fg_frame = cap.read()
                fg_frames.append(fg_frame if ret_fg else None)
            
            current_batch.append((bg_frame, fg_frames))
            
            if len(current_batch) >= batch_size:
                frame_batches.append(current_batch)
                current_batch = []
            
            frame_count += 1
            pbar.update(1)
        
        if current_batch:
            frame_batches.append(current_batch)
        
        pbar.close()
        
        print(f"开始并行处理 {len(frame_batches)} 个批次...")
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.process_frame_batch_ultra, batch) 
                      for batch in frame_batches]
            
            write_pbar = tqdm(total=len(frame_batches), desc="写入视频", unit="批次")
            for future in futures:
                batch_results = future.result()
                for frame in batch_results:
                    out.write(frame)
                write_pbar.update(1)
            write_pbar.close()
        
        out.release()
        self.bg_cap.release()
        for cap in self.fg_caps:
            cap.release()


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 直接指定视频文件夹路径
    video_folder = r"D:\Code\Sengi-MuJoCo\Video-Matlab\BatchScan\torque_0.305"
    
    print("\n" + "=" * 60)
    print("选择合成模式:")
    print("1. 标准模式（平衡质量和速度）")
    print("2. 优化模式（推荐，多线程加速）")
    print("3. 极致速度模式（最快，可能降低质量）")
    print("=" * 60)
    
    # 默认使用优化模式
    use_mode = 2
    
    if use_mode == 1:
        # 标准模式
        synthesizer = VideoSynthesizer(
            video_folder=video_folder,
            output_video_name='combined_video_standard.mp4'
        )
        synthesizer.set_parameters(
            frame_rate=30,
            white_threshold=200,
            feather_radius=1,  # 减小羽化半径提高速度
            use_multithreading=False
        )
        synthesizer.run(use_optimized=False)
    
    elif use_mode == 2:
        # 优化模式（推荐）
        synthesizer = VideoSynthesizer(
            video_folder=video_folder,
            output_video_name='combined_video_optimized.mp4'
        )
        synthesizer.set_parameters(
            frame_rate=30,
            white_threshold=200,
            feather_radius=1,
            use_multithreading=True,
            num_threads=mp.cpu_count(),
            frame_buffer_size=50,
            skip_frames_if_needed=False
        )
        synthesizer.run(use_optimized=True)
    
    elif use_mode == 3:
        # 极致速度模式
        synthesizer = VideoSynthesizerUltraFast(
            video_folder=video_folder,
            output_video_name='combined_video_ultrafast.mp4'
        )
        synthesizer.set_parameters(
            frame_rate=30,
            white_threshold=200,
            feather_radius=0,  # 完全禁用羽化
            use_multithreading=True,
            num_threads=mp.cpu_count(),
            frame_buffer_size=100,
            reduce_resolution_factor=0.5,  # 半分辨率处理
            use_fast_video_codec=True
        )
        synthesizer.synthesize_ultra_fast()