import os
import time
from datetime import datetime
import torch
import numpy as np
from PIL import Image
from pj_root import PROJECT_ROOT

class SegmentationLogger:
    """
    为语义分割模型训练设计的日志记录器。

    该类负责创建日志目录、记录训练指标、保存采样图像等。
    """

    def __init__(self, title: str):
        """
        初始化日志记录器。

        Args:
            title (str): 日志的标题，用于创建项目目录。
        """
        # 定义根目录路径
        self.base_path = os.path.join(PROJECT_ROOT, "resource", "Logger")
        os.makedirs(self.base_path, exist_ok=True)

        # 创建时间戳目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.base_path, f"{title}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # 保存标题和当前阶段信息
        self.title = title
        self.current_phase = -1
        self.phase_log_file = None
        self.metrics_log_file = None
        self.phase_sample_dir = None

    def _init_phase(self, phase_num: int):
        """初始化一个新的训练阶段，创建相应的日志文件和采样目录。"""
        if self.current_phase >= 0:
            # 如果不是第一个阶段，关闭之前的日志文件
            if self.phase_log_file:
                self.phase_log_file.close()
            if self.metrics_log_file:
                self.metrics_log_file.close()

        self.current_phase = phase_num

        # 创建新阶段的日志文件
        log_filename = os.path.join(self.run_dir, f"phase_{self.current_phase}_log.txt")
        metrics_filename = os.path.join(self.run_dir, f"phase_{self.current_phase}_metrics.csv")

        self.phase_log_file = open(log_filename, 'w', encoding='utf-8')
        self.metrics_log_file = open(metrics_filename, 'w', encoding='utf-8')

        # 写入CSV头部
        self.metrics_log_file.write("epoch,train_loss,val_loss,train_time,val_time,learning_rate,miou\n")

        # 创建新阶段的采样目录
        self.phase_sample_dir = os.path.join(self.run_dir, f"phase_{self.current_phase}_samples")
        os.makedirs(self.phase_sample_dir, exist_ok=True)

        print(
            f"[Logger] Initialized Phase {self.current_phase}. Log file: {log_filename}, Sample dir: {self.phase_sample_dir}")

    def log_initial_info(self, model_name: str, dataset_name: str, train_size: int, val_size: int,
                         optimizer_method: str):
        """
        记录训练开始前的初始信息。

        Args:
            model_name (str): 模型名称。
            dataset_name (str): 数据集名称。
            train_size (int): 训练集大小。
            val_size (int): 验证/测试集大小。
            optimizer_method (str): 优化方法。
        """
        self._init_phase(0)  # 开始第一个阶段

        initial_info = (
            f"Training initialized at: {datetime.now()}\n"
            f"Model: {model_name}\n"
            f"Dataset: {dataset_name}\n"
            f"Train Set Size: {train_size}\n"
            f"Val Set Size: {val_size}\n"
            f"Optimizer: {optimizer_method}\n"
            f"Log Directory: {self.run_dir}\n"
            f"{'-' * 50}\n"
        )
        self.phase_log_file.write(initial_info)
        self.phase_log_file.flush()
        print(f"[Logger] Initial info for Phase {self.current_phase} logged.")

    def log_epoch_metrics(self, epoch: int, train_loss: float, val_loss: float,
                          train_time: float, val_time: float, learning_rate: float, miou: float):
        """
        记录一个epoch结束后的各项指标。

        Args:
            epoch (int): 当前轮数编号。
            train_loss (float): 训练集Loss。
            val_loss (float): 测试集Loss。
            train_time (float): 训练集花费时间。
            val_time (float): 测试集花费时间。
            learning_rate (float): 当前学习率。
            miou (float): 平均交并比(mIoU)。
        """
        if self.current_phase < 0:
            raise RuntimeError("Logging metrics before initialization. Call log_initial_info first.")

        metric_line = f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_time:.4f},{val_time:.4f},{learning_rate},{miou:.6f}\n"
        self.metrics_log_file.write(metric_line)
        self.metrics_log_file.flush()

    def finalize_current_phase_and_start_new(self):
        """结束当前阶段，并为下一阶段做准备。"""
        print(f"[Logger] Finalizing Phase {self.current_phase}...")
        if self.phase_log_file:
            self.phase_log_file.write(f"\nPhase {self.current_phase} completed at: {datetime.now()}\n")
            self.phase_log_file.flush()

        new_phase_num = self.current_phase + 1
        self._init_phase(new_phase_num)

    def save_samples(self, images, targets, predictions, sample_count: int, class_colors: list):
        """
        保存采样的原始图像、目标标签和预测结果。

        Args:
            images (torch.Tensor): 原始输入图像张量, shape: (B, C, H, W)，值域 [0, 1]。
            targets (torch.Tensor): 真实标签, shape: (B, H, W)，值为类别ID。
            predictions (torch.Tensor): 模型预测, shape: (B, H, W)，值为类别ID。
            sample_count (int): 要保存的样本数量。
            class_colors (list): 一个包含 (R, G, B) 元组的列表，代表每个类别的颜色。
        """
        if not self.phase_sample_dir:
            raise RuntimeError(
                "Cannot save samples. No active phase directory. Call log_initial_info or finalize_current_phase_and_start_new first.")

        # 将张量转换为numpy数组以便处理
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        # 反归一化图像 (假设输入是归一化的 [0, 1])
        images = np.clip(images * 255, 0, 255).astype(np.uint8)

        for i in range(min(sample_count, len(images))):
            img_tensor = images[i]
            target_mask = targets[i]
            pred_mask = predictions[i]

            # --- 保存原始图像 ---
            orig_img = Image.fromarray(img_tensor.transpose(1, 2, 0))  # CHW -> HWC
            orig_img.save(os.path.join(self.phase_sample_dir, f"sample_{i}_original.png"))

            # --- 保存目标标签图 (使用调色盘) ---
            target_colored = self._apply_color_map(target_mask, class_colors)
            target_colored.save(os.path.join(self.phase_sample_dir, f"sample_{i}_target.png"))

            # --- 保存预测结果图 (使用调色盘) ---
            pred_colored = self._apply_color_map(pred_mask, class_colors)
            pred_colored.save(os.path.join(self.phase_sample_dir, f"sample_{i}_prediction.png"))

        print(f"[Logger] Saved {min(sample_count, len(images))} sample(s) to {self.phase_sample_dir}")

    @staticmethod
    def _apply_color_map(mask: np.ndarray, colors: list) -> Image.Image:
        """
        为单通道的类别ID掩码应用调色盘，生成彩色图像。
        """
        # 创建一个RGB图像
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in enumerate(colors):
            if class_id >= len(colors): break
            mask_indices = (mask == class_id)
            colored_mask[mask_indices] = color

        return Image.fromarray(colored_mask, mode='RGB')