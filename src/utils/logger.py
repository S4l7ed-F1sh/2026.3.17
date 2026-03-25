# logger.py

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pj_root import PROJECT_ROOT
import torch


class SemanticSegmentationLogger:
    """
    A logger for semantic segmentation model training.
    It manages directories, logs text data to specific files,
    saves images, and generates plots for metrics and samples.
    """

    def __init__(self, title: str):
        """
        Initializes the logger.

        Args:
            title (str): The title for the experiment/run.
        """
        self.title = title
        self.run_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directory structure: resource/Logger/<run_start_time>/<title>
        self.log_root_dir = os.path.join(PROJECT_ROOT, "resources", "Logger", self.run_start_time, title)
        os.makedirs(self.log_root_dir, exist_ok=True)

        # Initialize phase-specific attributes
        self.current_phase = 0
        self.phase_data = {}
        self._init_new_phase()

    def _init_new_phase(self):
        """Initializes data structures for a new training phase."""
        self.current_phase += 1
        self.phase_data[self.current_phase] = {
            'model_name': None,
            'dataset_name': None,
            'train_size': None,
            'test_size': None,
            'optimizer_method': None,
            'epochs_data': []  # List of dictionaries containing epoch metrics
        }

        # Create a new directory for the current phase's samples
        self.current_phase_sample_dir = os.path.join(self.log_root_dir, f"phase_{self.current_phase}_samples")
        os.makedirs(self.current_phase_sample_dir, exist_ok=True)

    def log_initialization(
            self,
            model_name: str,
            dataset_name: str,
            train_size: int,
            test_size: int,
            optimizer_method: str,
    ):
        """
        Logs initial information at the beginning of a new phase.
        This should be called once per phase, before any epoch logging.

        Args:
            model_name (str): Name of the model being trained.
            dataset_name (str): Name of the dataset used.
            train_size (int): Number of samples in the training set.
            test_size (int): Number of samples in the test set.
            optimizer_method (str): Optimization method used (e.g., Adam, SGD).
        """
        if self.current_phase < 0:
            raise RuntimeError("A new phase must be initialized before logging initialization data.")

        phase_info = self.phase_data[self.current_phase]
        phase_info['model_name'] = model_name
        phase_info['dataset_name'] = dataset_name
        phase_info['train_size'] = train_size
        phase_info['test_size'] = test_size
        phase_info['optimizer_method'] = optimizer_method

        # Write initialization info to a dedicated file for this phase
        init_log_path = os.path.join(self.log_root_dir, f"phase_{self.current_phase}_initialization.txt")
        with open(init_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Dataset Name: {dataset_name}\n")
            f.write(f"Train Set Size: {train_size}\n")
            f.write(f"Test Set Size: {test_size}\n")
            f.write(f"Optimizer Method: {optimizer_method}\n")
            f.write(f"Phase Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 30 + "\n")

    def log_epoch(
            self,
            epoch_num: int,
            train_loss: float,
            val_loss: float,
            train_time: float,
            val_time: float,
            learning_rate: float,
            train_miou: float,
            val_miou: float,
    ):
        """
        Logs metrics for a completed epoch.

        Args:
            epoch_num (int): The number of the epoch that just finished.
            train_loss (float): Average loss on the training set for this epoch.
            val_loss (float): Average loss on the validation/test set for this epoch.
            train_time (float): Time spent on the training pass for this epoch (in seconds).
            val_time (float): Time spent on the validation/test pass for this epoch (in seconds).
            learning_rate (float): The learning rate value at the end of this epoch.
            train_miou (float): Mean Intersection over Union (mIoU) on the training set.
            val_miou (float): Mean Intersection over Union (mIoU) on the validation/test set.
        """
        epoch_data = {
            "epoch": epoch_num,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_time": train_time,
            "val_time": val_time,
            "learning_rate": learning_rate,
            "train_miou": train_miou,
            "val_miou": val_miou,
        }
        self.phase_data[self.current_phase]['epochs_data'].append(epoch_data)

        # Append epoch data to the phase-specific metrics file
        metrics_log_path = os.path.join(self.log_root_dir, f"phase_{self.current_phase}_metrics.txt")
        with open(metrics_log_path, 'a', encoding='utf-8') as f:
            f.write(
                f"{epoch_num},{train_loss:.6f},{val_loss:.6f},"
                f"{train_time:.4f},{val_time:.4f},{learning_rate:.8f},"
                f"{train_miou:.4f},{val_miou:.4f}\n"
            )

    def save_samples(self, outputs_and_labels: List[Tuple[np.ndarray, np.ndarray]],
                     class_colors: List[Tuple[int, int, int]],
                     sample_prefix: Optional[str] = "0"):
        """
        Saves a sample of images and their predicted/ground truth masks.

        Args:
            outputs_and_labels (List[Tuple[np.ndarray, np.ndarray]]): A list of tuples.
                Each tuple contains an image (H, W, C) and its corresponding mask (H, W).
            class_colors (List[Tuple[int, int, int]]): A list of RGB tuples for each class.
                e.g., for 3 classes -> [(R, G, B), ...].
        """
        num_samples = len(outputs_and_labels)
        for i, (output, label) in enumerate(outputs_and_labels):
            # --- Create and Save Colored Mask ---
            # The mask is expected to have integer values representing class IDs.
            # We map these IDs to RGB colors using the provided palette.
            h, w = output.shape
            colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

            # Map each pixel's class ID to its color
            for cls_id, color in enumerate(class_colors):
                colored_mask[output == cls_id] = color

            mask_pil = Image.fromarray(colored_mask, mode="RGB")
            mask_path = os.path.join(self.current_phase_sample_dir, f"sample_{sample_prefix}_{i}_output.png")
            mask_pil.save(mask_path)

            # --- Create and Save Colored Mask ---
            # The mask is expected to have integer values representing class IDs.
            # We map these IDs to RGB colors using the provided palette.
            h, w = label.shape
            colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

            # Map each pixel's class ID to its color
            for cls_id, color in enumerate(class_colors):
                colored_mask[label == cls_id] = color

            mask_pil = Image.fromarray(colored_mask, mode="RGB")
            mask_path = os.path.join(self.current_phase_sample_dir, f"sample_{sample_prefix}_{i}_answer.png")
            mask_pil.save(mask_path)

    def finalize_phase_and_plot(self):
        """
        Finalizes the current phase by saving its data and generating plots.
        Then, initializes a new phase.
        """
        if not self.phase_data[self.current_phase]['epochs_data']:
            print(f"No epoch data found for phase {self.current_phase}. Skipping plot generation.")
            self._init_new_phase()
            return

        epochs_data = self.phase_data[self.current_phase]['epochs_data']
        epochs_list = [d['epoch'] for d in epochs_data]
        train_losses = [d['train_loss'] for d in epochs_data]
        val_losses = [d['val_loss'] for d in epochs_data]
        train_mious = [d['train_miou'] for d in epochs_data]
        val_mious = [d['val_miou'] for d in epochs_data]
        lrs = [d['learning_rate'] for d in epochs_data]

        # --- Plot Phase-Specific Curves ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Phase {self.current_phase} Training Summary')

        # Loss
        axes[0, 0].plot(epochs_list, train_losses, label='Train Loss', marker='o')
        axes[0, 0].plot(epochs_list, val_losses, label='Validation Loss', marker='s')
        axes[0, 0].set_title('Loss over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # mIoU
        axes[0, 1].plot(epochs_list, train_mious, label='Train mIoU', marker='o')
        axes[0, 1].plot(epochs_list, val_mious, label='Validation mIoU', marker='s')
        axes[0, 1].set_title('mIoU over Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mIoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning Rate
        axes[1, 0].plot(epochs_list, lrs, label='Learning Rate', color='red')
        axes[1, 0].set_title('Learning Rate over Epochs')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Time (example using train time)
        train_times = [d['train_time'] for d in epochs_data]
        val_times = [d['val_time'] for d in epochs_data]
        axes[1, 1].plot(epochs_list, train_times, label='Train Time (s)', marker='^')
        axes[1, 1].plot(epochs_list, val_times, label='Val Time (s)', marker='v')
        axes[1, 1].set_title('Time per Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.log_root_dir, f"phase_{self.current_phase}_summary_plots.png")
        plt.savefig(plot_path)
        plt.close(fig)  # Close the figure to free memory

        print(f"Phase {self.current_phase} finalized. Plots saved to {plot_path}.")

        # Initialize the next phase
        self._init_new_phase()

    def finalize_run_and_create_final_plots(self):
        """
        Finalizes the entire run by generating comprehensive plots
        from all collected phases' data.
        """
        all_epochs, all_train_l, all_val_l, all_train_m, all_val_m = [], [], [], [], []
        current_epoch_offset = 0

        for p_id in sorted(self.phase_data.keys()):
            phase_epochs_data = self.phase_data[p_id]['epochs_data']
            if not phase_epochs_data: continue

            # Offset epochs to create a continuous x-axis across phases
            p_epochs = [d['epoch'] + current_epoch_offset for d in phase_epochs_data]
            current_epoch_offset = p_epochs[-1] + 1

            all_epochs.extend(p_epochs)
            all_train_l.extend([d['train_loss'] for d in phase_epochs_data])
            all_val_l.extend([d['val_loss'] for d in phase_epochs_data])
            all_train_m.extend([d['train_miou'] for d in phase_epochs_data])
            all_val_m.extend([d['val_miou'] for d in phase_epochs_data])

        if not all_epochs:
            print("No data from any phase to generate final plots.")
            return

        # --- Plot Overall Run Curves ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Overall Loss
        axes[0].plot(all_epochs, all_train_l, label='Train Loss', alpha=0.7, marker='.')
        axes[0].plot(all_epochs, all_val_l, label='Validation Loss', alpha=0.7, marker='.')
        axes[0].set_title('Overall Training & Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Overall mIoU
        axes[1].plot(all_epochs, all_train_m, label='Train mIoU', alpha=0.7, marker='.')
        axes[1].plot(all_epochs, all_val_m, label='Validation mIoU', alpha=0.7, marker='.')
        axes[1].set_title('Overall Training & Validation mIoU')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mIoU')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.log_root_dir, f"final_combined_plots.png")
        plt.savefig(plot_path)
        plt.close(fig)

        print(f"All phases complete. Final combined plots saved to {plot_path}.")

    def save_model_checkpoint(self, model: torch.nn.Module, phase_id: int = None, custom_suffix: str = ""):
        """
        保存模型的当前状态（权重）。

        Args:
            model (torch.nn.Module): 要保存的 PyTorch 模型。
            phase_id (int, optional): 指定要保存到哪个阶段的目录。如果为 None，则默认保存到当前阶段。
            custom_suffix (str, optional): 文件名的自定义后缀，例如 '_best' 或 '_epoch10'。

        Raises:
            RuntimeError: 如果指定的 phase_id 尚未初始化且没有数据。
        """
        # 如果未指定 phase_id，默认使用当前 phase
        target_phase = phase_id if phase_id is not None else self.current_phase

        # 检查目标阶段是否有数据（确保该阶段已经通过 log_initialization 初始化过）
        if target_phase not in self.phase_data or not self.phase_data[target_phase]['epochs_data']:
            raise ValueError(f"无法保存模型：阶段 {target_phase} 不存在或未初始化。请先调用 log_initialization。")

        # 构建保存路径：log_root_dir/phase_X_checkpoints/...
        phase_checkpoint_dir = os.path.join(self.log_root_dir, f"phase_{target_phase}_checkpoints")
        os.makedirs(phase_checkpoint_dir, exist_ok=True)  # 确保目录存在

        # 生成文件名，例如：model_phase_0_best.pth
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"model_phase_{target_phase}{custom_suffix}_{timestamp}.pth"
        save_path = os.path.join(phase_checkpoint_dir, filename)

        # 保存模型状态字典
        # 注意：这里保存的是 state_dict。如果你需要保存整个模型结构，可以使用 torch.save(model, save_path)
        torch.save(model.state_dict(), save_path)

        print(f"[Model Save] 模型已保存至: {save_path}")
        return save_path
