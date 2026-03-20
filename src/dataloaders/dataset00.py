import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pj_root import PROJECT_ROOT

CURRENT_DIR = os.getcwd()
RES_DIR = os.path.join(PROJECT_ROOT, 'resources')

class MyDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        image_dir = os.path.join(RES_DIR, image_dir)
        label_dir = os.path.join(RES_DIR, label_dir)

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            raise FileNotFoundError("数据集图像目录或标签目录不存在，请检查路径是否正确！")

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        image_files_list = os.listdir(self.image_dir)
        label_files_list = os.listdir(self.label_dir)

        image_set = set(image_files_list)
        label_set = set(label_files_list)

        if image_set != label_set:
            print("警告：图像文件和标签文件不匹配！请检查目录中的文件是否正确对应。")
            matched_set = image_set & label_set
            self.files = list(matched_set)  # 转为列表以便后续索引操作

            unmatched_images = image_set - matched_set
            unmatched_labels = label_set - matched_set

            print(
                f"在 image 文件夹中，发现的不匹配文件: {', '.join(list(unmatched_images)[:3]) + (f' 等 {len(unmatched_images)} 个文件' if len(unmatched_images) > 3 else '')}")
            print(
                f"在 label 文件夹中，发现的不匹配文件: {', '.join(list(unmatched_labels)[:3]) + (f' 等 {len(unmatched_labels)} 个文件' if len(unmatched_labels) > 3 else '')}")
        else:
            self.files = list(image_set)

        print(f"成功匹配文件: {', '.join(self.files[:3]) + (f' 等 {len(self.files)} 个文件' if len(self.files) > 3 else '')}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.files[idx])
        label_path = os.path.join(self.label_dir, self.files[idx])

        image = Image.open(image_path).convert('RGB')

        label = Image.open(label_path)
        if label.mode != 'P':
            print(f"警告：标签图像 {self.files[idx]} 的模式不是 'P'，请检查标签图像是否正确。")
            label = label.convert('P')

        img = np.array(image)
        lbl = np.array(label, dtype=np.int64)

        if self.transform:
            img = self.transform(img)

        return img, lbl, self.files[idx]

    def check_sample(self):
        rd_idx = np.random.randint(len(self.files))

        img, lbl, filename = self.__getitem__(rd_idx)

        img_w, img_h = img.shape[1], img.shape[0]
        lbl_w, lbl_h = lbl.shape[1], lbl.shape[0]

        img_c = img.shape[2] if len(img.shape) == 3 else 1
        lbl_c = lbl.shape[2] if len(lbl.shape) == 3 else 1

        lbl_u = np.unique(lbl)

        print('=' * 70)
        print(f"随机检查样本: {filename}")
        print(f"image 图像尺寸: {img_w}x{img_h}，通道数: {img_c}")
        print(f"label 图像尺寸: {lbl_w}x{lbl_h}，通道数: {lbl_c}")
        print(f"label 图像中值域: {lbl_u}")
        print('=' * 70)

def main():
    image_dir = os.path.join("dataset", "dataset00", "images")
    label_dir = os.path.join("dataset", "dataset00", "labels")

    dataset = MyDataset(image_dir, label_dir)

    print(f"数据集大小: {len(dataset)}")

    dataset.check_sample()
    dataset.check_sample()
    dataset.check_sample()

if __name__ == '__main__':
    main()
