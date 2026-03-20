import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
RES_DIR = os.path.join(PROJECT_ROOT, 'resources')

if __name__ == '__main__':
    if os.path.join(PROJECT_ROOT, 'src') not in sys.path:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

    if os.path.join(PROJECT_ROOT, 'src', 'utils') not in sys.path:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'utils'))
    if os.path.join(PROJECT_ROOT, 'src', 'dataloaders') not in sys.path:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'dataloaders'))

    import dataset00 as dataset00
    from ds_check import check_ds
    from md_check import check_md

    model_path = check_md(
        md_name="unet_camvid_rgb",
        kaggle_path="awsaf49/unet-semantic-segmentation/pytorch"
    )
    print("模型路径:", model_path)

    dataset_path = check_ds(
        ds_name="dataset00",
        kaggle_path=""
    )
    print("数据集路径:", dataset_path)


