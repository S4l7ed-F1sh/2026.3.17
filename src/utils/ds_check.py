import os
import kagglehub # 补充导入：数据集下载需使用 kagglehub.dataset_download
import huggingface_hub
from pj_root import PROJECT_ROOT

RES_DIR = os.path.join(PROJECT_ROOT, 'resources')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# print(f"当前脚本路径: {CURRENT_DIR}")
# print(f"项目根路径: {PROJECT_ROOT}")
# print(f"资源目录路径: {RES_DIR}")

if not os.path.exists(RES_DIR):
    print("警告：检测到 resources 文件夹不存在，请检查路径是否正确！")
    os.makedirs(RES_DIR)
if not os.path.exists(SRC_DIR):
    print("警告：检测到 src 文件夹不存在，请检查路径是否正确！")
    os.makedirs(SRC_DIR)

def check_ds(ds_name, kaggle_path = None, hg_path = None):
    '''
     检查数据集是否存在，如果不存在则从 Kaggle 下载

    :param ds_name: 数据集的名称，下载的数据集将被保存在 resources/dataset/ds_name 目录下
    :param kaggle_path: Kaggle 数据集的完整路径，格式为 "用户名/数据集名"
    :param hg_path: Hugging Face 数据集的完整路径，格式为 "用户名/数据集名"

    :return: 数据集存储的绝对路径，即 resources/dataset/ds_name

        该函数会先检查 resources/dataset/ds_name 是否存在，如果存在则直接返回该路径；
        如果不存在则使用 kagglehub 从 Kaggle 下载数据集，并保存到该路径，然后返回该路径。
    '''
    target_path = os.path.join(RES_DIR, 'dataset')
    target_dir = os.path.join(RES_DIR, 'dataset', ds_name)

    if not os.path.exists(target_path):
        os.makedirs(target_path)  # 确保 dataset 目录存在

    if os.path.exists(target_dir):
        print(f"{ds_name} 数据集已存在，路径: {target_dir}")
    else:
        print(f"{ds_name} 数据集不存在，正在下载...")
        if kaggle_path is None and hg_path is None:
            print("警告：未提供 Kaggle 或 Hugging Face 数据集路径，无法下载数据集。请检查参数是否正确！")
            raise ValueError("Kaggle 数据集路径和 Hugging Face 数据集路径不能同时为空字符串。")

        if kaggle_path is not None:
            try:
                kagglehub.dataset_download(handle=kaggle_path, output_dir=target_dir)
                print(f"✓ 数据集下载完成！保存至: {target_dir}")
            except Exception as e:
                print(f"✗ Kaggle 数据集下载失败: {str(e)}")
                raise  # 保留异常以便上层处理
        elif hg_path is not None:
            try:
                huggingface_hub.hf_hub_download(repo_id=hg_path, repo_type='dataset', local_dir=target_dir)
                print(f"✓ 数据集下载完成！保存至: {target_dir}")
            except Exception as e:
                print(f"✗ Hugging Face 数据集下载失败: {str(e)}")
                raise  # 保留异常以便上层处理

    return os.path.abspath(target_dir)  # 返回标准化绝对路径，便于后续使用