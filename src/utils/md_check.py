import os
import kagglehub  # 补充导入：模型下载需使用 kagglehub.model_download
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

def check_md(md_name, kaggle_path = None, hg_path = None):
    target_dir = os.path.join(RES_DIR, 'model', md_name)

    if os.path.exists(target_dir):
        print(f"✓ {md_name} 模型已存在，路径: {target_dir}")
        return os.path.abspath(target_dir)

    print(f"↓ {md_name} 模型不存在，准备下载...")

    # 参数校验
    if not kaggle_path and not hg_path:
        raise ValueError("必须提供 Kaggle 或 Hugging Face 模型路径")
    if kaggle_path and hg_path:
        raise ValueError("Kaggle 和 Hugging Face 路径只能选择其一")

    # 创建目录
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    try:
        if kaggle_path:
            print(f"→ 从 Kaggle 下载: {kaggle_path}")
            kagglehub.model_download(handle=kaggle_path, output_dir=target_dir)

        elif hg_path:
            print(f"→ 从 Hugging Face 下载: {hg_path}")
            # 关键修正：使用 snapshot_download 下载整个仓库
            huggingface_hub.snapshot_download(
                repo_id=hg_path,
                repo_type="model",
                local_dir=target_dir,
                local_dir_use_symlinks=False,  # 避免符号链接问题
                token=False  # 无需 token（公开模型）
            )

        print(f"✓ 下载完成！保存至: {target_dir}")
        return os.path.abspath(target_dir)

    except Exception as e:
        # 清理失败下载的残留目录
        if os.path.exists(target_dir) and not os.listdir(target_dir):
            os.rmdir(target_dir)
        print(f"✗ 下载失败: {str(e)}")
        raise

# if __name__ == '__main__':
#     # 测试函数
#     try:
#         model_path = check_md(
#             md_name="pet_unet",  # 本地保存目录名（自定义）
#             hg_path="selfmaker/unet_segmentation_pet"  # Hugging Face 模型ID
#         )
#         print("模型路径:", model_path)
#
#         model = tf.keras.models.load_model(os.path.join(model_path, 'segmentation_pets.h5'))
#
#         # 查看模型结构
#         model.summary()
#     except Exception as e:
#         print(f"测试失败: {str(e)}")
