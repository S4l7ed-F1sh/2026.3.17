import os
import kagglehub  # 补充导入：模型下载需使用 kagglehub.model_download

try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../src/utils
    PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))  # .../项目根
except ValueError as e:
    if '__vsc_ipynb_file__' in globals():
        start_dir = os.path.dirname(os.path.abspath(globals()['__vsc_ipynb_file__']))
    else:
        start_dir = os.getcwd()

    current_dir = start_dir
    PROJECT_ROOT = None
    while os.path.dirname(current_dir) != current_dir:
        current_dir = os.path.dirname(current_dir)
        if os.path.exists(os.path.join(current_dir, 'src')):
            PROJECT_ROOT = current_dir
            break

    if PROJECT_ROOT is None:
        raise FileNotFoundError("无法确定项目根目录，请确保项目结构正确，并且在正确的环境中运行脚本。")

RES_DIR = os.path.join(PROJECT_ROOT, 'resources')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# print(f"当前脚本路径: {CURRENT_DIR}")
# print(f"项目根路径: {PROJECT_ROOT}")
# print(f"资源目录路径: {RES_DIR}")

if not os.path.exists(RES_DIR):
    print("警告：检测到 resources 文件夹不存在，请检查路径是否正确！")
    os.makedirs(RES_DIR)

def check_md(md_name, kaggle_path):
    """
    检查并下载 Kaggle 模型

    参数:
        md_name (str): 本地保存的模型目录名称
        kaggle_path (str): Kaggle 模型完整路径 (格式: "用户名/模型名/框架", 例如 "google/bert/tensorflow2")

    返回:
        str: 模型存储的绝对路径
    """
    target_path = os.path.join(RES_DIR, 'model')
    target_dir = os.path.join(RES_DIR, 'model', md_name)  # 与 dataset 目录平行，结构清晰

    if not os.path.exists(target_path):
        os.makedirs(target_path)  # 确保 model 目录存在

    if os.path.exists(target_dir):
        print(f"✓ {md_name} 模型已存在，路径: {target_dir}")
    else:
        print(f"↓ {md_name} 模型不存在，正在从 Kaggle 下载...")
        if kaggle_path == "":
            print("警告：未提供 Kaggle 模型路径，无法下载模型。请检查参数是否正确！")
            raise ValueError("Kaggle 模型路径不能为空字符串。")

        try:
            kagglehub.model_download(handle=kaggle_path, output_dir=target_dir)
            print(f"✓ 模型下载完成！保存至: {target_dir}")
        except Exception as e:
            print(f"✗ 模型下载失败: {str(e)}")
            raise  # 保留异常以便上层处理

    return os.path.abspath(target_dir)  # 返回标准化绝对路径，便于后续使用