import os
import sys

my_dir = os.getcwd()

try:
    # ============ 添加 dataloaders 路径 ============
    dataloaders_path = os.path.abspath(os.path.join(my_dir, 'dataloaders'))
    # 规范化 sys.path 中所有路径用于安全比较（处理大小写/斜杠差异）
    normalized_sys_path = [os.path.abspath(p) for p in sys.path]

    if dataloaders_path not in normalized_sys_path:
        sys.path.insert(0, dataloaders_path)  # 修正：insert(位置, 路径)
    from dataset00 import MyDataset  # 移到 if 外：确保无论路径是否已存在都执行导入

    # ============ 添加 utils 路径 ============
    utils_path = os.path.abspath(os.path.join(my_dir, 'utils'))
    if utils_path not in normalized_sys_path:  # 重新计算避免被上一步修改影响
        sys.path.insert(0, utils_path)
    from ds_check import check_ds
    from md_check import check_md

except ImportError as e:
    print(f'❌ 模块导入失败: {e}')
    print(f'   dataloaders路径: {dataloaders_path}')
    print(f'   utils路径: {utils_path}')
    print(f'   当前 sys.path 预览: {sys.path[:3]}...')  # 辅助调试
    raise  # 保留异常栈便于定位
except Exception as e:
    print(f'❌ 路径处理异常: {type(e).__name__}: {e}')
    raise