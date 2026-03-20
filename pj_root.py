import os

PROJECT_ROOT = "";

try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except ValueError as e:
    if '__vsc_ipynb_file__' in globals():
        start_dir = os.path.dirname(os.path.abspath(globals()['__vsc_ipynb_file__']))
    else:
        start_dir = os.getcwd()

    current_dir = start_dir
    while os.path.dirname(current_dir) != current_dir:
        current_dir = os.path.dirname(current_dir)
        if os.path.exists(os.path.join(current_dir, 'src')):
            PROJECT_ROOT = current_dir
            break

    if PROJECT_ROOT == "":
        raise FileNotFoundError("无法确定项目根目录，请确保项目结构正确，并且在正确的环境中运行脚本。")