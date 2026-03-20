import subprocess

try:
    res = subprocess.check_output(["git", "push", "mastergit "], stderr=subprocess.STDOUT, text=True)
    print(res)
except subprocess.CalledProcessError as e:
    print(f"Git push 失败，错误信息:\n{e.output}")
