import subprocess
import os
import platform

# Getting the current working directory
current_directory = os.getcwd()
agario_path = os.path.join(current_directory, "agario.py")

# For MacOS
if platform.system() == 'Darwin':
    subprocess.run(['osascript', '-e', f'tell application "Terminal" to do script "conda deactivate && python3 {agario_path} --server"'])

    times_to_run = 10
    for i in range(times_to_run):
        subprocess.run(['osascript', '-e', f'tell application "Terminal" to do script "conda deactivate && python3 {agario_path}"'])

# For Windows
elif platform.system() == 'Windows':
    subprocess.run(['start', 'cmd.exe', '/K', f'conda deactivate && python {agario_path} --server'])

    times_to_run = 10
    for i in range(times_to_run):
        subprocess.run(['start', 'cmd.exe', '/K', f'conda deactivate && python {agario_path}'])