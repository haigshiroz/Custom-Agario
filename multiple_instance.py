import subprocess
import os
import platform

# Getting the current working directory
current_directory = os.getcwd()
agario_path = os.path.join(current_directory, "agario.py")

# For MacOS
if platform.system() == 'Darwin':
    command = f"cd '{current_directory}' && conda deactivate && python3 '{agario_path}' --server"
    subprocess.run(['osascript', '-e', f'tell application "Terminal" to do script "{command}"'])

    times_to_run = 10
    for i in range(times_to_run):
        command = f"cd '{current_directory}' && conda deactivate && python3 '{agario_path}'"
        subprocess.run(['osascript', '-e', f'tell application "Terminal" to do script "{command}"'])
        
# For Windows
elif platform.system() == 'Windows':
    # subprocess.run(['start', 'cmd.exe', '/K', f'conda deactivate && python {agario_path} --server'])
    subprocess.Popen(['powershell', '-NoExit', '-Command', f'python {agario_path} --server'])

    times_to_run = 10
    for i in range(times_to_run):
        # subprocess.run(['start', 'cmd.exe', '/K', f'conda deactivate && python {agario_path}'])
        subprocess.Popen(['powershell', '-NoExit', '-Command', f'python {agario_path}'])


