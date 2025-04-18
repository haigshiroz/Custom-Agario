import subprocess
import os
import platform
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Launch multiple agar.io instances",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:
  Basic usage:
    python multiple_instance.py --train           # Server + 10 training agents (default)
    python multiple_instance.py --test --agents 5 # 5 testing agents + server
  
  Training options:
    python multiple_instance.py --train --empty   # Fresh training (clear Q-tables)
    python multiple_instance.py --train --history # Continue previous training
  
  Server control:  
    python multiple_instance.py --train --no-server # Only launch 10 training agents
    python multiple_instance.py --test --agents 3   # 3 testing agents + server
""")
    
    # Training/testing mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--train', action='store_true', help='Run in training mode (epsilon=1.0)')
    mode_group.add_argument('--test', action='store_true', help='Run in testing mode (epsilon=0.0)')
    
    # Training options
    parser.add_argument('--empty', action='store_true', help='Initialize empty Q-tables (default for training)')
    parser.add_argument('--history', action='store_true', help='Continue training from existing Q-tables')
    
    # Instance control
    parser.add_argument('--agents', type=int, default=10, help='Number of client instances to launch (default: 10)')
    parser.add_argument('--no-server', action='store_true', help='Skip launching server')
    
    args = parser.parse_args()

    current_directory = os.getcwd()
    agario_path = os.path.join(current_directory, "agario.py")

    # Build command components
    mode_arg = '--train' if args.train else '--test' if args.test else ''
    train_args = ''
    if args.train:
        if args.empty:
            train_args = '--empty'
        elif args.history:
            train_args = '--history'

    # Platform-specific launching
    if platform.system() == 'Darwin':
        # Launch server unless disabled
        if not args.no_server:
            server_cmd = f"cd '{current_directory}' && python3 '{agario_path}' --server {mode_arg} {train_args}"
            subprocess.run(['osascript', '-e', f'tell application "Terminal" to do script "{server_cmd}"'])
        
        # Launch clients
        for _ in range(args.agents):
            client_cmd = f"cd '{current_directory}' && python3 '{agario_path}' {mode_arg} {train_args}"
            subprocess.run(['osascript', '-e', f'tell application "Terminal" to do script "{client_cmd}"'])
            
    elif platform.system() == 'Windows':
        # Launch server unless disabled
        if not args.no_server:
            server_cmd = f'python "{agario_path}" --server {mode_arg} {train_args}'
            subprocess.Popen(['powershell', '-NoExit', '-Command', server_cmd])
        
        # Launch clients
        for _ in range(args.agents):
            client_cmd = f'python "{agario_path}" {mode_arg} {train_args}'
            subprocess.Popen(['powershell', '-NoExit', '-Command', client_cmd])

if __name__ == "__main__":
    main()