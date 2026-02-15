#!/usr/bin/env python3
"""
EMG Hand Movement Detector - Unified Startup Script
Starts both backend and frontend servers
"""

import subprocess
import sys
import time
import os
import signal

# Color codes for terminal
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message, color=Colors.END):
    """Print colored message"""
    print(f"{color}{message}{Colors.END}")

def check_requirements():
    """Check if required software is installed"""
    print_colored("🔍 Checking requirements...", Colors.BLUE)
    
    # Check Python
    try:
        result = subprocess.run(['python3', '--version'], 
                              capture_output=True, text=True, check=True)
        print_colored(f"✅ Python: {result.stdout.strip()}", Colors.GREEN)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print_colored("❌ Python3 not found! Please install Python 3.8+", Colors.RED)
        return False
    
    # Check Node/npm
    try:
        result = subprocess.run(['npm', '--version'], 
                              capture_output=True, text=True, check=True)
        print_colored(f"✅ npm: v{result.stdout.strip()}", Colors.GREEN)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print_colored("❌ npm not found! Please install Node.js 16+", Colors.RED)
        return False
    
    print_colored("✅ All requirements met!\n", Colors.GREEN)
    return True

def start_backend():
    """Start the Flask backend server"""
    print_colored("📡 Starting Backend Server (Python Flask)...", Colors.BLUE)
    
    # Start backend
    process = subprocess.Popen(
        ['python3', 'server.py'],
        cwd='backend',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    return process

def start_frontend():
    """Start the React frontend server"""
    print_colored("🌐 Starting Frontend Server (React)...", Colors.BLUE)
    
    # Use 'emd' folder instead of 'frontend'
    process = subprocess.Popen(
        ['npm', 'start'],
        cwd='emd',  # Changed from 'frontend' to 'emd'
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    return process

def main():
    """Main startup function"""
    print_colored("""
╔══════════════════════════════════════════════╗
║   🧠 EMG Hand Movement Detector              ║
║   Real-time AI-Powered Gesture Detection    ║
╚══════════════════════════════════════════════╝
    """, Colors.BOLD)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    processes = []
    
    try:
        # Start backend
        backend_process = start_backend()
        processes.append(('Backend', backend_process))
        
        # Wait for backend to initialize
        print_colored("⏳ Waiting for backend to initialize (3 seconds)...", Colors.YELLOW)
        time.sleep(3)
        
        # Start frontend
        frontend_process = start_frontend()
        processes.append(('Frontend', frontend_process))
        
        # Wait a bit for frontend to start
        time.sleep(2)
        
        # Success message
        print_colored("""
╔══════════════════════════════════════════════╗
║   ✅ Both servers started successfully!      ║
╚══════════════════════════════════════════════╝

📡 Backend:  http://localhost:5000
🌐 Frontend: http://localhost:3000

⚡ The browser should open automatically!

Press Ctrl+C to stop both servers
        """, Colors.GREEN)
        
        # Keep running until interrupted
        while True:
            # Check if any process died
            for name, proc in processes:
                if proc.poll() is not None:
                    print_colored(f"\n⚠️  {name} process stopped unexpectedly!", Colors.RED)
                    raise Exception(f"{name} crashed")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print_colored("\n\n🛑 Shutting down servers...", Colors.YELLOW)
        for name, process in processes:
            print_colored(f"   Stopping {name}...", Colors.BLUE)
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print_colored("\n✅ All servers stopped successfully!", Colors.GREEN)
        sys.exit(0)
        
    except Exception as e:
        print_colored(f"\n❌ Error: {e}", Colors.RED)
        print_colored("Cleaning up...", Colors.YELLOW)
        for name, process in processes:
            process.terminate()
            try:
                process.wait(timeout=3)
            except:
                process.kill()
        sys.exit(1)

if __name__ == "__main__":
    main()
