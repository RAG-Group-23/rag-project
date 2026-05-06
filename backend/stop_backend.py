#!/usr/bin/env python3
"""
Teardown script for RAG backend server
This script:
1. Stops the running backend server
2. Cleans up PID and log files
"""

import os
import sys
import time
import signal
from pathlib import Path

# Colors for output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

# Paths
PID_DIR = Path("/tmp")
BACKEND_PID_FILE = PID_DIR / "rag_backend.pid"
BACKEND_LOG_FILE = PID_DIR / "backend.log"


def print_colored(color, message):
    """Print colored message."""
    print(f"{color}{message}{NC}")


def print_header(message):
    """Print section header."""
    print_colored(YELLOW, f"\n{message}")


def print_success(message):
    """Print success message."""
    print_colored(GREEN, f"✓ {message}")


def print_error(message):
    """Print error message."""
    print_colored(RED, f"✗ {message}")


def stop_backend():
    """Stop the backend server."""
    print_header("Stopping backend server...")
    
    if not BACKEND_PID_FILE.exists():
        print("Backend server is not running (no PID file found)")
        return True
    
    try:
        pid = int(BACKEND_PID_FILE.read_text().strip())
        print(f"Stopping backend server (PID: {pid})...")
        
        # Check if process exists
        try:
            os.kill(pid, 0)  # Signal 0 just checks if process exists
        except OSError:
            print("Backend server is not running")
            BACKEND_PID_FILE.unlink()
            return True
        
        # Try graceful shutdown first
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            print(f"Error sending SIGTERM: {e}")
        
        # Wait for graceful shutdown
        for i in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(1)
            except OSError:
                # Process is dead
                break
        
        # Force kill if still running
        try:
            os.kill(pid, 0)
            print("Process did not terminate gracefully, forcing shutdown...")
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
        except OSError:
            pass
        
        print_success("Backend server stopped")
        BACKEND_PID_FILE.unlink()
        return True
        
    except (ValueError, FileNotFoundError) as e:
        print_error(f"Error reading PID file: {e}")
        return False


def cleanup_files():
    """Clean up log files."""
    print_header("Cleaning up files...")
    
    if BACKEND_LOG_FILE.exists():
        BACKEND_LOG_FILE.unlink()
        print_success("Removed log file")
    else:
        print("No log file to clean up")


def main():
    """Main teardown function."""
    print_colored(YELLOW, "=== RAG Backend Teardown ===\n")
    
    # Stop backend server
    if not stop_backend():
        sys.exit(1)
    
    # Clean up files
    cleanup_files()
    
    # Success message
    print_colored(GREEN, "\n=== Backend Stopped! ===\n")
    print("To start the backend again, run: python3 start_backend.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTeardown interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
