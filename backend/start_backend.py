#!/usr/bin/env python3
"""
Startup script for RAG backend server
This script:
1. Checks database connection
2. Starts the backend server
3. Saves PID for later teardown
"""

import os
import sys
import time
import subprocess
import psycopg2
from pathlib import Path

# Colors for output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

# Database configuration from environment or defaults
DB_HOST = os.getenv("DB_HOST", "nv-service-d54c9117d23473fa7f28948da0635011")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "nuvolos")
DB_USER = os.getenv("DB_USER", "nuvolos")
DB_PASSWORD = os.getenv("DB_PASSWORD", "nuvolos")

# Backend configuration
BACKEND_PORT = os.getenv("BACKEND_PORT", "8500")

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
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


def check_database_connection():
    """Check if database is accessible."""
    print_header("Checking database connection...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.close()
        print_success("Database connection successful\n")
        return True
    except Exception as e:
        print_error("Cannot connect to database")
        print(f"Error: {e}")
        print("Please ensure PostgreSQL is running and accessible at:")
        print(f"  Host: {DB_HOST}")
        print(f"  Port: {DB_PORT}")
        print(f"  Database: {DB_NAME}")
        print(f"  User: {DB_USER}")
        return False


def check_if_running():
    """Check if backend is already running."""
    if BACKEND_PID_FILE.exists():
        try:
            pid = int(BACKEND_PID_FILE.read_text().strip())
            # Check if process is actually running
            os.kill(pid, 0)
            print_error(f"Backend server is already running (PID: {pid})")
            print(f"To stop it, run: python3 stop_backend.py")
            return True
        except (OSError, ValueError):
            # Process not running, clean up stale PID file
            BACKEND_PID_FILE.unlink()
    return False


def start_backend():
    """Start the backend server."""
    print_header("Starting backend server...")
    
    # Create log file
    backend_log = open(BACKEND_LOG_FILE, 'w')
    
    # Start backend server
    backend_process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=SCRIPT_DIR,
        stdout=backend_log,
        stderr=subprocess.STDOUT,
        env={**os.environ, **{
            "DB_HOST": DB_HOST,
            "DB_PORT": DB_PORT,
            "DB_NAME": DB_NAME,
            "DB_USER": DB_USER,
            "DB_PASSWORD": DB_PASSWORD
        }}
    )
    
    # Save PID
    BACKEND_PID_FILE.write_text(str(backend_process.pid))
    
    print_success(f"Backend server started (PID: {backend_process.pid})")
    print(f"Running on port {BACKEND_PORT}")
    print()
    print(f"Logs: tail -f {BACKEND_LOG_FILE}")
    
    # Wait a moment to check if process starts successfully
    time.sleep(2)
    
    if backend_process.poll() is not None:
        print_error("Backend server failed to start")
        print(f"Check logs at: {BACKEND_LOG_FILE}")
        return False
    
    return True


def main():
    """Main startup function."""
    print_colored(GREEN, "=== RAG Backend Startup ===\n")
    
    # Check if already running
    if check_if_running():
        sys.exit(1)
    
    # Check database connection
    if not check_database_connection():
        sys.exit(1)
    
    # Start backend server
    if not start_backend():
        sys.exit(1)
    
    # Success message
    print_colored(GREEN, "\n=== Backend Started! ===\n")
    print(f"API Endpoint: http://localhost:{BACKEND_PORT}")
    print(f"API Documentation: http://localhost:{BACKEND_PORT}/docs")
    print(f"Health Check: http://localhost:{BACKEND_PORT}/health")
    print()
    print("To stop the server, run: python3 stop_backend.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStartup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
