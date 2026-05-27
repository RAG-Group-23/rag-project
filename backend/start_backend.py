#!/usr/bin/env python3
"""
Startup script for RAG backend server.

Steps:
1. Check database connection.
2. Start the backend server (main.py via uvicorn).
3. Save PID to /tmp for later teardown.

Model loading modes (mutually exclusive flags):
  --full        LOAD_MODELS=true  + hardcoded Qwen embedding + Gemma LLM  [default]
  --embed-only  LOAD_MODELS=false + hardcoded Qwen embedding, no LLM
  --no-models   No model env vars injected — reads from your shell environment
"""

import argparse
import os
import sys
import time
import subprocess
import psycopg2
from pathlib import Path

# ----------------------------------------
# Output colours
# ----------------------------------------
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
CYAN = '\033[0;36m'
NC = '\033[0m'

# ----------------------------------------
# Configuration — mirrors db.py defaults
# ----------------------------------------
DB_HOST = os.getenv(
    "DB_HOST",     "nv-service-d54c9117d23473fa7f28948da0635011")
DB_PORT = os.getenv("DB_PORT",     "5432")
DB_NAME = os.getenv("DB_NAME",     "nuvolos")
DB_USER = os.getenv("DB_USER",     "nuvolos")
DB_PASSWORD = os.getenv("DB_PASSWORD", "nuvolos")

BACKEND_PORT = os.getenv("BACKEND_PORT", "8500")

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_LLM_MODEL = "google/gemma-3-4b-it"

# ----------------------------------------
# Paths
# ----------------------------------------
SCRIPT_DIR = Path(__file__).parent.absolute()
PID_DIR = Path("/tmp")
BACKEND_PID_FILE = PID_DIR / "rag_backend.pid"
BACKEND_LOG_FILE = PID_DIR / "backend.log"


def print_colored(color, message):
    print(f"{color}{message}{NC}")


def print_header(message):
    print_colored(YELLOW, f"\n{message}")


def print_success(message):
    print_colored(GREEN, f"✓ {message}")


def print_error(message):
    print_colored(RED, f"✗ {message}")


def print_info(message):
    print_colored(CYAN, f"  {message}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start the RAG backend server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model loading modes (pick one):
  (default)     LOAD_MODELS=true  — loads both Qwen embeddings + Gemma LLM
  --embed-only  LOAD_MODELS=false — loads Qwen embeddings only, skips LLM
  --no-models   No model env vars injected — uses whatever is in your shell
                  (set them yourself: export LLM_MODEL=... EMBEDDING_MODEL=...)
        """,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--embed-only",
        action="store_true",
        help=(
            "Start with LOAD_MODELS=false. Loads Qwen embeddings but skips the LLM. "
            "Useful for testing the chunker/embeddings pipeline without the LLM overhead."
        ),
    )
    mode_group.add_argument(
        "--no-models",
        action="store_true",
        help=(
            "Start without injecting any model env vars. "
            "The backend will rely entirely on your shell environment "
            "(export LLM_MODEL=... and EMBEDDING_MODEL=... beforehand)."
        ),
    )

    return parser.parse_args()


def build_model_env(args) -> dict:
    """Return the model-related env vars to inject based on the chosen mode."""
    if args.no_models:
        # Pass nothing — shell exports are inherited via **os.environ below
        return {}
    elif args.embed_only:
        return {
            "LOAD_MODELS":      "false",
            "EMBEDDING_MODEL":  DEFAULT_EMBEDDING_MODEL,
        }
    else:
        # Default: full mode
        return {
            "LOAD_MODELS":      "true",
            "EMBEDDING_MODEL":  DEFAULT_EMBEDDING_MODEL,
            "LLM_MODEL":        DEFAULT_LLM_MODEL,
        }


def describe_mode(args):
    if args.no_models:
        print_info("Mode: no-models  — model env vars read from your shell")
        emb = os.getenv("EMBEDDING_MODEL", "(not set)")
        llm = os.getenv("LLM_MODEL",       "(not set)")
        print_info(f"  EMBEDDING_MODEL = {emb}")
        print_info(f"  LLM_MODEL       = {llm}")
    elif args.embed_only:
        print_info("Mode: embed-only — LOAD_MODELS=false, Qwen embeddings only")
        print_info(f"  EMBEDDING_MODEL = {DEFAULT_EMBEDDING_MODEL}")
        print_info("  LLM_MODEL       = (not loaded)")
    else:
        print_info("Mode: full       — LOAD_MODELS=true, embeddings + LLM")
        print_info(f"  EMBEDDING_MODEL = {DEFAULT_EMBEDDING_MODEL}")
        print_info(f"  LLM_MODEL       = {DEFAULT_LLM_MODEL}")


def check_database_connection() -> bool:
    print_header("Checking database connection...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT,
            database=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        )
        conn.close()
        print_success("Database connection successful\n")
        return True
    except Exception as e:
        print_error("Cannot connect to database")
        print(f"  Error: {e}")
        print(
            f"  Host: {DB_HOST}  Port: {DB_PORT}  DB: {DB_NAME}  User: {DB_USER}")
        return False


def check_if_running() -> bool:
    """Return True (and warn) if the backend is already running."""
    if BACKEND_PID_FILE.exists():
        try:
            pid = int(BACKEND_PID_FILE.read_text().strip())
            os.kill(pid, 0)  # raises OSError if dead
            print_error(f"Backend server is already running (PID: {pid})")
            print("  To stop it, run: python3 stop_backend.py")
            return True
        except (OSError, ValueError):
            BACKEND_PID_FILE.unlink(missing_ok=True)
    return False


def start_backend(model_env: dict) -> bool:
    print_header("Starting backend server...")

    backend_log = open(BACKEND_LOG_FILE, "w")

    backend_process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=SCRIPT_DIR,
        stdout=backend_log,
        stderr=subprocess.STDOUT,
        env={
            **os.environ,
            "DB_HOST":     DB_HOST,
            "DB_PORT":     DB_PORT,
            "DB_NAME":     DB_NAME,
            "DB_USER":     DB_USER,
            "DB_PASSWORD": DB_PASSWORD,
            **model_env,   # layered on top — empty dict is a no-op for --no-models
        },
    )

    BACKEND_PID_FILE.write_text(str(backend_process.pid))
    print_success(f"Backend server started (PID: {backend_process.pid})")
    print(f"  Running on port {BACKEND_PORT}")
    print(f"  Logs: tail -f {BACKEND_LOG_FILE}")

    time.sleep(2)

    if backend_process.poll() is not None:
        print_error("Backend server failed to start")
        print(f"  Check logs at: {BACKEND_LOG_FILE}")
        return False

    return True


def main():
    args = parse_args()

    print_colored(GREEN, "=== RAG Backend Startup ===\n")
    describe_mode(args)

    if check_if_running():
        sys.exit(1)

    if not check_database_connection():
        sys.exit(1)

    model_env = build_model_env(args)

    if not start_backend(model_env):
        sys.exit(1)

    print_colored(GREEN, "\n=== Backend Started! ===\n")
    print(f"  API Endpoint:      http://localhost:{BACKEND_PORT}")
    print(f"  API Documentation: http://localhost:{BACKEND_PORT}/docs")
    print(f"  Health Check:      http://localhost:{BACKEND_PORT}/health")
    print("\nTo stop the server, run: python3 stop_backend.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStartup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
