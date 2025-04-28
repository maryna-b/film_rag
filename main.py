"""
Main entry point for the Film RAG system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cli import run_cli

if __name__ == "__main__":
    run_cli()
