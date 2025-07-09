#!/usr/bin/env python3
"""
Standalone REPL launcher for CodeGen CLI
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.repl import start_repl

if __name__ == '__main__':
    start_repl()
