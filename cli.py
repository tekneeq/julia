#!/usr/bin/env python3
"""
Development CLI entry point for Julia Options Analysis

This script allows running the CLI during development without installing the package.
Usage: python cli.py greeks --ticker SPY --show-gex
"""

import sys
import os

# Add src to Python path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from julia.main import cli

if __name__ == "__main__":
    cli()