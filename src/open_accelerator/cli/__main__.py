#!/usr/bin/env python3
"""
Command line interface main entry point for Open Accelerator.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import sys
from typing import Optional

try:
    from . import main as cli_main
except ImportError:
    # Fallback for development
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from open_accelerator.cli import main as cli_main


def main(args: Optional[list] = None) -> int:
    """Main entry point for CLI."""
    try:
        cli_main()
        return 0
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
