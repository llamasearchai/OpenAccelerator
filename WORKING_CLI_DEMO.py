#!/usr/bin/env python3
"""
OpenAccelerator CLI Working Demonstration

This script demonstrates the complete working CLI interface for OpenAccelerator.
It shows all the available commands and their functionality.

Author: Nik Jois <nikjois@llamasearch.ai>
Version: 1.0.0
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a CLI command and display results."""
    print(f"\n{'='*60}")
    print(f"[DEMO] {description}")
    print(f"[COMMAND] {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print(f"[STDERR] {result.stderr}")
        
        if result.returncode == 0:
            print(f"[SUCCESS] Command completed successfully")
            return True
        else:
            print(f"[ERROR] Command failed with exit code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Command timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run command: {e}")
        return False


def main():
    """Main demonstration function."""
    print("[DEMO] OpenAccelerator CLI Working Demonstration")
    print("=" * 60)
    print("This demo shows all the CLI features working correctly")
    print("Author: Nik Jois <nikjois@llamasearch.ai>")
    print("=" * 60)
    
    # Check if CLI script exists
    cli_script = Path("scripts/accelerator_cli.py")
    wrapper_script = Path("openaccel")
    
    if not cli_script.exists():
        print(f"[ERROR] CLI script not found: {cli_script}")
        return False
    
    if not wrapper_script.exists():
        print(f"[ERROR] Wrapper script not found: {wrapper_script}")
        return False
    
    # List of commands to demonstrate
    commands = [
        {
            "cmd": ["python", "scripts/accelerator_cli.py", "--help"],
            "desc": "Show CLI help and available commands"
        },
        {
            "cmd": ["./openaccel", "status"],
            "desc": "Check system status and server connectivity"
        },
        {
            "cmd": ["./openaccel", "simulate", "gemm", "--local", "-M", "4", "-K", "4", "-P", "4"],
            "desc": "Run a small GEMM simulation locally"
        },
        {
            "cmd": ["./openaccel", "simulate", "gemm", "--local", "-M", "8", "-K", "8", "-P", "8"],
            "desc": "Run a medium GEMM simulation locally"
        },
        {
            "cmd": ["./openaccel", "medical"],
            "desc": "Run medical compliance validation"
        },
        {
            "cmd": ["./openaccel", "agents"],
            "desc": "Demonstrate AI agents functionality"
        },
        {
            "cmd": ["./openaccel", "benchmark"],
            "desc": "Run comprehensive benchmark suite"
        },
        {
            "cmd": ["./openaccel", "test"],
            "desc": "Run test suite validation"
        }
    ]
    
    # Track results
    results = []
    
    print(f"\n[DEMO] Running {len(commands)} CLI demonstrations...")
    
    for i, command_info in enumerate(commands, 1):
        print(f"\n[DEMO] Step {i}/{len(commands)}")
        
        success = run_command(command_info["cmd"], command_info["desc"])
        results.append({
            "command": " ".join(command_info["cmd"]),
            "description": command_info["desc"],
            "success": success
        })
        
        # Small delay between commands
        time.sleep(1)
    
    # Summary
    print(f"\n{'='*60}")
    print("[DEMO] CLI Demonstration Summary")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"[RESULTS] Commands executed: {total}")
    print(f"[RESULTS] Successful: {successful}")
    print(f"[RESULTS] Failed: {total - successful}")
    print(f"[RESULTS] Success rate: {successful/total*100:.1f}%")
    
    print(f"\n[DETAILED RESULTS]")
    print("-" * 60)
    for i, result in enumerate(results, 1):
        status = "[SUCCESS]" if result["success"] else "[FAILED]"
        print(f"{i:2d}. {status} {result['description']}")
    
    # Overall assessment
    if successful == total:
        print(f"\n[SUCCESS] All CLI commands working perfectly!")
        print("[INFO] The OpenAccelerator CLI is fully functional and ready for use")
        return True
    elif successful >= total * 0.8:
        print(f"\n[WARNING] Most CLI commands working ({successful}/{total})")
        print("[INFO] The OpenAccelerator CLI is mostly functional")
        return True
    else:
        print(f"\n[ERROR] Many CLI commands failed ({total-successful}/{total})")
        print("[INFO] The OpenAccelerator CLI needs troubleshooting")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Demo cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        sys.exit(1) 