#!/usr/bin/env python3
"""
Comprehensive Build and Deployment Script for OpenAccelerator
Author: LlamaFarms Team <team@llamafarms.ai>

This script handles building, testing, and deploying the OpenAccelerator system.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class BuildDeployManager:
    """Manages the complete build and deployment process."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = time.time()
        self.build_log = []
        self.project_root = Path.cwd()

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.build_log.append(log_entry)
        
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(log_entry)

    def run_command(self, command: str, description: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command with logging."""
        self.log(f"Running: {description}")
        self.log(f"Command: {command}")
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                check=check
            )
            
            if result.stdout and self.verbose:
                self.log(f"Output: {result.stdout.strip()}")
            
            if result.stderr and result.returncode != 0:
                self.log(f"Error: {result.stderr.strip()}", "ERROR")
            
            return result
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {e}", "ERROR")
            if e.stdout:
                self.log(f"Stdout: {e.stdout}", "ERROR")
            if e.stderr:
                self.log(f"Stderr: {e.stderr}", "ERROR")
            raise

    def check_prerequisites(self) -> bool:
        """Check that all prerequisites are met."""
        self.log("Checking prerequisites...")
        
        checks = [
            ("python", "Python interpreter"),
            ("pip", "Python package manager"),
            ("git", "Git version control"),
        ]
        
        all_good = True
        for command, description in checks:
            try:
                result = self.run_command(f"which {command}", f"Check {description}", check=False)
                if result.returncode == 0:
                    self.log(f"✓ {description} found")
                else:
                    self.log(f"✗ {description} not found", "ERROR")
                    all_good = False
            except Exception as e:
                self.log(f"✗ {description} check failed: {e}", "ERROR")
                all_good = False
        
        return all_good

    def clean_build_artifacts(self):
        """Clean previous build artifacts."""
        self.log("Cleaning build artifacts...")
        
        cleanup_paths = [
            "dist/",
            "build/",
            "src/open_accelerator.egg-info/",
            "htmlcov/",
            ".coverage",
            ".pytest_cache/",
            "__pycache__/",
        ]
        
        for path in cleanup_paths:
            full_path = self.project_root / path
            if full_path.exists():
                if full_path.is_dir():
                    self.run_command(f"rm -rf {full_path}", f"Remove directory {path}")
                else:
                    self.run_command(f"rm -f {full_path}", f"Remove file {path}")

    def run_linting(self) -> bool:
        """Run code linting and formatting checks."""
        self.log("Running linting and formatting checks...")
        
        lint_commands = [
            ("ruff check src/ tests/", "Ruff linting"),
            ("black --check src/ tests/", "Black formatting check"),
            ("isort --check-only src/ tests/", "Import sorting check"),
        ]
        
        all_passed = True
        for command, description in lint_commands:
            try:
                self.run_command(command, description)
                self.log(f"✓ {description} passed")
            except subprocess.CalledProcessError:
                self.log(f"✗ {description} failed", "WARNING")
                all_passed = False
        
        return all_passed

    def run_type_checking(self) -> bool:
        """Run type checking."""
        self.log("Running type checking...")
        
        try:
            self.run_command("mypy src/open_accelerator/", "Type checking with mypy")
            self.log("✓ Type checking passed")
            return True
        except subprocess.CalledProcessError:
            self.log("✗ Type checking failed", "WARNING")
            return False

    def run_security_checks(self) -> bool:
        """Run security checks."""
        self.log("Running security checks...")
        
        security_commands = [
            ("bandit -r src/open_accelerator/", "Security analysis with Bandit"),
            ("safety check", "Dependency security check"),
        ]
        
        all_passed = True
        for command, description in security_commands:
            try:
                self.run_command(command, description)
                self.log(f"✓ {description} passed")
            except subprocess.CalledProcessError:
                self.log(f"✗ {description} failed", "WARNING")
                all_passed = False
        
        return all_passed

    def run_tests(self, coverage: bool = True) -> bool:
        """Run the test suite."""
        self.log("Running test suite...")
        
        if coverage:
            test_command = "python -m pytest tests/ --cov=open_accelerator --cov-report=html --cov-report=term-missing"
        else:
            test_command = "python -m pytest tests/ -v"
        
        try:
            self.run_command(test_command, "Test suite execution")
            self.log("✓ All tests passed")
            return True
        except subprocess.CalledProcessError:
            self.log("✗ Test suite failed", "ERROR")
            return False

    def run_medical_compliance_checks(self) -> bool:
        """Run medical compliance validation."""
        self.log("Running medical compliance checks...")
        
        compliance_scripts = [
            ("python scripts/medical_compliance_check.py --output-format json --output-file medical_compliance.json", "Medical compliance check"),
            ("python scripts/hipaa_compliance_check.py --output-format json --output-file hipaa_compliance.json", "HIPAA compliance check"),
            ("python scripts/fda_validation_check.py --output-format json --output-file fda_validation.json", "FDA validation check"),
        ]
        
        all_passed = True
        for command, description in compliance_scripts:
            try:
                self.run_command(command, description)
                self.log(f"✓ {description} passed")
            except subprocess.CalledProcessError:
                self.log(f"✗ {description} failed", "WARNING")
                all_passed = False
        
        return all_passed

    def build_package(self) -> bool:
        """Build the Python package."""
        self.log("Building Python package...")
        
        try:
            self.run_command("python -m build", "Package building")
            self.log("✓ Package built successfully")
            
            # Verify build artifacts
            dist_dir = self.project_root / "dist"
            if not dist_dir.exists():
                self.log("✗ Distribution directory not found", "ERROR")
                return False
            
            wheel_files = list(dist_dir.glob("*.whl"))
            tar_files = list(dist_dir.glob("*.tar.gz"))
            
            if not wheel_files:
                self.log("✗ Wheel file not found", "ERROR")
                return False
            
            if not tar_files:
                self.log("✗ Source distribution not found", "ERROR")
                return False
            
            self.log(f"✓ Built wheel: {wheel_files[0].name}")
            self.log(f"✓ Built source dist: {tar_files[0].name}")
            
            return True
        except subprocess.CalledProcessError:
            self.log("✗ Package build failed", "ERROR")
            return False

    def validate_package(self) -> bool:
        """Validate the built package."""
        self.log("Validating built package...")
        
        try:
            # Check package with twine
            self.run_command("twine check dist/*", "Package validation with twine")
            self.log("✓ Package validation passed")
            return True
        except subprocess.CalledProcessError:
            self.log("✗ Package validation failed", "ERROR")
            return False

    def build_documentation(self) -> bool:
        """Build documentation."""
        self.log("Building documentation...")
        
        try:
            docs_dir = self.project_root / "docs"
            if docs_dir.exists():
                self.run_command("cd docs && make clean && make html", "Documentation building")
                self.log("✓ Documentation built successfully")
                return True
            else:
                self.log("Documentation directory not found", "WARNING")
                return False
        except subprocess.CalledProcessError:
            self.log("✗ Documentation build failed", "ERROR")
            return False

    def run_system_validation(self) -> bool:
        """Run comprehensive system validation."""
        self.log("Running comprehensive system validation...")
        
        try:
            self.run_command("python FINAL_SYSTEM_VALIDATION.py", "System validation")
            self.log("✓ System validation passed")
            return True
        except subprocess.CalledProcessError:
            self.log("✗ System validation failed", "ERROR")
            return False

    def generate_build_report(self) -> Dict[str, Any]:
        """Generate a comprehensive build report."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "project": "OpenAccelerator",
            "version": "1.0.1",
            "author": "LlamaFarms Team <team@llamafarms.ai>",
            "build_duration_seconds": duration,
            "build_log": self.build_log,
            "artifacts": {
                "package": list((self.project_root / "dist").glob("*")) if (self.project_root / "dist").exists() else [],
                "documentation": (self.project_root / "docs" / "_build" / "html").exists(),
                "coverage_report": (self.project_root / "htmlcov").exists(),
            }
        }
        
        # Save report
        report_file = self.project_root / "build_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log(f"Build report saved to {report_file}")
        return report

    def deploy_to_test_pypi(self) -> bool:
        """Deploy to Test PyPI."""
        self.log("Deploying to Test PyPI...")
        
        try:
            self.run_command(
                "twine upload --repository testpypi dist/*",
                "Upload to Test PyPI"
            )
            self.log("✓ Successfully deployed to Test PyPI")
            return True
        except subprocess.CalledProcessError:
            self.log("✗ Test PyPI deployment failed", "ERROR")
            return False

    def deploy_to_pypi(self) -> bool:
        """Deploy to PyPI."""
        self.log("Deploying to PyPI...")
        
        try:
            self.run_command("twine upload dist/*", "Upload to PyPI")
            self.log("✓ Successfully deployed to PyPI")
            return True
        except subprocess.CalledProcessError:
            self.log("✗ PyPI deployment failed", "ERROR")
            return False

    def full_build_pipeline(self, skip_tests: bool = False, skip_docs: bool = False) -> bool:
        """Run the complete build pipeline."""
        self.log("Starting complete build pipeline...")
        
        pipeline_steps = [
            ("Prerequisites", self.check_prerequisites),
            ("Clean artifacts", lambda: self.clean_build_artifacts() or True),
            ("Linting", self.run_linting),
            ("Type checking", self.run_type_checking),
            ("Security checks", self.run_security_checks),
        ]
        
        if not skip_tests:
            pipeline_steps.extend([
                ("Test suite", self.run_tests),
                ("Medical compliance", self.run_medical_compliance_checks),
                ("System validation", self.run_system_validation),
            ])
        
        pipeline_steps.extend([
            ("Package build", self.build_package),
            ("Package validation", self.validate_package),
        ])
        
        if not skip_docs:
            pipeline_steps.append(("Documentation", self.build_documentation))
        
        # Run pipeline
        failed_steps = []
        for step_name, step_func in pipeline_steps:
            self.log(f"Pipeline step: {step_name}")
            try:
                if not step_func():
                    failed_steps.append(step_name)
                    self.log(f"✗ Pipeline step failed: {step_name}", "ERROR")
            except Exception as e:
                failed_steps.append(step_name)
                self.log(f"✗ Pipeline step error: {step_name} - {e}", "ERROR")
        
        # Generate report
        report = self.generate_build_report()
        
        if failed_steps:
            self.log(f"✗ Build pipeline completed with {len(failed_steps)} failures", "ERROR")
            self.log(f"Failed steps: {', '.join(failed_steps)}", "ERROR")
            return False
        else:
            self.log("✓ Build pipeline completed successfully")
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build and Deploy OpenAccelerator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test execution")
    parser.add_argument("--skip-docs", action="store_true", help="Skip documentation build")
    parser.add_argument("--deploy-test", action="store_true", help="Deploy to Test PyPI")
    parser.add_argument("--deploy-prod", action="store_true", help="Deploy to PyPI")
    parser.add_argument("--clean-only", action="store_true", help="Only clean build artifacts")
    
    args = parser.parse_args()
    
    # Create build manager
    builder = BuildDeployManager(verbose=args.verbose)
    
    if args.clean_only:
        builder.clean_build_artifacts()
        print("Build artifacts cleaned.")
        return 0
    
    # Run build pipeline
    success = builder.full_build_pipeline(
        skip_tests=args.skip_tests,
        skip_docs=args.skip_docs
    )
    
    if not success:
        print("Build pipeline failed. Check build_report.json for details.")
        return 1
    
    # Handle deployment
    if args.deploy_test:
        if not builder.deploy_to_test_pypi():
            print("Test PyPI deployment failed.")
            return 1
    
    if args.deploy_prod:
        if not builder.deploy_to_pypi():
            print("PyPI deployment failed.")
            return 1
    
    print("Build and deployment completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 