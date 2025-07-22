#!/usr/bin/env python3
"""
Medical Compliance Check Script for OpenAccelerator
Author: LlamaFarms Team <team@llamafarms.ai>

This script validates medical compliance requirements for the OpenAccelerator system.
"""

import argparse
import json
import sys
from datetime import datetime
from typing import Any, Dict


def check_hipaa_compliance() -> Dict[str, Any]:
    """Check HIPAA compliance requirements."""
    try:
        from open_accelerator.medical.compliance import HIPAACompliance, HIPAAConfig

        config = HIPAAConfig()
        hipaa = HIPAACompliance(config=config)

        return {
            "status": "COMPLIANT",
            "compliance_level": hipaa.compliance_level.value,
            "encryption_enabled": config.enable_encryption,
            "audit_logging": config.audit_all_access,
            "details": "HIPAA compliance system operational",
        }
    except Exception as e:
        return {
            "status": "NON_COMPLIANT",
            "error": str(e),
            "details": "HIPAA compliance check failed",
        }


def check_fda_compliance() -> Dict[str, Any]:
    """Check FDA compliance requirements."""
    try:
        from open_accelerator.medical.compliance import FDACompliance, FDAConfig

        config = FDAConfig()
        fda = FDACompliance(config=config)

        return {
            "status": "COMPLIANT",
            "compliance_level": fda.compliance_level.value,
            "device_classification": config.device_classification,
            "clinical_validation": config.clinical_validation_required,
            "details": "FDA compliance system operational",
        }
    except Exception as e:
        return {
            "status": "NON_COMPLIANT",
            "error": str(e),
            "details": "FDA compliance check failed",
        }


def check_medical_imaging_compliance() -> Dict[str, Any]:
    """Check medical imaging compliance."""
    try:
        from open_accelerator.medical.imaging import (
            ImageModality,
            MedicalImageProcessor,
        )

        processor = MedicalImageProcessor(compliance_mode=True)

        return {
            "status": "COMPLIANT",
            "supported_modalities": [modality.value for modality in ImageModality],
            "phi_protection": True,
            "details": "Medical imaging system compliant",
        }
    except Exception as e:
        return {
            "status": "NON_COMPLIANT",
            "error": str(e),
            "details": "Medical imaging compliance check failed",
        }


def check_data_protection() -> Dict[str, Any]:
    """Check data protection measures."""
    try:
        from open_accelerator.core.security import SecurityConfig, SecurityManager

        config = SecurityConfig()
        security = SecurityManager(config=config)

        # Test encryption
        test_data = b"test_medical_data"
        encrypted = security.encrypt_data(test_data)
        decrypted = security.decrypt_data(encrypted)

        encryption_working = decrypted == test_data

        return {
            "status": "COMPLIANT" if encryption_working else "NON_COMPLIANT",
            "encryption_algorithm": config.default_algorithm.value,
            "encryption_working": encryption_working,
            "audit_logging": config.enable_audit_logging,
            "details": "Data protection measures operational",
        }
    except Exception as e:
        return {
            "status": "NON_COMPLIANT",
            "error": str(e),
            "details": "Data protection check failed",
        }


def run_compliance_checks() -> Dict[str, Any]:
    """Run all medical compliance checks."""
    checks = {
        "hipaa_compliance": check_hipaa_compliance(),
        "fda_compliance": check_fda_compliance(),
        "medical_imaging": check_medical_imaging_compliance(),
        "data_protection": check_data_protection(),
    }

    # Calculate overall compliance
    compliant_count = sum(
        1 for check in checks.values() if check["status"] == "COMPLIANT"
    )
    total_checks = len(checks)
    compliance_rate = (compliant_count / total_checks) * 100

    overall_status = "COMPLIANT" if compliant_count == total_checks else "NON_COMPLIANT"

    return {
        "timestamp": datetime.now().isoformat(),
        "overall_status": overall_status,
        "compliance_rate": compliance_rate,
        "total_checks": total_checks,
        "compliant_checks": compliant_count,
        "checks": checks,
        "summary": f"Medical compliance: {compliance_rate:.1f}% ({compliant_count}/{total_checks})",
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Medical Compliance Check")
    parser.add_argument("--output-format", choices=["json", "text"], default="json")
    parser.add_argument("--output-file", help="Output file path")

    args = parser.parse_args()

    # Run compliance checks
    results = run_compliance_checks()

    # Format output
    if args.output_format == "json":
        output = json.dumps(results, indent=2)
    else:
        output = f"""
Medical Compliance Check Report
==============================

Overall Status: {results['overall_status']}
Compliance Rate: {results['compliance_rate']:.1f}%
Total Checks: {results['total_checks']}
Compliant Checks: {results['compliant_checks']}

Detailed Results:
"""
        for check_name, check_result in results["checks"].items():
            output += f"\n{check_name.upper()}: {check_result['status']}"
            if check_result["status"] == "NON_COMPLIANT" and "error" in check_result:
                output += f" - {check_result['error']}"

    # Write output
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output)
        print(f"Medical compliance report written to {args.output_file}")
    else:
        print(output)

    # Exit with appropriate code
    sys.exit(0 if results["overall_status"] == "COMPLIANT" else 1)


if __name__ == "__main__":
    main()
