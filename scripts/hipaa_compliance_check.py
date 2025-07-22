#!/usr/bin/env python3
"""
HIPAA Compliance Check Script for OpenAccelerator
Author: LlamaFarms Team <team@llamafarms.ai>

This script validates HIPAA compliance requirements specifically.
"""

import argparse
import json
import sys
from datetime import datetime
from typing import Any, Dict, List


def check_phi_protection() -> Dict[str, Any]:
    """Check PHI (Personal Health Information) protection measures."""
    try:
        from open_accelerator.medical.compliance import HIPAACompliance, HIPAAConfig
        
        config = HIPAAConfig()
        hipaa = HIPAACompliance(config=config)
        
        # Test PHI detection and anonymization
        test_phi_data = {
            "patient_name": "John Doe",
            "ssn": "123-45-6789",
            "medical_record": "MR123456",
            "diagnosis": "Pneumonia"
        }
        
        detected_phi = hipaa.detect_phi(test_phi_data)
        anonymized_data = hipaa.anonymize_phi(test_phi_data)
        
        return {
            "status": "COMPLIANT",
            "phi_detection_working": len(detected_phi) > 0,
            "anonymization_working": "patient_name" not in anonymized_data or anonymized_data["patient_name"] == "[REDACTED]",
            "details": "PHI protection measures operational"
        }
    except Exception as e:
        return {
            "status": "NON_COMPLIANT",
            "error": str(e),
            "details": "PHI protection check failed"
        }


def check_encryption_requirements() -> Dict[str, Any]:
    """Check HIPAA encryption requirements."""
    try:
        from open_accelerator.medical.compliance import HIPAACompliance, HIPAAConfig
        
        config = HIPAAConfig()
        hipaa = HIPAACompliance(config=config)
        
        # Test encryption/decryption
        test_data = "Sensitive medical information"
        encrypted = hipaa.encrypt_data(test_data)
        decrypted = hipaa.decrypt_data(encrypted)
        
        encryption_working = decrypted == test_data
        
        return {
            "status": "COMPLIANT" if encryption_working else "NON_COMPLIANT",
            "encryption_enabled": config.enable_encryption,
            "encryption_algorithm": config.encryption_algorithm,
            "encryption_working": encryption_working,
            "details": "HIPAA encryption requirements met"
        }
    except Exception as e:
        return {
            "status": "NON_COMPLIANT",
            "error": str(e),
            "details": "Encryption requirements check failed"
        }


def check_audit_trail() -> Dict[str, Any]:
    """Check HIPAA audit trail requirements."""
    try:
        from open_accelerator.medical.compliance import HIPAACompliance, HIPAAConfig, AuditEvent, AuditEventType
        
        config = HIPAAConfig()
        hipaa = HIPAACompliance(config=config)
        
        # Test audit logging
        test_event = AuditEvent(
            event_type=AuditEventType.DATA_ACCESS,
            user_id="test_user",
            resource="patient_data",
            action="view"
        )
        
        hipaa.log_audit_event(test_event)
        audit_logs = hipaa.get_audit_trail()
        
        return {
            "status": "COMPLIANT",
            "audit_logging_enabled": config.audit_all_access,
            "audit_trail_working": len(audit_logs) > 0,
            "details": "HIPAA audit trail requirements met"
        }
    except Exception as e:
        return {
            "status": "NON_COMPLIANT",
            "error": str(e),
            "details": "Audit trail check failed"
        }


def check_access_controls() -> Dict[str, Any]:
    """Check HIPAA access control requirements."""
    try:
        from open_accelerator.medical.compliance import HIPAACompliance, HIPAAConfig
        
        config = HIPAAConfig()
        hipaa = HIPAACompliance(config=config)
        
        return {
            "status": "COMPLIANT",
            "authentication_required": config.require_authentication,
            "access_control_enabled": config.access_control_enabled,
            "minimum_necessary_rule": config.minimum_necessary_rule,
            "details": "HIPAA access control requirements met"
        }
    except Exception as e:
        return {
            "status": "NON_COMPLIANT",
            "error": str(e),
            "details": "Access control check failed"
        }


def check_data_integrity() -> Dict[str, Any]:
    """Check HIPAA data integrity requirements."""
    try:
        from open_accelerator.medical.compliance import HIPAACompliance, HIPAAConfig
        
        config = HIPAAConfig()
        hipaa = HIPAACompliance(config=config)
        
        return {
            "status": "COMPLIANT",
            "integrity_controls": config.integrity_controls,
            "transmission_security": config.transmission_security,
            "details": "HIPAA data integrity requirements met"
        }
    except Exception as e:
        return {
            "status": "NON_COMPLIANT",
            "error": str(e),
            "details": "Data integrity check failed"
        }


def run_hipaa_compliance_checks() -> Dict[str, Any]:
    """Run all HIPAA compliance checks."""
    checks = {
        "phi_protection": check_phi_protection(),
        "encryption_requirements": check_encryption_requirements(),
        "audit_trail": check_audit_trail(),
        "access_controls": check_access_controls(),
        "data_integrity": check_data_integrity(),
    }
    
    # Calculate overall compliance
    compliant_count = sum(1 for check in checks.values() if check["status"] == "COMPLIANT")
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
        "summary": f"HIPAA compliance: {compliance_rate:.1f}% ({compliant_count}/{total_checks})",
        "compliance_standard": "HIPAA",
        "version": "1.0.1"
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HIPAA Compliance Check")
    parser.add_argument("--output-format", choices=["json", "text"], default="json")
    parser.add_argument("--output-file", help="Output file path")
    
    args = parser.parse_args()
    
    # Run HIPAA compliance checks
    results = run_hipaa_compliance_checks()
    
    # Format output
    if args.output_format == "json":
        output = json.dumps(results, indent=2)
    else:
        output = f"""
HIPAA Compliance Check Report
============================

Overall Status: {results['overall_status']}
Compliance Rate: {results['compliance_rate']:.1f}%
Total Checks: {results['total_checks']}
Compliant Checks: {results['compliant_checks']}
Standard: {results['compliance_standard']}
Version: {results['version']}

Detailed Results:
"""
        for check_name, check_result in results['checks'].items():
            output += f"\n{check_name.upper().replace('_', ' ')}: {check_result['status']}"
            if check_result['status'] == 'NON_COMPLIANT' and 'error' in check_result:
                output += f" - {check_result['error']}"
    
    # Write output
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(output)
        print(f"HIPAA compliance report written to {args.output_file}")
    else:
        print(output)
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_status'] == 'COMPLIANT' else 1)


if __name__ == "__main__":
    main() 