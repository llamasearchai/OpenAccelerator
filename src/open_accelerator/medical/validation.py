"""
Medical validation module for Open Accelerator.

Provides comprehensive validation capabilities for medical AI systems including
clinical trial validation, regulatory validation, and safety validation.
"""

import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ValidationType(Enum):
    """Types of medical validation."""
    CLINICAL = "clinical"
    REGULATORY = "regulatory"
    SAFETY = "safety"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    STATISTICAL = "statistical"


class ValidationStatus(Enum):
    """Validation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    CONDITIONAL = "conditional"


@dataclass
class ValidationCriteria:
    """Validation criteria definition."""
    criteria_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    validation_type: ValidationType = ValidationType.CLINICAL
    parameter: str = ""
    expected_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    tolerance: float = 0.05
    required: bool = True
    weight: float = 1.0


@dataclass
class ValidationResult:
    """Validation result."""
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    criteria_id: str = ""
    status: ValidationStatus = ValidationStatus.PENDING
    measured_value: Optional[float] = None
    expected_value: Optional[float] = None
    deviation: Optional[float] = None
    passed: bool = False
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


class MedicalValidator:
    """
    Comprehensive medical validation system.
    
    Provides validation capabilities for medical AI systems with support for
    multiple validation types and regulatory requirements.
    """
    
    def __init__(self, validation_config: Optional[Dict[str, Any]] = None):
        """Initialize medical validator."""
        self.config = validation_config or {}
        self.validation_criteria: List[ValidationCriteria] = []
        self.validation_results: List[ValidationResult] = []
        self.validation_history: List[Dict[str, Any]] = []
        
        # Initialize default validation criteria
        self._initialize_default_criteria()
        
        logger.info("Medical validator initialized")
    
    def _initialize_default_criteria(self):
        """Initialize default validation criteria."""
        default_criteria = [
            ValidationCriteria(
                name="accuracy_validation",
                description="Validate model accuracy meets medical standards",
                validation_type=ValidationType.PERFORMANCE,
                parameter="accuracy",
                min_value=0.95,
                required=True,
                weight=1.0
            ),
            ValidationCriteria(
                name="sensitivity_validation",
                description="Validate model sensitivity for medical diagnosis",
                validation_type=ValidationType.CLINICAL,
                parameter="sensitivity",
                min_value=0.90,
                required=True,
                weight=1.0
            ),
            ValidationCriteria(
                name="specificity_validation",
                description="Validate model specificity for medical diagnosis",
                validation_type=ValidationType.CLINICAL,
                parameter="specificity",
                min_value=0.85,
                required=True,
                weight=1.0
            ),
            ValidationCriteria(
                name="reliability_validation",
                description="Validate system reliability",
                validation_type=ValidationType.SAFETY,
                parameter="reliability",
                min_value=0.999,
                required=True,
                weight=1.0
            )
        ]
        
        self.validation_criteria.extend(default_criteria)
    
    def add_validation_criteria(self, criteria: ValidationCriteria):
        """Add custom validation criteria."""
        self.validation_criteria.append(criteria)
        logger.info(f"Added validation criteria: {criteria.name}")
    
    def validate_system(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate medical system against all criteria.
        
        Args:
            system_data: System data to validate
            
        Returns:
            Validation results
        """
        validation_start = datetime.now()
        self.validation_results = []
        
        try:
            # Validate against each criteria
            for criteria in self.validation_criteria:
                result = self._validate_against_criteria(criteria, system_data)
                self.validation_results.append(result)
            
            # Generate validation summary
            validation_summary = self._generate_validation_summary()
            
            # Log validation
            self._log_validation(validation_start, validation_summary)
            
            logger.info("System validation completed")
            return validation_summary
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise
    
    def _validate_against_criteria(self, criteria: ValidationCriteria, 
                                 system_data: Dict[str, Any]) -> ValidationResult:
        """Validate system data against specific criteria."""
        result = ValidationResult(
            criteria_id=criteria.criteria_id,
            expected_value=criteria.expected_value
        )
        
        # Extract measured value
        measured_value = system_data.get(criteria.parameter)
        result.measured_value = measured_value
        
        if measured_value is None:
            result.status = ValidationStatus.FAILED
            result.message = f"Parameter '{criteria.parameter}' not found in system data"
            result.passed = False
            return result
        
        # Validate against criteria
        validation_passed = True
        deviation = 0.0
        
        if criteria.min_value is not None:
            if measured_value < criteria.min_value:
                validation_passed = False
                deviation = criteria.min_value - measured_value
        
        if criteria.max_value is not None:
            if measured_value > criteria.max_value:
                validation_passed = False
                deviation = measured_value - criteria.max_value
        
        if criteria.expected_value is not None:
            value_deviation = abs(measured_value - criteria.expected_value)
            if value_deviation > criteria.tolerance:
                validation_passed = False
                deviation = value_deviation
        
        # Set result status
        result.passed = validation_passed
        result.deviation = deviation
        
        if validation_passed:
            result.status = ValidationStatus.PASSED
            result.message = f"Validation passed for {criteria.name}"
        else:
            result.status = ValidationStatus.FAILED if criteria.required else ValidationStatus.WARNING
            result.message = f"Validation failed for {criteria.name}: deviation {deviation:.4f}"
        
        result.details = {
            "criteria_name": criteria.name,
            "validation_type": criteria.validation_type.value,
            "required": criteria.required,
            "tolerance": criteria.tolerance
        }
        
        return result
    
    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        total_criteria = len(self.validation_criteria)
        passed_criteria = len([r for r in self.validation_results if r.passed])
        failed_criteria = len([r for r in self.validation_results if not r.passed and r.status == ValidationStatus.FAILED])
        warning_criteria = len([r for r in self.validation_results if not r.passed and r.status == ValidationStatus.WARNING])
        
        overall_passed = failed_criteria == 0
        
        summary = {
            "overall_status": ValidationStatus.PASSED if overall_passed else ValidationStatus.FAILED,
            "overall_passed": overall_passed,
            "total_criteria": total_criteria,
            "passed_criteria": passed_criteria,
            "failed_criteria": failed_criteria,
            "warning_criteria": warning_criteria,
            "pass_rate": passed_criteria / total_criteria if total_criteria > 0 else 0.0,
            "validation_results": [
                {
                    "criteria_id": r.criteria_id,
                    "status": r.status.value,
                    "passed": r.passed,
                    "measured_value": r.measured_value,
                    "expected_value": r.expected_value,
                    "deviation": r.deviation,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.validation_results
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    def _log_validation(self, start_time: datetime, summary: Dict[str, Any]):
        """Log validation results."""
        validation_log = {
            "timestamp": start_time.isoformat(),
            "duration": (datetime.now() - start_time).total_seconds(),
            "overall_status": summary["overall_status"].value,
            "pass_rate": summary["pass_rate"],
            "total_criteria": summary["total_criteria"],
            "passed_criteria": summary["passed_criteria"],
            "failed_criteria": summary["failed_criteria"]
        }
        
        self.validation_history.append(validation_log)


class ClinicalTrialValidator:
    """Validator for clinical trial compliance and data quality."""
    
    def __init__(self):
        """Initialize clinical trial validator."""
        self.trial_requirements = {
            "sample_size_adequacy": True,
            "randomization_quality": True,
            "blinding_integrity": True,
            "protocol_adherence": True,
            "data_quality": True,
            "statistical_power": True
        }
        
        logger.info("Clinical trial validator initialized")
    
    def validate_trial_design(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate clinical trial design."""
        validation_results = {
            "trial_id": trial_data.get("trial_id", "unknown"),
            "validation_passed": True,
            "validation_details": {},
            "recommendations": []
        }
        
        # Validate sample size
        sample_size = trial_data.get("sample_size", 0)
        required_sample_size = trial_data.get("required_sample_size", 100)
        
        if sample_size < required_sample_size:
            validation_results["validation_passed"] = False
            validation_results["validation_details"]["sample_size"] = "Insufficient sample size"
            validation_results["recommendations"].append("Increase sample size")
        else:
            validation_results["validation_details"]["sample_size"] = "Adequate sample size"
        
        # Validate randomization
        randomization_method = trial_data.get("randomization_method", "none")
        if randomization_method == "none":
            validation_results["validation_passed"] = False
            validation_results["validation_details"]["randomization"] = "No randomization method specified"
            validation_results["recommendations"].append("Implement proper randomization")
        else:
            validation_results["validation_details"]["randomization"] = f"Using {randomization_method}"
        
        # Validate statistical power
        statistical_power = trial_data.get("statistical_power", 0.0)
        if statistical_power < 0.8:
            validation_results["validation_passed"] = False
            validation_results["validation_details"]["statistical_power"] = "Insufficient statistical power"
            validation_results["recommendations"].append("Increase statistical power")
        else:
            validation_results["validation_details"]["statistical_power"] = "Adequate statistical power"
        
        logger.info(f"Clinical trial validation completed: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
        return validation_results
    
    def validate_trial_data(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate clinical trial data quality."""
        validation_results = {
            "data_quality_score": 0.0,
            "missing_data_rate": 0.0,
            "data_consistency": True,
            "outlier_detection": {},
            "quality_passed": True
        }
        
        # Simulate data quality validation
        data_points = trial_data.get("data_points", [])
        if data_points:
            # Calculate missing data rate
            total_expected = len(data_points) * 10  # Assuming 10 fields per data point
            total_missing = sum(1 for dp in data_points for field in dp.values() if field is None)
            missing_rate = total_missing / total_expected if total_expected > 0 else 0
            
            validation_results["missing_data_rate"] = missing_rate
            
            # Calculate quality score
            quality_score = max(0, 1.0 - missing_rate)
            validation_results["data_quality_score"] = quality_score
            
            # Check if quality threshold is met
            if quality_score < 0.9 or missing_rate > 0.1:
                validation_results["quality_passed"] = False
        
        logger.info(f"Trial data validation completed: quality score {validation_results['data_quality_score']:.3f}")
        return validation_results


class RegulatoryValidator:
    """Validator for regulatory compliance (FDA, CE, etc.)."""
    
    def __init__(self, regulatory_framework: str = "FDA"):
        """Initialize regulatory validator."""
        self.framework = regulatory_framework
        self.compliance_requirements = self._get_compliance_requirements(regulatory_framework)
        
        logger.info(f"Regulatory validator initialized for {regulatory_framework}")
    
    def _get_compliance_requirements(self, framework: str) -> Dict[str, Any]:
        """Get compliance requirements for regulatory framework."""
        if framework == "FDA":
            return {
                "device_classification": "required",
                "predicate_device": "optional",
                "clinical_data": "required",
                "risk_analysis": "required",
                "software_documentation": "required",
                "cybersecurity": "required",
                "quality_system": "required"
            }
        elif framework == "CE":
            return {
                "conformity_assessment": "required",
                "technical_documentation": "required",
                "clinical_evaluation": "required",
                "risk_management": "required",
                "post_market_surveillance": "required"
            }
        else:
            return {}
    
    def validate_regulatory_compliance(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate regulatory compliance."""
        compliance_results = {
            "framework": self.framework,
            "overall_compliance": True,
            "compliance_details": {},
            "missing_requirements": [],
            "compliance_score": 0.0
        }
        
        total_requirements = len(self.compliance_requirements)
        met_requirements = 0
        
        for requirement, status in self.compliance_requirements.items():
            requirement_data = system_data.get(requirement)
            
            if status == "required":
                if requirement_data is None or requirement_data == "":
                    compliance_results["overall_compliance"] = False
                    compliance_results["missing_requirements"].append(requirement)
                    compliance_results["compliance_details"][requirement] = "MISSING"
                else:
                    met_requirements += 1
                    compliance_results["compliance_details"][requirement] = "COMPLIANT"
            elif status == "optional":
                if requirement_data:
                    met_requirements += 1
                    compliance_results["compliance_details"][requirement] = "PROVIDED"
                else:
                    compliance_results["compliance_details"][requirement] = "NOT_PROVIDED"
        
        # Calculate compliance score
        compliance_results["compliance_score"] = met_requirements / total_requirements if total_requirements > 0 else 0
        
        logger.info(f"Regulatory compliance validation completed: {compliance_results['compliance_score']:.3f}")
        return compliance_results


class SafetyValidator:
    """Validator for medical safety requirements."""
    
    def __init__(self):
        """Initialize safety validator."""
        self.safety_standards = {
            "IEC 62304": "medical_device_software",
            "ISO 14971": "risk_management",
            "IEC 62366": "usability_engineering",
            "ISO 27799": "health_informatics_security"
        }
        
        logger.info("Safety validator initialized")
    
    def validate_safety_requirements(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate safety requirements."""
        safety_results = {
            "overall_safety": True,
            "safety_score": 0.0,
            "standard_compliance": {},
            "risk_assessment": {},
            "safety_recommendations": []
        }
        
        # Validate against safety standards
        compliant_standards = 0
        total_standards = len(self.safety_standards)
        
        for standard, description in self.safety_standards.items():
            compliance_data = system_data.get(f"{standard}_compliance", False)
            
            if compliance_data:
                safety_results["standard_compliance"][standard] = "COMPLIANT"
                compliant_standards += 1
            else:
                safety_results["standard_compliance"][standard] = "NON_COMPLIANT"
                safety_results["overall_safety"] = False
                safety_results["safety_recommendations"].append(f"Ensure compliance with {standard}")
        
        # Calculate safety score
        safety_results["safety_score"] = compliant_standards / total_standards if total_standards > 0 else 0
        
        # Perform risk assessment
        safety_results["risk_assessment"] = self._perform_risk_assessment(system_data)
        
        logger.info(f"Safety validation completed: {safety_results['safety_score']:.3f}")
        return safety_results
    
    def _perform_risk_assessment(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform basic risk assessment."""
        risk_assessment = {
            "overall_risk": "low",
            "identified_risks": [],
            "mitigation_measures": [],
            "residual_risk": "acceptable"
        }
        
        # Check for high-risk indicators
        accuracy = system_data.get("accuracy", 1.0)
        reliability = system_data.get("reliability", 1.0)
        
        if accuracy < 0.95:
            risk_assessment["identified_risks"].append("Low accuracy risk")
            risk_assessment["mitigation_measures"].append("Improve model accuracy")
            risk_assessment["overall_risk"] = "medium"
        
        if reliability < 0.999:
            risk_assessment["identified_risks"].append("Reliability risk")
            risk_assessment["mitigation_measures"].append("Implement redundancy")
            risk_assessment["overall_risk"] = "medium"
        
        return risk_assessment


def create_validation_pipeline(validation_types: List[str]) -> Dict[str, Any]:
    """
    Create a comprehensive validation pipeline.
    
    Args:
        validation_types: List of validation types to include
        
    Returns:
        Configured validation pipeline
    """
    pipeline = {
        "validators": {},
        "validation_sequence": validation_types,
        "pipeline_config": {
            "stop_on_failure": True,
            "generate_report": True,
            "log_results": True
        }
    }
    
    # Initialize validators based on types
    for validation_type in validation_types:
        if validation_type == "medical":
            pipeline["validators"]["medical"] = MedicalValidator()
        elif validation_type == "clinical_trial":
            pipeline["validators"]["clinical_trial"] = ClinicalTrialValidator()
        elif validation_type == "regulatory":
            pipeline["validators"]["regulatory"] = RegulatoryValidator()
        elif validation_type == "safety":
            pipeline["validators"]["safety"] = SafetyValidator()
    
    logger.info(f"Created validation pipeline with {len(validation_types)} validators")
    return pipeline 