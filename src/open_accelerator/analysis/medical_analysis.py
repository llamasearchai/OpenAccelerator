"""
Medical-grade analysis and validation tools.

Provides specialized analysis for medical AI workloads with regulatory compliance,
safety validation, and precision verification.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ComplianceStandard(Enum):
    """Medical compliance standards."""

    FDA_510K = "fda_510k"
    FDA_DE_NOVO = "fda_de_novo"
    CE_MARKING = "ce_marking"
    ISO_13485 = "iso_13485"
    ISO_14155 = "iso_14155"
    IEC_62304 = "iec_62304"


class SafetyLevel(Enum):
    """Medical device safety classification."""

    CLASS_I = "class_i"  # Low risk
    CLASS_II = "class_ii"  # Moderate risk
    CLASS_III = "class_iii"  # High risk


@dataclass
class MedicalValidationConfig:
    """Configuration for medical validation."""

    compliance_standards: list[ComplianceStandard]
    safety_level: SafetyLevel
    precision_requirements: dict[str, float]
    error_tolerance: float
    validation_dataset_size: int
    cross_validation_folds: int = 5
    statistical_significance: float = 0.05


@dataclass
class MedicalMetrics:
    """Medical-specific performance metrics."""

    # Accuracy metrics
    sensitivity: float = 0.0  # True positive rate
    specificity: float = 0.0  # True negative rate
    precision: float = 0.0  # Positive predictive value
    recall: float = 0.0  # Same as sensitivity
    f1_score: float = 0.0
    auc_roc: float = 0.0  # Area under ROC curve
    auc_pr: float = 0.0  # Area under Precision-Recall curve

    # Medical-specific metrics
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    diagnostic_accuracy: float = 0.0
    clinical_agreement: float = 0.0  # Agreement with clinician

    # Safety metrics
    safety_violations: int = 0
    precision_violations: int = 0
    reliability_score: float = 0.0

    # Performance metrics
    inference_time_ms: float = 0.0
    energy_per_inference: float = 0.0
    throughput_fps: float = 0.0

    # Compliance metrics
    regulatory_compliance_score: float = 0.0
    validation_passed: bool = False


class MedicalAnalyzer:
    """
    Advanced medical AI analysis and validation.

    Provides comprehensive analysis tools for medical AI workloads including
    regulatory compliance checking, safety validation, and clinical metrics.
    """

    def __init__(self, validation_config: MedicalValidationConfig):
        """
        Initialize medical analyzer.

        Args:
            validation_config: Medical validation configuration
        """
        self.config = validation_config
        self.validation_results: dict[str, Any] = {}
        self.compliance_checklist: dict[str, bool] = {}
        self.safety_log: list[dict] = []

        # Initialize compliance checkers
        self._initialize_compliance_checkers()

        logger.info(
            f"Initialized medical analyzer for {validation_config.safety_level.value}"
        )

    def _initialize_compliance_checkers(self) -> None:
        """Initialize compliance checking systems."""
        for standard in self.config.compliance_standards:
            if standard == ComplianceStandard.FDA_510K:
                self.compliance_checklist.update(self._get_fda_510k_requirements())
            elif standard == ComplianceStandard.IEC_62304:
                self.compliance_checklist.update(self._get_iec_62304_requirements())
            elif standard == ComplianceStandard.ISO_13485:
                self.compliance_checklist.update(self._get_iso_13485_requirements())

    def _get_fda_510k_requirements(self) -> dict[str, bool]:
        """Get FDA 510(k) compliance requirements."""
        return {
            "substantial_equivalence_demonstrated": False,
            "predicate_device_identified": False,
            "safety_effectiveness_data": False,
            "clinical_validation_completed": False,
            "risk_analysis_documented": False,
            "software_lifecycle_process": False,
            "cybersecurity_documentation": False,
            "labeling_requirements_met": False,
        }

    def _get_iec_62304_requirements(self) -> dict[str, bool]:
        """Get IEC 62304 (Medical Device Software) requirements."""
        return {
            "software_safety_classification": False,
            "software_development_planning": False,
            "software_requirements_analysis": False,
            "software_architectural_design": False,
            "software_detailed_design": False,
            "software_implementation": False,
            "software_integration_testing": False,
            "software_system_testing": False,
            "software_release": False,
            "software_maintenance_process": False,
        }

    def _get_iso_13485_requirements(self) -> dict[str, bool]:
        """Get ISO 13485 (Quality Management) requirements."""
        return {
            "quality_management_system": False,
            "management_responsibility": False,
            "resource_management": False,
            "product_realization": False,
            "measurement_analysis_improvement": False,
            "design_controls": False,
            "risk_management": False,
            "corrective_preventive_action": False,
        }

    def analyze_medical_performance(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        execution_stats: dict[str, Any],
    ) -> MedicalMetrics:
        """
        Analyze medical AI performance with clinical metrics.

        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            execution_stats: Accelerator execution statistics

        Returns:
            Comprehensive medical metrics
        """
        # Calculate basic classification metrics
        metrics = self._calculate_classification_metrics(predictions, ground_truth)

        # Add performance metrics from execution stats
        metrics.inference_time_ms = execution_stats.get("total_time_ms", 0.0)
        metrics.energy_per_inference = execution_stats.get("total_energy_j", 0.0)
        metrics.throughput_fps = execution_stats.get("throughput_fps", 0.0)

        # Medical-specific validation
        metrics = self._validate_medical_requirements(
            metrics, predictions, ground_truth
        )

        # Safety assessment
        metrics = self._assess_safety_compliance(metrics, execution_stats)

        # Regulatory compliance scoring
        metrics.regulatory_compliance_score = self._calculate_compliance_score()

        logger.info(
            f"Medical analysis complete: Sensitivity={metrics.sensitivity:.3f}, "
            f"Specificity={metrics.specificity:.3f}, AUC={metrics.auc_roc:.3f}"
        )

        return metrics

    def _calculate_classification_metrics(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> MedicalMetrics:
        """Calculate standard classification metrics."""
        # Convert to binary if needed
        if predictions.ndim > 1:
            pred_binary = np.argmax(predictions, axis=1)
        else:
            pred_binary = (predictions > 0.5).astype(int)

        if ground_truth.ndim > 1:
            gt_binary = np.argmax(ground_truth, axis=1)
        else:
            gt_binary = ground_truth.astype(int)

        # Calculate confusion matrix components
        tp = np.sum((pred_binary == 1) & (gt_binary == 1))
        tn = np.sum((pred_binary == 0) & (gt_binary == 0))
        fp = np.sum((pred_binary == 1) & (gt_binary == 0))
        fn = np.sum((pred_binary == 0) & (gt_binary == 1))

        # Calculate metrics
        metrics = MedicalMetrics()

        # Sensitivity (Recall)
        metrics.sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics.recall = metrics.sensitivity

        # Specificity
        metrics.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Precision
        metrics.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # F1 Score
        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = (
                2
                * (metrics.precision * metrics.recall)
                / (metrics.precision + metrics.recall)
            )

        # False rates
        metrics.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics.false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Diagnostic accuracy
        metrics.diagnostic_accuracy = (
            (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        )

        # AUC-ROC (simplified)
        if predictions.ndim > 1:
            try:
                from sklearn.metrics import average_precision_score, roc_auc_score

                metrics.auc_roc = roc_auc_score(
                    gt_binary,
                    predictions[:, 1] if predictions.shape[1] > 1 else predictions,
                )
                metrics.auc_pr = average_precision_score(
                    gt_binary,
                    predictions[:, 1] if predictions.shape[1] > 1 else predictions,
                )
            except ImportError:
                logger.warning("sklearn not available for AUC calculation")
                metrics.auc_roc = 0.5  # Random classifier baseline
                metrics.auc_pr = np.mean(gt_binary)  # Baseline for PR-AUC

        return metrics

    def _validate_medical_requirements(
        self, metrics: MedicalMetrics, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> MedicalMetrics:
        """Validate against medical-specific requirements."""

        # Check sensitivity requirements (critical for medical screening)
        min_sensitivity = self.config.precision_requirements.get(
            "min_sensitivity", 0.95
        )
        if metrics.sensitivity < min_sensitivity:
            metrics.safety_violations += 1
            self.safety_log.append(
                {
                    "violation_type": "insufficient_sensitivity",
                    "required": min_sensitivity,
                    "actual": metrics.sensitivity,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Check specificity requirements (important for avoiding false alarms)
        min_specificity = self.config.precision_requirements.get(
            "min_specificity", 0.90
        )
        if metrics.specificity < min_specificity:
            metrics.safety_violations += 1
            self.safety_log.append(
                {
                    "violation_type": "insufficient_specificity",
                    "required": min_specificity,
                    "actual": metrics.specificity,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Check false negative rate (critical for patient safety)
        max_fnr = self.config.precision_requirements.get(
            "max_false_negative_rate", 0.05
        )
        if metrics.false_negative_rate > max_fnr:
            metrics.safety_violations += 1
            self.safety_log.append(
                {
                    "violation_type": "excessive_false_negatives",
                    "max_allowed": max_fnr,
                    "actual": metrics.false_negative_rate,
                    "severity": "critical",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Clinical agreement simulation (would use expert annotations in practice)
        metrics.clinical_agreement = self._simulate_clinical_agreement(
            predictions, ground_truth
        )

        return metrics

    def _simulate_clinical_agreement(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Simulate agreement with clinical expert (Cohen's kappa)."""
        # Simplified simulation - in practice would use expert annotations
        # Add some noise to ground truth to simulate inter-observer variability
        np.random.seed(42)  # Reproducible simulation
        simulated_expert = ground_truth.copy()

        # Add 5% disagreement to simulate expert variability
        disagreement_mask = np.random.random(len(ground_truth)) < 0.05
        simulated_expert[disagreement_mask] = 1 - simulated_expert[disagreement_mask]

        # Calculate agreement between predictions and simulated expert
        if predictions.ndim > 1:
            pred_labels = np.argmax(predictions, axis=1)
        else:
            pred_labels = (predictions > 0.5).astype(int)

        agreement = np.mean(pred_labels == simulated_expert)
        return float(agreement)

    def _assess_safety_compliance(
        self, metrics: MedicalMetrics, execution_stats: dict[str, Any]
    ) -> MedicalMetrics:
        """Assess safety and reliability compliance."""

        # Check computational precision
        precision_errors = execution_stats.get("precision_violations", 0)
        metrics.precision_violations = precision_errors

        if precision_errors > 0:
            metrics.safety_violations += precision_errors
            self.safety_log.append(
                {
                    "violation_type": "precision_error",
                    "count": precision_errors,
                    "severity": "high",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Calculate reliability score
        total_operations = execution_stats.get("total_operations", 1)
        error_rate = (
            metrics.safety_violations + metrics.precision_violations
        ) / total_operations
        metrics.reliability_score = max(0.0, 1.0 - error_rate)

        # Safety level assessment
        if self.config.safety_level == SafetyLevel.CLASS_III:
            # Highest requirements for Class III devices
            min_reliability = 0.9999  # 99.99% reliability
            if metrics.reliability_score < min_reliability:
                metrics.safety_violations += 1
                self.safety_log.append(
                    {
                        "violation_type": "insufficient_reliability_class_iii",
                        "required": min_reliability,
                        "actual": metrics.reliability_score,
                        "severity": "critical",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return metrics

    def _calculate_compliance_score(self) -> float:
        """Calculate overall regulatory compliance score."""
        if not self.compliance_checklist:
            return 1.0  # No requirements specified

        completed_requirements = sum(self.compliance_checklist.values())
        total_requirements = len(self.compliance_checklist)

        return completed_requirements / total_requirements

    def validate_against_standard(
        self, standard: ComplianceStandard, evidence: dict[str, Any]
    ) -> bool:
        """
        Validate against specific regulatory standard.

        Args:
            standard: Compliance standard to check
            evidence: Evidence documents and data

        Returns:
            True if compliant with standard
        """
        if standard == ComplianceStandard.FDA_510K:
            return self._validate_fda_510k(evidence)
        elif standard == ComplianceStandard.IEC_62304:
            return self._validate_iec_62304(evidence)
        elif standard == ComplianceStandard.ISO_13485:
            return self._validate_iso_13485(evidence)
        else:
            logger.warning(f"Validation for {standard.value} not implemented")
            return False

    def _validate_fda_510k(self, evidence: dict[str, Any]) -> bool:
        """Validate FDA 510(k) compliance."""
        required_evidence = [
            "predicate_device_comparison",
            "substantial_equivalence_analysis",
            "safety_effectiveness_data",
            "clinical_data",
            "risk_analysis",
            "software_documentation",
        ]

        compliance = True
        for requirement in required_evidence:
            if requirement not in evidence:
                logger.error(f"Missing evidence for 510(k): {requirement}")
                compliance = False
            else:
                self.compliance_checklist[f"fda_510k_{requirement}"] = True

        return
