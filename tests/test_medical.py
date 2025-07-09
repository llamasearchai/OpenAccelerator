"""
Comprehensive tests for medical modules.

Tests medical compliance, imaging, models, optimization, validation, and workflows.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest

from open_accelerator.medical.compliance import (
    AuditEvent,
    AuditEventType,
    ComplianceLevel,
    FDACompliance,
    HIPAACompliance,
    MedicalComplianceManager,
)
from open_accelerator.medical.imaging import ImageModality, MedicalImageProcessor
from open_accelerator.medical.models import MedicalModel, ModelMetrics, ModelType
from open_accelerator.medical.optimization import (
    MedicalOptimizer,
    create_medical_optimizer,
)
from open_accelerator.medical.validation import (
    MedicalValidator,
    create_medical_validator,
    validate_medical_model,
)
from open_accelerator.medical.workflows import (
    MedicalWorkflow,
    WorkflowStatus,
    WorkflowStep,
    create_medical_workflow,
)
from open_accelerator.utils.config import MedicalConfig


class TestHIPAACompliance:
    """Test HIPAA compliance features."""

    @pytest.fixture
    def hipaa_compliance(self):
        """Test HIPAA compliance instance."""
        from open_accelerator.medical.compliance import HIPAAConfig

        config = HIPAAConfig()
        return HIPAACompliance(config)

    def test_hipaa_initialization(self, hipaa_compliance):
        """Test HIPAA compliance initialization."""
        assert hipaa_compliance.compliance_level == ComplianceLevel.HIPAA
        assert hipaa_compliance.is_enabled is True
        assert hipaa_compliance.audit_trail is not None

    def test_phi_detection(self, hipaa_compliance):
        """Test PHI (Personal Health Information) detection."""
        # Test data with PHI
        phi_data = {
            "patient_name": "John Doe",
            "ssn": "123-45-6789",
            "medical_record": "MR123456",
            "image_data": "base64encodedimage",
        }

        detected_phi = hipaa_compliance.detect_phi(phi_data)
        assert len(detected_phi) > 0
        assert "patient_name" in detected_phi
        assert "ssn" in detected_phi

    def test_phi_anonymization(self, hipaa_compliance):
        """Test PHI anonymization."""
        phi_data = {
            "patient_name": "John Doe",
            "age": 45,
            "diagnosis": "Pneumonia",
            "image_data": "base64encodedimage",
        }

        anonymized_data = hipaa_compliance.anonymize_phi(phi_data)

        # Check that PHI is removed/anonymized
        assert (
            "patient_name" not in anonymized_data
            or anonymized_data["patient_name"] == "[REDACTED]"
        )
        assert "age" in anonymized_data  # Age might be preserved if within safe range
        assert "diagnosis" in anonymized_data  # Medical data should be preserved
        assert "image_data" in anonymized_data

    def test_encryption_requirements(self, hipaa_compliance):
        """Test encryption requirements."""
        sensitive_data = "Patient medical record data"

        # Test encryption
        encrypted_data = hipaa_compliance.encrypt_data(sensitive_data)
        assert encrypted_data != sensitive_data
        assert len(encrypted_data) > 0

        # Test decryption
        decrypted_data = hipaa_compliance.decrypt_data(encrypted_data)
        assert decrypted_data == sensitive_data

    def test_audit_logging(self, hipaa_compliance):
        """Test audit logging."""
        # Simulate access event
        access_event = AuditEvent(
            event_type=AuditEventType.DATA_ACCESS,
            user_id="doctor123",
            resource="patient_image",
            action="view",
            timestamp=datetime.now(),
        )

        hipaa_compliance.log_audit_event(access_event)

        # Check audit trail
        audit_logs = hipaa_compliance.get_audit_trail()
        assert len(audit_logs) > 0
        assert audit_logs[-1].event_type == AuditEventType.DATA_ACCESS
        assert audit_logs[-1].user_id == "doctor123"

    def test_compliance_validation(self, hipaa_compliance):
        """Test compliance validation."""
        # Test compliant data
        compliant_data = {
            "anonymized_id": "patient_001",
            "diagnosis": "Pneumonia",
            "image_data": "encrypted_base64_data",
        }

        is_compliant = hipaa_compliance.validate_compliance(compliant_data)
        assert is_compliant is True

        # Test non-compliant data
        non_compliant_data = {
            "patient_name": "John Doe",
            "ssn": "123-45-6789",
            "image_data": "unencrypted_data",
        }

        is_compliant = hipaa_compliance.validate_compliance(non_compliant_data)
        assert is_compliant is False


class TestFDACompliance:
    """Test FDA compliance features."""

    @pytest.fixture
    def fda_compliance(self):
        """Test FDA compliance instance."""
        return FDACompliance()

    def test_fda_initialization(self, fda_compliance):
        """Test FDA compliance initialization."""
        assert fda_compliance.compliance_level == ComplianceLevel.FDA
        assert fda_compliance.is_enabled is True
        assert fda_compliance.validation_requirements is not None

    def test_model_validation(self, fda_compliance):
        """Test FDA model validation."""
        # Mock model for testing
        mock_model = Mock()
        mock_model.accuracy = 0.95
        mock_model.precision = 0.92
        mock_model.recall = 0.88
        mock_model.f1_score = 0.90

        validation_result = fda_compliance.validate_model(mock_model)

        assert validation_result.is_valid is True
        assert validation_result.accuracy >= 0.9  # FDA requirement
        assert validation_result.precision >= 0.9  # FDA requirement

    def test_clinical_validation(self, fda_compliance):
        """Test clinical validation requirements."""
        clinical_data = {
            "study_id": "STUDY_001",
            "patient_count": 1000,
            "validation_accuracy": 0.94,
            "clinical_endpoints": ["sensitivity", "specificity"],
            "adverse_events": 0,
        }

        validation_result = fda_compliance.validate_clinical_data(clinical_data)

        assert validation_result.is_valid is True
        assert validation_result.patient_count >= 500  # Minimum requirement
        assert validation_result.adverse_events == 0

    def test_software_validation(self, fda_compliance):
        """Test software validation (510k requirements)."""
        software_info = {
            "version": "1.0.0",
            "validation_tests": 150,
            "test_coverage": 0.98,
            "regression_tests": 50,
            "security_tests": 25,
        }

        validation_result = fda_compliance.validate_software(software_info)

        assert validation_result.is_valid is True
        assert validation_result.test_coverage >= 0.95
        assert validation_result.validation_tests >= 100

    def test_risk_assessment(self, fda_compliance):
        """Test risk assessment."""
        risk_factors = {
            "patient_safety_impact": "low",
            "diagnostic_accuracy_impact": "medium",
            "failure_modes": ["false_positive", "false_negative"],
            "mitigation_strategies": ["human_oversight", "confidence_thresholds"],
        }

        risk_assessment = fda_compliance.assess_risk(risk_factors)

        assert risk_assessment.overall_risk_level in ["low", "medium", "high"]
        assert len(risk_assessment.mitigation_strategies) > 0


class TestMedicalImageProcessor:
    """Test medical image processing."""

    @pytest.fixture
    def image_processor(self):
        """Test medical image processor."""
        return MedicalImageProcessor()

    def test_image_processor_initialization(self, image_processor):
        """Test image processor initialization."""
        assert image_processor.supported_modalities is not None
        assert image_processor.supported_formats is not None
        assert ImageModality.CT in image_processor.supported_modalities
        assert ImageModality.MRI in image_processor.supported_modalities

    def test_dicom_processing(self, image_processor):
        """Test DICOM image processing."""
        # Mock DICOM data
        mock_dicom_data = b"DICOM_MOCK_DATA"

        with patch("open_accelerator.medical.imaging._pydicom") as mock_pydicom:
            mock_dataset = Mock()
            mock_dataset.pixel_array = np.random.rand(512, 512)
            mock_dataset.Modality = "CT"
            mock_dataset.PatientID = "PATIENT_001"
            mock_pydicom.dcmread.return_value = mock_dataset

            processed_image = image_processor.process_dicom(mock_dicom_data)

            assert processed_image.shape == (512, 512)
            assert processed_image.dtype == np.float32

    def test_nifti_processing(self, image_processor):
        """Test NIfTI image processing."""
        # Mock NIfTI data
        mock_nifti_path = "/path/to/brain.nii.gz"

        with patch("open_accelerator.medical.imaging._nibabel") as mock_nibabel:
            mock_img = Mock()
            mock_img.get_fdata.return_value = np.random.rand(256, 256, 128)
            mock_nibabel.load.return_value = mock_img

            processed_image = image_processor.process_nifti(mock_nifti_path)

            assert processed_image.shape == (256, 256, 128)
            assert processed_image.dtype == np.float32

    def test_image_preprocessing(self, image_processor):
        """Test image preprocessing."""
        # Create test image
        test_image = np.random.rand(512, 512) * 255

        # Test normalization
        normalized = image_processor.normalize_image(test_image)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

        # Test resizing
        resized = image_processor.resize_image(test_image, (256, 256))
        assert resized.shape == (256, 256)

        # Test augmentation
        augmented = image_processor.augment_image(test_image)
        assert augmented.shape == test_image.shape

    def test_medical_specific_processing(self, image_processor):
        """Test medical-specific processing."""
        # Create test CT image
        ct_image = np.random.rand(512, 512) * 4096 - 1024  # Hounsfield units

        # Test Hounsfield unit normalization
        normalized_ct = image_processor.normalize_hounsfield_units(ct_image)
        assert normalized_ct.min() >= -1.0
        assert normalized_ct.max() <= 1.0

        # Test windowing
        windowed = image_processor.apply_window(
            ct_image, window_center=40, window_width=400
        )
        assert windowed.min() >= 0
        assert windowed.max() <= 255

    def test_privacy_preservation(self, image_processor):
        """Test privacy preservation in image processing."""
        # Create test image with metadata
        test_image = np.random.rand(512, 512)
        metadata = {
            "patient_id": "PATIENT_001",
            "study_date": "20240108",
            "modality": "CT",
        }

        # Test metadata removal
        cleaned_image, cleaned_metadata = image_processor.remove_phi_metadata(
            test_image, metadata
        )

        assert np.array_equal(cleaned_image, test_image)  # Image should be unchanged
        assert "patient_id" not in cleaned_metadata
        assert "modality" in cleaned_metadata  # Medical data should be preserved


class TestMedicalModel:
    """Test medical AI models."""

    @pytest.fixture
    def medical_model(self):
        """Test medical model."""
        return MedicalModel(model_type=ModelType.SEGMENTATION)

    def test_model_initialization(self, medical_model):
        """Test model initialization."""
        assert medical_model.model_type == ModelType.SEGMENTATION
        assert medical_model.is_trained is False
        assert medical_model.metrics is not None

    def test_model_training(self, medical_model):
        """Test model training."""
        # Mock training data
        train_images = np.random.rand(10, 256, 256, 1)
        train_labels = np.random.randint(0, 2, (10, 256, 256, 1))

        with patch.object(medical_model, "_train_model") as mock_train:
            mock_train.return_value = True

            result = medical_model.train(train_images, train_labels)

            assert result is True
            assert medical_model.is_trained is True

    def test_model_inference(self, medical_model):
        """Test model inference."""
        # Mock trained model
        medical_model.is_trained = True

        # Mock input image
        input_image = np.random.rand(256, 256, 1)

        with patch.object(medical_model, "_predict") as mock_predict:
            mock_predict.return_value = np.random.rand(256, 256, 1)

            prediction = medical_model.predict(input_image)

            assert prediction.shape == (256, 256, 1)
            assert prediction.dtype == np.float32

    def test_model_validation(self, medical_model):
        """Test model validation."""
        # Mock validation data
        val_images = np.random.rand(5, 256, 256, 1)
        val_labels = np.random.randint(0, 2, (5, 256, 256, 1))

        medical_model.is_trained = True

        with patch.object(medical_model, "_validate_model") as mock_validate:
            mock_metrics = ModelMetrics(
                accuracy=0.95,
                precision=0.92,
                recall=0.88,
                f1_score=0.90,
                dice_score=0.85,
            )
            mock_validate.return_value = mock_metrics

            validation_result = medical_model.validate(val_images, val_labels)

            assert validation_result.accuracy >= 0.9
            assert validation_result.dice_score >= 0.8

    def test_model_explainability(self, medical_model):
        """Test model explainability features."""
        medical_model.is_trained = True

        # Mock input image
        input_image = np.random.rand(256, 256, 1)

        with patch.object(medical_model, "_generate_explanation") as mock_explain:
            mock_explain.return_value = {
                "attention_map": np.random.rand(256, 256),
                "feature_importance": np.random.rand(10),
                "confidence_score": 0.92,
            }

            explanation = medical_model.explain_prediction(input_image)

            assert "attention_map" in explanation
            assert "confidence_score" in explanation
            assert explanation["confidence_score"] >= 0.0
            assert explanation["confidence_score"] <= 1.0


class TestMedicalOptimizer:
    """Test medical optimization."""

    @pytest.fixture
    def medical_optimizer(self):
        """Test medical optimizer."""
        return MedicalOptimizer()

    def test_optimizer_initialization(self, medical_optimizer):
        """Test optimizer initialization."""
        assert medical_optimizer.optimization_strategies is not None
        assert medical_optimizer.metrics is not None

    def test_performance_optimization(self, medical_optimizer):
        """Test performance optimization."""
        # Mock model for optimization
        mock_model = Mock()
        mock_model.accuracy = 0.90
        mock_model.inference_time = 0.5
        mock_model.memory_usage = 1024

        optimization_result = medical_optimizer.optimize_performance(mock_model)

        assert optimization_result.optimized_model is not None
        assert optimization_result.performance_improvement > 0
        assert optimization_result.accuracy_preserved is True

    def test_memory_optimization(self, medical_optimizer):
        """Test memory optimization."""
        # Mock memory usage data
        memory_profile = {
            "model_size": 512,  # MB
            "activation_memory": 256,  # MB
            "gradient_memory": 128,  # MB
            "optimizer_memory": 64,  # MB
        }

        optimization_result = medical_optimizer.optimize_memory(memory_profile)

        assert optimization_result.memory_reduction > 0
        assert (
            optimization_result.optimized_profile["model_size"]
            <= memory_profile["model_size"]
        )

    def test_accuracy_preservation(self, medical_optimizer):
        """Test accuracy preservation during optimization."""
        # Mock model with high accuracy
        mock_model = Mock()
        mock_model.accuracy = 0.95
        mock_model.precision = 0.92
        mock_model.recall = 0.88

        # Test that optimization preserves accuracy
        optimization_result = medical_optimizer.optimize_with_accuracy_constraint(
            mock_model, min_accuracy=0.93
        )

        assert optimization_result.final_accuracy >= 0.93
        assert optimization_result.optimization_successful is True

    def test_medical_specific_optimization(self, medical_optimizer):
        """Test medical-specific optimization."""
        # Test optimization for medical workloads
        medical_workload = {
            "modality": "CT",
            "image_size": (512, 512),
            "batch_size": 4,
            "precision_requirement": "high",
            "latency_requirement": "low",
        }

        optimization_result = medical_optimizer.optimize_for_medical_workload(
            medical_workload
        )

        assert optimization_result.optimized_config is not None
        assert optimization_result.meets_medical_requirements is True


class TestMedicalValidator:
    """Test medical validation."""

    @pytest.fixture
    def medical_validator(self):
        """Test medical validator."""
        return MedicalValidator()

    def test_validator_initialization(self, medical_validator):
        """Test validator initialization."""
        assert medical_validator.validation_criteria is not None

    def test_clinical_validation(self, medical_validator):
        """Test clinical validation."""
        # Mock clinical data
        clinical_data = {
            "patient_demographics": {
                "age_range": "18-80",
                "gender_distribution": "balanced",
            },
            "study_design": "prospective",
            "sample_size": 1000,
            "ground_truth": "expert_annotation",
            "validation_metrics": {"sensitivity": 0.95, "specificity": 0.98},
        }

        validation_result = medical_validator.validate_clinical_data(clinical_data)

        assert validation_result.is_valid is True
        assert validation_result.confidence_level >= 0.95
        assert validation_result.sample_size_adequate is True

    def test_regulatory_validation(self, medical_validator):
        """Test regulatory validation."""
        # Mock regulatory requirements
        regulatory_data = {
            "fda_clearance": True,
            "ce_marking": True,
            "iso_compliance": ["ISO 13485", "ISO 14155"],
            "clinical_evidence": "sufficient",
            "risk_classification": "Class II",
        }

        validation_result = medical_validator.validate_regulatory_compliance(
            regulatory_data
        )

        assert validation_result.is_compliant is True
        assert validation_result.regulatory_approval is True

    def test_technical_validation(self, medical_validator):
        """Test technical validation."""
        # Mock technical specifications
        technical_specs = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88,
            "processing_time": 0.5,
            "memory_usage": 512,
            "robustness_score": 0.90,
        }

        validation_result = medical_validator.validate_technical_specs(technical_specs)

        assert validation_result.meets_requirements is True
        assert validation_result.performance_adequate is True

    def test_safety_validation(self, medical_validator):
        """Test safety validation."""
        # Mock safety data
        safety_data = {
            "adverse_events": 0,
            "failure_modes": ["false_positive", "false_negative"],
            "mitigation_strategies": ["human_oversight", "confidence_thresholds"],
            "safety_monitoring": "continuous",
            "risk_assessment": "low",
        }

        validation_result = medical_validator.validate_safety(safety_data)

        assert validation_result.is_safe is True
        assert validation_result.risk_level == "low"
        assert len(validation_result.mitigation_strategies) > 0


class TestMedicalWorkflow:
    """Test medical workflows."""

    @pytest.fixture
    def medical_workflow(self):
        """Test medical workflow."""
        return MedicalWorkflow("CT_Segmentation_Workflow")

    def test_workflow_initialization(self, medical_workflow):
        """Test workflow initialization."""
        assert medical_workflow.name == "CT_Segmentation_Workflow"
        assert medical_workflow.status == WorkflowStatus.CREATED
        assert medical_workflow.steps == []

    def test_workflow_step_addition(self, medical_workflow):
        """Test adding steps to workflow."""
        # Add image preprocessing step
        preprocess_step = WorkflowStep(
            name="image_preprocessing",
            description="Preprocess CT image with windowing",
            parameters={"window_center": 40, "window_width": 400},
        )

        medical_workflow.add_step(preprocess_step)

        assert len(medical_workflow.steps) == 1
        assert medical_workflow.steps[0].name == "image_preprocessing"

    def test_workflow_execution(self, medical_workflow):
        """Test workflow execution."""
        # Add mock steps
        steps = [
            WorkflowStep(
                "load_image", "Load DICOM image", parameters={"path": "/test/image.dcm"}
            ),
            WorkflowStep(
                "preprocess", "Preprocess image", parameters={"normalize": True}
            ),
            WorkflowStep(
                "segment", "Segment image", parameters={"model": "segmentation_model"}
            ),
            WorkflowStep(
                "postprocess", "Postprocess segmentation", parameters={"smooth": True}
            ),
        ]

        for step in steps:
            medical_workflow.add_step(step)

        # Mock step execution
        with patch.object(medical_workflow, "_execute_step") as mock_execute:
            mock_execute.return_value = {
                "status": "success",
                "output": "processed_data",
            }

            result = medical_workflow.execute()

            assert result.status == WorkflowStatus.COMPLETED
            assert result.success is True
            assert mock_execute.call_count == 4

    def test_workflow_validation(self, medical_workflow):
        """Test workflow validation."""
        # Add a step to make the workflow valid
        medical_workflow.add_step(
            WorkflowStep(
                name="input_validation",
                description="Validate input data",
                required=True,
            )
        )
        validation_result = medical_workflow.validate()
        assert validation_result.is_valid is True
        assert validation_result.has_input_validation is True
        assert validation_result.has_output_validation is False

    def test_workflow_error_handling(self, medical_workflow):
        """Test workflow error handling."""
        # Add step that will fail
        failing_step = WorkflowStep("failing_step", "Step that fails", parameters={})
        medical_workflow.add_step(failing_step)

        with patch.object(medical_workflow, "_execute_step") as mock_execute:
            mock_execute.side_effect = Exception("Step execution failed")

            result = medical_workflow.execute()

            assert result.status == WorkflowStatus.FAILED
            assert result.success is False
            assert result.error_message is not None

    def test_workflow_monitoring(self, medical_workflow):
        """Test workflow monitoring."""
        # Add steps with monitoring
        monitored_steps = [
            WorkflowStep("step1", "First step", parameters={"monitor": True}),
            WorkflowStep("step2", "Second step", parameters={"monitor": True}),
        ]

        for step in monitored_steps:
            medical_workflow.add_step(step)

        # Enable monitoring
        medical_workflow.enable_monitoring(True)

        with patch.object(medical_workflow, "_execute_step") as mock_execute:
            mock_execute.return_value = {
                "status": "success",
                "metrics": {"duration": 0.5},
            }

            result = medical_workflow.execute()

            assert result.monitoring_data is not None
            assert len(result.monitoring_data) == 2


class TestMedicalIntegration:
    """Test integration between medical components."""

    @pytest.fixture
    def medical_config(self):
        """Test medical configuration."""
        return MedicalConfig(
            enable_medical_mode=True,
            hipaa_compliance=True,
            fda_compliance=True,
            audit_logging=True,
            encryption_enabled=True,
        )

    def test_compliance_integration(self, medical_config):
        """Test integration between compliance components."""
        # Create compliance manager
        compliance_manager = MedicalComplianceManager(medical_config)

        # Test that all compliance features are enabled
        assert compliance_manager.hipaa_compliance is not None
        assert compliance_manager.fda_compliance is not None
        assert compliance_manager.audit_logging_enabled is True

    def test_end_to_end_medical_workflow(self, medical_config):
        """Test complete medical workflow."""
        # Create medical workflow
        workflow = create_medical_workflow("End_to_End_CT_Analysis")

        # Add comprehensive steps
        steps = [
            WorkflowStep(
                "load_dicom", "Load DICOM image", parameters={"path": "/test/ct.dcm"}
            ),
            WorkflowStep("phi_removal", "Remove PHI", parameters={"anonymize": True}),
            WorkflowStep(
                "preprocessing", "Preprocess CT", parameters={"normalize": True}
            ),
            WorkflowStep(
                "segmentation",
                "Segment organs",
                parameters={"model": "organ_segmentation"},
            ),
            WorkflowStep(
                "validation", "Validate results", parameters={"clinical_review": True}
            ),
            WorkflowStep(
                "compliance_check",
                "Check compliance",
                parameters={"hipaa": True, "fda": True},
            ),
            WorkflowStep(
                "audit_log",
                "Log audit event",
                parameters={"event_type": "analysis_complete"},
            ),
        ]

        for step in steps:
            workflow.add_step(step)

        # Mock execution
        with patch.object(workflow, "_execute_step") as mock_execute:
            mock_execute.return_value = {"status": "success", "compliant": True}

            # Execute the workflow
            result = workflow.execute(
                input_data={"patient_id": "test_patient", "data": {}}
            )

            # Validate results
            assert result["status"] == "completed"

    def test_medical_ai_optimization(self, medical_config):
        """Test medical AI optimization."""
        optimizer = create_medical_optimizer("performance", config=medical_config)
        medical_workload = {
            "modality": "CT",
            "image_size": (512, 512),
            "batch_size": 4,
            "precision_requirement": "high",
            "latency_requirement": "low",
        }
        optimization_result = optimizer.optimize_for_medical_workload(medical_workload)
        assert optimization_result.meets_medical_requirements is True

    def test_medical_validation_pipeline(self, medical_config):
        """Test the full medical validation pipeline."""
        mock_model = {"accuracy": 0.98}
        _ = create_medical_validator(medical_config)
        validation_result = validate_medical_model(mock_model, None)
        assert validation_result["model_validation"]["is_valid"] is True
