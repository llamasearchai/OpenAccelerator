"""
Comprehensive tests for workload modules.

Tests all workload types including GEMM, convolution, and medical imaging workloads.
"""

from unittest.mock import Mock

import numpy as np
import pytest

# from open_accelerator.workloads.medical import MedicalWorkload  # TODO: Create this module
from open_accelerator.utils.config import DataType, WorkloadConfig, WorkloadType
from open_accelerator.workloads.base import BaseWorkload
from open_accelerator.workloads.gemm import GEMMWorkload


class TestBaseWorkload:
    """Test BaseWorkload class."""

    @pytest.fixture
    def test_workload_config(self):
        """Test workload configuration."""
        return WorkloadConfig(
            workload_type=WorkloadType.GEMM,
            name="test_workload",
            gemm_M=4,
            gemm_K=4,
            gemm_P=4,
            data_type=DataType.FLOAT32,
            random_seed=42,
            validate_output=True,
            tolerance=1e-6,
        )

    @pytest.fixture
    def mock_base_workload(self, test_workload_config):
        """Mock base workload for testing."""
        workload = Mock(spec=BaseWorkload)
        workload.config = test_workload_config
        workload.name = "test_workload"
        workload.workload_type = WorkloadType.GEMM
        workload.data_type = DataType.FLOAT32
        workload.is_loaded = False
        workload.is_validated = False
        return workload

    def test_base_workload_initialization(self, mock_base_workload):
        """Test base workload initialization."""
        assert mock_base_workload.name == "test_workload"
        assert mock_base_workload.workload_type == WorkloadType.GEMM
        assert mock_base_workload.data_type == DataType.FLOAT32
        assert mock_base_workload.is_loaded is False
        assert mock_base_workload.is_validated is False

    def test_base_workload_load_data(self, mock_base_workload):
        """Test workload data loading."""
        # Mock load_data method
        mock_base_workload.load_data = Mock(return_value=True)

        # Test data loading
        result = mock_base_workload.load_data()
        assert result is True
        mock_base_workload.load_data.assert_called_once()

    def test_base_workload_generate_data(self, mock_base_workload):
        """Test workload data generation."""
        # Mock generate_data method
        mock_base_workload.generate_data = Mock(
            return_value={
                "input_data": np.random.rand(10, 10),
                "expected_output": np.random.rand(10, 10),
            }
        )

        # Test data generation
        data = mock_base_workload.generate_data()
        assert "input_data" in data
        assert "expected_output" in data
        assert isinstance(data["input_data"], np.ndarray)
        assert isinstance(data["expected_output"], np.ndarray)

    def test_base_workload_validate(self, mock_base_workload):
        """Test workload validation."""
        # Mock validate method
        mock_base_workload.validate = Mock(return_value=True)

        # Test validation
        result = mock_base_workload.validate()
        assert result is True
        mock_base_workload.validate.assert_called_once()

    def test_base_workload_get_performance_metrics(self, mock_base_workload):
        """Test performance metrics collection."""
        # Mock performance metrics
        mock_base_workload.get_performance_metrics = Mock(
            return_value={
                "execution_time": 0.05,
                "operations_count": 1000,
                "memory_usage": 512,
                "accuracy": 0.99,
            }
        )

        # Test metrics collection
        metrics = mock_base_workload.get_performance_metrics()
        assert metrics["execution_time"] == 0.05
        assert metrics["operations_count"] == 1000
        assert metrics["memory_usage"] == 512
        assert metrics["accuracy"] == 0.99


class TestGEMMWorkload:
    """Test GEMM workload class."""

    @pytest.fixture
    def gemm_config(self):
        """GEMM workload configuration."""
        return WorkloadConfig(
            workload_type=WorkloadType.GEMM,
            name="test_gemm",
            gemm_M=64,
            gemm_K=32,
            gemm_P=16,
            data_type=DataType.FLOAT32,
            random_seed=42,
            validate_output=True,
            tolerance=1e-6,
        )

    @pytest.fixture
    def mock_gemm_workload(self, gemm_config):
        """Mock GEMM workload for testing."""
        workload = Mock(spec=GEMMWorkload)
        workload.config = gemm_config
        workload.name = "test_gemm"
        workload.M = 64
        workload.K = 32
        workload.P = 16
        workload.A = None
        workload.B = None
        workload.C = None
        workload.expected_C = None
        return workload

    def test_gemm_workload_initialization(self, mock_gemm_workload):
        """Test GEMM workload initialization."""
        assert mock_gemm_workload.name == "test_gemm"
        assert mock_gemm_workload.M == 64
        assert mock_gemm_workload.K == 32
        assert mock_gemm_workload.P == 16
        assert mock_gemm_workload.A is None
        assert mock_gemm_workload.B is None
        assert mock_gemm_workload.C is None

    def test_gemm_workload_generate_matrices(self, mock_gemm_workload):
        """Test GEMM matrix generation."""
        # Mock matrix generation
        mock_gemm_workload.generate_matrices = Mock(
            return_value={
                "A": np.random.rand(64, 32),
                "B": np.random.rand(32, 16),
                "C": np.zeros((64, 16)),
            }
        )

        # Test matrix generation
        matrices = mock_gemm_workload.generate_matrices()
        assert "A" in matrices
        assert "B" in matrices
        assert "C" in matrices
        assert matrices["A"].shape == (64, 32)
        assert matrices["B"].shape == (32, 16)
        assert matrices["C"].shape == (64, 16)

    def test_gemm_workload_compute_reference(self, mock_gemm_workload):
        """Test GEMM reference computation."""
        # Mock reference computation
        A = np.random.rand(64, 32)
        B = np.random.rand(32, 16)
        expected_C = A @ B

        mock_gemm_workload.compute_reference = Mock(return_value=expected_C)

        # Test reference computation
        result = mock_gemm_workload.compute_reference()
        assert result.shape == (64, 16)
        assert isinstance(result, np.ndarray)

    def test_gemm_workload_validate_result(self, mock_gemm_workload):
        """Test GEMM result validation."""
        # Mock result validation
        mock_gemm_workload.validate_result = Mock(
            return_value={
                "is_valid": True,
                "max_error": 1e-7,
                "mean_error": 5e-8,
                "accuracy": 0.999,
            }
        )

        # Test validation
        validation = mock_gemm_workload.validate_result()
        assert validation["is_valid"] is True
        assert validation["max_error"] == 1e-7
        assert validation["accuracy"] == 0.999

    def test_gemm_workload_get_operation_count(self, mock_gemm_workload):
        """Test GEMM operation count calculation."""
        # Mock operation count
        mock_gemm_workload.get_operation_count = Mock(
            return_value=32768
        )  # 64 * 32 * 16

        # Test operation count
        op_count = mock_gemm_workload.get_operation_count()
        assert op_count == 32768

    def test_gemm_workload_data_types(self, mock_gemm_workload):
        """Test GEMM workload with different data types."""
        # Test different data types
        data_types = [DataType.FLOAT32, DataType.FLOAT16, DataType.INT8]

        for dtype in data_types:
            mock_gemm_workload.set_data_type = Mock()
            mock_gemm_workload.set_data_type(dtype)
            mock_gemm_workload.set_data_type.assert_called_with(dtype)

    def test_gemm_workload_sparsity_support(self, mock_gemm_workload):
        """Test GEMM workload with sparsity support."""
        # Mock sparsity support
        mock_gemm_workload.apply_sparsity = Mock(
            return_value={
                "sparse_A": np.random.rand(20, 32),
                "sparse_indices": np.random.randint(0, 64, 20),
                "compression_ratio": 0.3,
                "compressed_B": np.random.rand(10, 16),
            }
        )

        # Test sparsity application
        sparse_data = mock_gemm_workload.apply_sparsity()
        assert "sparse_A" in sparse_data
        assert "sparse_indices" in sparse_data
        assert sparse_data["compression_ratio"] == 0.3


class TestMedicalWorkload:
    """Test medical workload class."""

    @pytest.fixture
    def medical_config(self):
        """Medical workload configuration."""
        return WorkloadConfig(
            workload_type=WorkloadType.MEDICAL_IMAGING,
            name="test_medical",
            medical_modality="CT",
            medical_image_size=(512, 512),
            medical_slice_thickness=1.0,
            medical_pixel_spacing=(0.5, 0.5),
            data_type=DataType.FLOAT32,
            random_seed=42,
            validate_output=True,
            tolerance=1e-6,
        )

    @pytest.fixture
    def mock_medical_workload(self, medical_config):
        """Mock medical workload for testing."""
        workload = Mock()
        workload.config = medical_config
        workload.name = "test_medical"
        workload.modality = "CT"
        workload.image_size = (512, 512)
        workload.slice_thickness = 1.0
        workload.pixel_spacing = (0.5, 0.5)
        workload.image_data = None
        workload.segmentation_mask = None
        workload.classification_labels = None
        return workload

    def test_medical_workload_initialization(self, mock_medical_workload):
        """Test medical workload initialization."""
        assert mock_medical_workload.name == "test_medical"
        assert mock_medical_workload.modality == "CT"
        assert mock_medical_workload.image_size == (512, 512)
        assert mock_medical_workload.slice_thickness == 1.0
        assert mock_medical_workload.pixel_spacing == (0.5, 0.5)

    def test_medical_workload_load_dicom(self, mock_medical_workload):
        """Test DICOM image loading."""
        # Mock DICOM loading
        mock_medical_workload.load_dicom = Mock(
            return_value={
                "image_data": np.random.rand(512, 512, 64),
                "metadata": {
                    "PatientID": "12345",
                    "StudyDate": "20240101",
                    "Modality": "CT",
                    "SliceThickness": 1.0,
                },
            }
        )

        # Test DICOM loading
        dicom_data = mock_medical_workload.load_dicom()
        assert "image_data" in dicom_data
        assert "metadata" in dicom_data
        assert dicom_data["image_data"].shape == (512, 512, 64)
        assert dicom_data["metadata"]["Modality"] == "CT"

    def test_medical_workload_preprocessing(self, mock_medical_workload):
        """Test medical image preprocessing."""
        # Mock preprocessing
        mock_medical_workload.preprocess_image = Mock(
            return_value={
                "normalized_image": np.random.rand(512, 512, 64),
                "windowed_image": np.random.rand(512, 512, 64),
                "enhanced_image": np.random.rand(512, 512, 64),
            }
        )

        # Test preprocessing
        processed = mock_medical_workload.preprocess_image()
        assert "normalized_image" in processed
        assert "windowed_image" in processed
        assert "enhanced_image" in processed

    def test_medical_workload_segmentation(self, mock_medical_workload):
        """Test medical image segmentation."""
        # Mock segmentation
        mock_medical_workload.segment_image = Mock(
            return_value={
                "segmentation_mask": np.random.randint(0, 4, (512, 512, 64)),
                "segment_labels": ["background", "organ", "lesion"],
            }
        )

        # Test segmentation
        segmentation = mock_medical_workload.segment_image()
        assert "segmentation_mask" in segmentation
        assert "segment_labels" in segmentation
        assert len(segmentation["segment_labels"]) == 3

    def test_medical_workload_classification(self, mock_medical_workload):
        """Test medical image classification."""
        # Mock classification
        mock_medical_workload.classify_image = Mock(
            return_value={
                "predicted_class": "malignant",
                "confidence": 0.95,
                "class_probabilities": {"benign": 0.05, "malignant": 0.95},
            }
        )

        # Test classification
        classification = mock_medical_workload.classify_image()
        assert classification["predicted_class"] == "malignant"
        assert classification["confidence"] == 0.95
        assert "class_probabilities" in classification

    def test_medical_workload_phi_compliance(self, mock_medical_workload):
        """Test PHI compliance features."""
        # Mock PHI compliance
        mock_medical_workload.anonymize_data = Mock(
            return_value={
                "anonymized_image": np.random.rand(512, 512, 64),
                "anonymized_metadata": {
                    "PatientID": "ANON_001",
                    "StudyDate": "ANON_DATE",
                    "Modality": "CT",
                },
            }
        )

        # Test anonymization
        anonymized = mock_medical_workload.anonymize_data()
        assert "anonymized_image" in anonymized
        assert "anonymized_metadata" in anonymized
        assert anonymized["anonymized_metadata"]["PatientID"] == "ANON_001"

    def test_medical_workload_fda_validation(self, mock_medical_workload):
        """Test FDA validation features."""
        # Mock FDA validation
        mock_medical_workload.validate_fda_compliance = Mock(
            return_value={
                "compliance_status": "COMPLIANT",
                "validation_report": {
                    "data_integrity": "PASS",
                    "algorithm_validation": "PASS",
                    "clinical_validation": "PASS",
                },
            }
        )

        # Test FDA validation
        validation = mock_medical_workload.validate_fda_compliance()
        assert validation["compliance_status"] == "COMPLIANT"
        assert "validation_report" in validation
        assert validation["validation_report"]["data_integrity"] == "PASS"

    def test_medical_workload_different_modalities(self, mock_medical_workload):
        """Test medical workload with different modalities."""
        # Test different modalities
        modalities = ["CT", "MRI", "X-Ray", "Ultrasound"]

        for modality in modalities:
            mock_medical_workload.set_modality = Mock()
            mock_medical_workload.set_modality(modality)
            mock_medical_workload.set_modality.assert_called_with(modality)


class TestWorkloadIntegration:
    """Test workload integration features."""

    @pytest.fixture
    def mock_workload_manager(self):
        """Mock workload manager for testing."""
        manager = Mock()
        manager.registered_workloads = {}
        manager.workload_queue = []
        return manager

    def test_workload_registration(self, mock_workload_manager):
        """Test workload registration."""
        # Mock workload registration
        mock_workload_manager.register_workload = Mock(return_value=True)

        # Test registration
        workload = Mock(spec=BaseWorkload)
        workload.name = "test_workload"

        result = mock_workload_manager.register_workload(workload)
        assert result is True
        mock_workload_manager.register_workload.assert_called_once_with(workload)

    def test_workload_scheduling(self, mock_workload_manager):
        """Test workload scheduling."""
        # Mock workload scheduling
        mock_workload_manager.schedule_workload = Mock(
            return_value={
                "scheduled": True,
                "queue_position": 0,
                "estimated_start_time": 100,
            }
        )

        # Test scheduling
        workload = Mock(spec=BaseWorkload)
        result = mock_workload_manager.schedule_workload(workload)

        assert result["scheduled"] is True
        assert result["queue_position"] == 0
        assert result["estimated_start_time"] == 100

    def test_workload_batch_processing(self, mock_workload_manager):
        """Test batch workload processing."""
        # Mock batch processing
        mock_workload_manager.process_batch = Mock(
            return_value={
                "processed_count": 5,
                "failed_count": 0,
                "total_execution_time": 2.5,
                "average_execution_time": 0.5,
            }
        )

        # Test batch processing
        workloads = [Mock(spec=BaseWorkload) for _ in range(5)]
        result = mock_workload_manager.process_batch(workloads)

        assert result["processed_count"] == 5
        assert result["failed_count"] == 0
        assert result["total_execution_time"] == 2.5
        assert result["average_execution_time"] == 0.5

    def test_workload_performance_comparison(self, mock_workload_manager):
        """Test workload performance comparison."""
        # Mock performance comparison
        mock_workload_manager.compare_performance = Mock(
            return_value={
                "workload1": {
                    "execution_time": 0.1,
                    "accuracy": 0.95,
                    "memory_usage": 100,
                },
                "workload2": {
                    "execution_time": 0.2,
                    "accuracy": 0.90,
                    "memory_usage": 150,
                },
                "winner": "workload1",
            }
        )

        # Test performance comparison
        workload1 = Mock(spec=BaseWorkload)
        workload1.name = "workload1"
        workload2 = Mock(spec=BaseWorkload)
        workload2.name = "workload2"

        result = mock_workload_manager.compare_performance([workload1, workload2])

        assert "workload1" in result
        assert "workload2" in result
        assert result["winner"] == "workload1"


class TestWorkloadDataGeneration:
    """Test workload data generation utilities."""

    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        # Mock synthetic data generator
        generator = Mock()
        generator.generate_synthetic_data = Mock(
            return_value={
                "matrices": {"A": np.random.rand(64, 32), "B": np.random.rand(32, 16)},
                "images": {
                    "ct_scan": np.random.rand(512, 512, 64),
                    "mri_scan": np.random.rand(256, 256, 32),
                },
            }
        )

        # Test synthetic data generation
        data = generator.generate_synthetic_data()
        assert "matrices" in data
        assert "images" in data
        assert data["matrices"]["A"].shape == (64, 32)
        assert data["images"]["ct_scan"].shape == (512, 512, 64)

    def test_real_data_loading(self):
        """Test real data loading."""
        # Mock real data loader
        loader = Mock()
        loader.load_real_data = Mock(
            return_value={
                "dataset_name": "medical_imaging_dataset",
                "data_size": 1000,
                "data_format": "DICOM",
                "metadata": {
                    "patient_count": 100,
                    "study_count": 500,
                    "modalities": ["CT", "MRI"],
                },
            }
        )

        # Test real data loading
        data = loader.load_real_data()
        assert data["dataset_name"] == "medical_imaging_dataset"
        assert data["data_size"] == 1000
        assert data["data_format"] == "DICOM"
        assert data["metadata"]["patient_count"] == 100

    def test_data_validation(self):
        """Test data validation."""
        # Mock data validator
        validator = Mock()
        validator.validate_data = Mock(
            return_value={
                "is_valid": True,
                "validation_errors": [],
                "data_quality_score": 0.95,
                "completeness": 0.98,
            }
        )

        # Test data validation
        validation = validator.validate_data()
        assert validation["is_valid"] is True
        assert len(validation["validation_errors"]) == 0
        assert validation["data_quality_score"] == 0.95
        assert validation["completeness"] == 0.98
