"""
Medical AI workload implementations.

Specialized workloads for medical imaging and healthcare applications
with emphasis on precision, safety, and regulatory compliance.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils.config import DataType, WorkloadConfig
from .base import BaseWorkload

logger = logging.getLogger(__name__)


class MedicalModalityType(Enum):
    """Medical imaging modality types."""

    CT_SCAN = "ct_scan"
    MRI = "mri"
    XRAY = "xray"
    ULTRASOUND = "ultrasound"
    MAMMOGRAPHY = "mammography"
    PET_SCAN = "pet_scan"
    ENDOSCOPY = "endoscopy"


class MedicalTaskType(Enum):
    """Medical AI task types."""

    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    RECONSTRUCTION = "reconstruction"
    ENHANCEMENT = "enhancement"
    DIAGNOSIS_SUPPORT = "diagnosis_support"


@dataclass
class MedicalWorkloadConfig(WorkloadConfig):
    """Configuration for medical workloads."""

    modality: MedicalModalityType
    task_type: MedicalTaskType
    image_size: Tuple[int, int, int]  # (H, W, D) or (H, W, C)
    batch_size: int = 1
    precision_level: str = "high"  # high, medical, research
    enable_safety_checks: bool = True
    regulatory_compliance: bool = True
    patient_data_protection: bool = True


class MedicalConvolution(BaseWorkload):
    """
    Medical imaging convolution workload.

    Optimized for medical imaging with high precision and safety features.
    """

    def __init__(self, config: MedicalWorkloadConfig):
        super().__init__(config)
        self.config = config

        # Medical-specific parameters
        self.precision_threshold = 1e-8 if config.precision_level == "medical" else 1e-6
        self.enable_redundant_computation = config.regulatory_compliance

        # Workload data
        self.input_tensor: Optional[np.ndarray] = None
        self.kernel_weights: Optional[np.ndarray] = None
        self.expected_output: Optional[np.ndarray] = None

        # Medical compliance tracking
        self.computation_log: List[Dict] = []
        self.safety_violations: List[str] = []

        logger.info(
            f"Initialized medical {config.task_type.value} workload for {config.modality.value}"
        )

    def generate_medical_ct_data(self, seed: int = 42) -> None:
        """Generate synthetic CT scan data for testing."""
        np.random.seed(seed)

        H, W, D = self.config.image_size

        # Generate realistic CT scan patterns
        # Simulate bone (high intensity), soft tissue (medium), air (low)
        self.input_tensor = np.zeros(
            (self.config.batch_size, H, W, D), dtype=np.float32
        )

        for batch in range(self.config.batch_size):
            # Create anatomical structures
            image = np.random.normal(0.3, 0.1, (H, W, D))  # Soft tissue baseline

            # Add bone structures (higher intensity)
            bone_mask = self._generate_bone_pattern(H, W, D)
            image[bone_mask] = np.random.normal(0.8, 0.1, np.sum(bone_mask))

            # Add air cavities (lower intensity)
            air_mask = self._generate_air_pattern(H, W, D)
            image[air_mask] = np.random.normal(0.05, 0.02, np.sum(air_mask))

            # Add noise (realistic CT noise)
            noise = np.random.normal(0, 0.02, (H, W, D))
            image += noise

            # Ensure medical range [0, 1] for Hounsfield units normalization
            image = np.clip(image, 0, 1)

            self.input_tensor[batch] = image

        # Generate appropriate convolution kernel for medical imaging
        self._generate_medical_kernel()

        logger.info(f"Generated CT scan data: {self.input_tensor.shape}")

    def _generate_bone_pattern(self, H: int, W: int, D: int) -> np.ndarray:
        """Generate realistic bone structure patterns."""
        bone_mask = np.zeros((H, W, D), dtype=bool)

        # Simple vertebrae-like structures
        center_x, center_y = H // 2, W // 2
        for z in range(D):
            # Circular bone structures
            y, x = np.ogrid[:H, :W]
            radius = 10 + 2 * np.sin(z * 0.3)  # Varying radius
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 < radius**2
            bone_mask[:, :, z] |= mask

        return bone_mask

    def _generate_air_pattern(self, H: int, W: int, D: int) -> np.ndarray:
        """Generate air cavity patterns (lungs, etc.)."""
        air_mask = np.zeros((H, W, D), dtype=bool)

        # Simple lung-like structures
        left_lung_x, left_lung_y = H // 4, W // 3
        right_lung_x, right_lung_y = H // 4, 2 * W // 3

        for z in range(D):
            y, x = np.ogrid[:H, :W]
            # Left lung
            mask_left = (x - left_lung_x) ** 2 + (y - left_lung_y) ** 2 < (15 + z) ** 2
            # Right lung
            mask_right = (x - right_lung_x) ** 2 + (y - right_lung_y) ** 2 < (
                15 + z
            ) ** 2
            air_mask[:, :, z] |= mask_left | mask_right

        return air_mask

    def _generate_medical_kernel(self) -> None:
        """Generate medical imaging appropriate convolution kernel."""
        if self.config.task_type == MedicalTaskType.ENHANCEMENT:
            # Sharpening kernel for edge enhancement
            self.kernel_weights = np.array(
                [[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]], dtype=np.float32
            ).reshape(1, 3, 3, 1)

        elif self.config.task_type == MedicalTaskType.SEGMENTATION:
            # Edge detection kernel for segmentation
            self.kernel_weights = np.array(
                [[[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype=np.float32
            ).reshape(1, 3, 3, 1)

        else:
            # General Gaussian blur for noise reduction
            kernel_size = 5
            sigma = 1.0
            kernel = np.zeros((kernel_size, kernel_size))
            center = kernel_size // 2

            for i in range(kernel_size):
                for j in range(kernel_size):
                    x, y = i - center, j - center
                    kernel[i, j] = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

            kernel = kernel / np.sum(kernel)  # Normalize
            self.kernel_weights = kernel.reshape(1, kernel_size, kernel_size, 1).astype(
                np.float32
            )

    def convert_to_systolic_gemm(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert convolution to GEMM format for systolic array execution.

        Returns:
            Tuple of (input_matrix, weight_matrix) for GEMM
        """
        if self.input_tensor is None or self.kernel_weights is None:
            raise ValueError(
                "Input data not generated. Call generate_medical_ct_data() first."
            )

        # Im2col transformation for convolution as GEMM
        batch_size, H, W, D = self.input_tensor.shape
        kernel_h, kernel_w = self.kernel_weights.shape[1:3]

        # Calculate output dimensions
        out_h = H - kernel_h + 1
        out_w = W - kernel_w + 1

        # Transform input using im2col
        input_cols = []
        for b in range(batch_size):
            for z in range(D):
                for i in range(out_h):
                    for j in range(out_w):
                        # Extract patch
                        patch = self.input_tensor[
                            b, i : i + kernel_h, j : j + kernel_w, z
                        ]
                        input_cols.append(patch.flatten())

        input_matrix = np.array(input_cols, dtype=np.float32)  # Shape: (N, K)
        weight_matrix = self.kernel_weights.reshape(-1, 1).astype(
            np.float32
        )  # Shape: (K, 1)

        # Medical compliance: log transformation
        if self.config.regulatory_compliance:
            self.computation_log.append(
                {
                    "operation": "im2col_transform",
                    "input_shape": self.input_tensor.shape,
                    "output_shape": input_matrix.shape,
                    "kernel_shape": self.kernel_weights.shape,
                    "precision_used": input_matrix.dtype,
                    "timestamp": np.datetime64("now"),
                }
            )

        logger.info(
            f"Converted to GEMM: A{input_matrix.shape} × B{weight_matrix.shape}"
        )
        return input_matrix, weight_matrix

    def verify_medical_precision(
        self, computed_output: np.ndarray, reference_output: np.ndarray
    ) -> bool:
        """
        Verify computation meets medical precision requirements.

        Args:
            computed_output: Output from accelerator
            reference_output: Reference implementation output

        Returns:
            True if precision requirements are met
        """
        if computed_output.shape != reference_output.shape:
            self.safety_violations.append("Output shape mismatch")
            return False

        # Check element-wise precision
        max_error = np.max(np.abs(computed_output - reference_output))
        relative_error = np.max(
            np.abs((computed_output - reference_output) / (reference_output + 1e-10))
        )

        precision_ok = (
            max_error < self.precision_threshold
            and relative_error < self.precision_threshold
        )

        if not precision_ok:
            self.safety_violations.append(
                f"Precision violation: max_error={max_error:.2e}, "
                f"rel_error={relative_error:.2e}, threshold={self.precision_threshold:.2e}"
            )

        # Medical compliance logging
        if self.config.regulatory_compliance:
            self.computation_log.append(
                {
                    "operation": "precision_verification",
                    "max_absolute_error": float(max_error),
                    "max_relative_error": float(relative_error),
                    "precision_threshold": self.precision_threshold,
                    "passed": precision_ok,
                    "timestamp": np.datetime64("now"),
                }
            )

        return precision_ok

    def get_medical_compliance_report(self) -> Dict[str, Any]:
        """
        Generate medical compliance report.

        Returns:
            Comprehensive compliance report
        """
        return {
            "workload_config": {
                "modality": self.config.modality.value,
                "task_type": self.config.task_type.value,
                "precision_level": self.config.precision_level,
                "regulatory_compliance": self.config.regulatory_compliance,
            },
            "computation_log": self.computation_log,
            "safety_violations": self.safety_violations,
            "precision_threshold": self.precision_threshold,
            "total_operations": len(self.computation_log),
            "violation_count": len(self.safety_violations),
            "compliance_status": "PASSED"
            if len(self.safety_violations) == 0
            else "FAILED",
        }


class MedicalResNet(BaseWorkload):
    """
    Medical ResNet workload for image classification.

    Implements ResNet-like operations common in medical imaging.
    """

    def __init__(self, config: MedicalWorkloadConfig):
        super().__init__(config)
        self.config = config

        # ResNet parameters
        self.num_classes = self._get_num_classes()
        self.layers = self._define_resnet_layers()

        # Medical validation
        self.diagnostic_confidence_threshold = 0.95
        self.safety_margin = 0.05

    def _get_num_classes(self) -> int:
        """Get number of classes based on medical task."""
        class_map = {
            MedicalModalityType.CT_SCAN: 5,  # Normal, Pneumonia, COVID, Cancer, Other
            MedicalModalityType.XRAY: 3,  # Normal, Pneumonia, Other
            MedicalModalityType.MRI: 4,  # Normal, Tumor, Lesion, Other
            MedicalModalityType.MAMMOGRAPHY: 2,  # Normal, Malignant
        }
        return class_map.get(self.config.modality, 2)

    def _define_resnet_layers(self) -> List[Dict]:
        """Define ResNet layer configuration for medical imaging."""
        layers = [
            {"type": "conv", "filters": 64, "kernel_size": 7, "stride": 2},
            {"type": "pool", "kernel_size": 3, "stride": 2},
            {"type": "resblock", "filters": 64, "num_blocks": 2},
            {"type": "resblock", "filters": 128, "num_blocks": 2},
            {"type": "resblock", "filters": 256, "num_blocks": 2},
            {"type": "resblock", "filters": 512, "num_blocks": 2},
            {"type": "global_pool"},
            {"type": "fc", "units": self.num_classes},
        ]
        return layers

    def generate_resnet_gemm_sequence(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate sequence of GEMM operations for ResNet inference.

        Returns:
            List of (input_matrix, weight_matrix) tuples
        """
        gemm_ops = []

        # Simulate feature map dimensions through network
        current_h, current_w = self.config.image_size[:2]
        current_channels = (
            1 if len(self.config.image_size) == 2 else self.config.image_size[2]
        )

        for layer in self.layers:
            if layer["type"] == "conv":
                # Convert convolution to GEMM
                input_matrix, weight_matrix = self._conv_to_gemm(
                    current_h,
                    current_w,
                    current_channels,
                    layer["filters"],
                    layer["kernel_size"],
                    layer.get("stride", 1),
                )
                gemm_ops.append((input_matrix, weight_matrix))

                # Update dimensions
                current_h = (
                    current_h - layer["kernel_size"] + layer.get("stride", 1)
                ) // layer.get("stride", 1)
                current_w = (
                    current_w - layer["kernel_size"] + layer.get("stride", 1)
                ) // layer.get("stride", 1)
                current_channels = layer["filters"]

            elif layer["type"] == "resblock":
                # ResNet block contains multiple conv layers
                for _ in range(layer["num_blocks"]):
                    # Main path convolutions
                    for kernel_size in [3, 3]:  # Typical ResNet block
                        input_matrix, weight_matrix = self._conv_to_gemm(
                            current_h,
                            current_w,
                            current_channels,
                            layer["filters"],
                            kernel_size,
                            1,
                        )
                        gemm_ops.append((input_matrix, weight_matrix))
                    current_channels = layer["filters"]

            elif layer["type"] == "fc":
                # Fully connected layer as GEMM
                input_size = current_h * current_w * current_channels
                input_matrix = np.random.randn(
                    self.config.batch_size, input_size
                ).astype(np.float32)
                weight_matrix = (
                    np.random.randn(input_size, layer["units"]).astype(np.float32)
                    * 0.01
                )
                gemm_ops.append((input_matrix, weight_matrix))

            elif layer["type"] == "pool":
                # Update dimensions for pooling
                current_h = current_h // layer.get("stride", 2)
                current_w = current_w // layer.get("stride", 2)

        logger.info(f"Generated {len(gemm_ops)} GEMM operations for ResNet")
        return gemm_ops

    def _conv_to_gemm(
        self,
        input_h: int,
        input_w: int,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert single convolution layer to GEMM format."""
        # Calculate output dimensions
        output_h = (input_h - kernel_size) // stride + 1
        output_w = (input_w - kernel_size) // stride + 1

        # Im2col matrix dimensions
        patch_size = kernel_size * kernel_size * input_channels
        num_patches = output_h * output_w * self.config.batch_size

        # Generate random data for simulation
        input_matrix = np.random.randn(num_patches, patch_size).astype(np.float32) * 0.1
        weight_matrix = (
            np.random.randn(patch_size, output_channels).astype(np.float32) * 0.01
        )

        return input_matrix, weight_matrix


class MedicalTransformer(BaseWorkload):
    """
    Medical Vision Transformer workload.

    Implements attention mechanisms for medical image analysis.
    """

    def __init__(self, config: MedicalWorkloadConfig):
        super().__init__(config)
        self.config = config

        # Transformer parameters
        self.patch_size = 16
        self.embed_dim = 768
        self.num_heads = 12
        self.num_layers = 12
        self.seq_length = self._calculate_sequence_length()

        # Medical-specific attention patterns
        self.anatomical_attention_bias = True
        self.diagnostic_attention_weights = None

    def _calculate_sequence_length(self) -> int:
        """Calculate sequence length based on image patches."""
        H, W = self.config.image_size[:2]
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        return num_patches + 1  # +1 for CLS token

    def generate_attention_gemm_ops(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate GEMM operations for multi-head attention.

        Returns:
            List of GEMM operations for attention computation
        """
        gemm_ops = []

        for layer in range(self.num_layers):
            # Self-attention operations
            for head in range(self.num_heads):
                head_dim = self.embed_dim // self.num_heads

                # Query, Key, Value projections
                input_matrix = (
                    np.random.randn(
                        self.config.batch_size * self.seq_length, self.embed_dim
                    ).astype(np.float32)
                    * 0.02
                )

                # Q projection
                q_weights = (
                    np.random.randn(self.embed_dim, head_dim).astype(np.float32) * 0.02
                )
                gemm_ops.append((input_matrix, q_weights))

                # K projection
                k_weights = (
                    np.random.randn(self.embed_dim, head_dim).astype(np.float32) * 0.02
                )
                gemm_ops.append((input_matrix, k_weights))

                # V projection
                v_weights = (
                    np.random.randn(self.embed_dim, head_dim).astype(np.float32) * 0.02
                )
                gemm_ops.append((input_matrix, v_weights))

                # Attention scores (Q @ K^T)
                q_matrix = (
                    np.random.randn(
                        self.config.batch_size, self.seq_length, head_dim
                    ).astype(np.float32)
                    * 0.02
                )
                k_matrix = (
                    np.random.randn(
                        self.config.batch_size, head_dim, self.seq_length
                    ).astype(np.float32)
                    * 0.02
                )

                # Reshape for batch GEMM
                q_reshaped = q_matrix.reshape(-1, head_dim)
                k_reshaped = k_matrix.transpose(0, 2, 1).reshape(-1, head_dim)
                gemm_ops.append((q_reshaped, k_reshaped.T))

                # Attention output (Attention @ V)
                v_matrix = (
                    np.random.randn(
                        self.config.batch_size, self.seq_length, head_dim
                    ).astype(np.float32)
                    * 0.02
                )
                attention_weights = np.random.softmax(
                    np.random.randn(
                        self.config.batch_size, self.seq_length, self.seq_length
                    ),
                    axis=-1,
                ).astype(np.float32)

                att_reshaped = attention_weights.reshape(-1, self.seq_length)
                v_reshaped = v_matrix.reshape(-1, head_dim)
                gemm_ops.append((att_reshaped, v_reshaped))

            # Feed-forward network
            ff_input = (
                np.random.randn(
                    self.config.batch_size * self.seq_length, self.embed_dim
                ).astype(np.float32)
                * 0.02
            )

            # First FF layer (expand)
            ff_weights1 = (
                np.random.randn(self.embed_dim, 4 * self.embed_dim).astype(np.float32)
                * 0.02
            )
            gemm_ops.append((ff_input, ff_weights1))

            # Second FF layer (contract)
            ff_hidden = (
                np.random.randn(
                    self.config.batch_size * self.seq_length, 4 * self.embed_dim
                ).astype(np.float32)
                * 0.02
            )
            ff_weights2 = (
                np.random.randn(4 * self.embed_dim, self.embed_dim).astype(np.float32)
                * 0.02
            )
            gemm_ops.append((ff_hidden, ff_weights2))

        # Final classification head
        cls_input = (
            np.random.randn(self.config.batch_size, self.embed_dim).astype(np.float32)
            * 0.02
        )
        cls_weights = (
            np.random.randn(self.embed_dim, self.num_classes).astype(np.float32) * 0.02
        )
        gemm_ops.append((cls_input, cls_weights))

        logger.info(
            f"Generated {len(gemm_ops)} GEMM operations for Medical Transformer"
        )
        return gemm_ops

    def apply_medical_attention_bias(self, attention_weights: np.ndarray) -> np.ndarray:
        """
        Apply medical domain-specific attention bias.

        Args:
            attention_weights: Raw attention weights [batch, heads, seq, seq]

        Returns:
            Biased attention weights focusing on diagnostically relevant regions
        """
        if not self.anatomical_attention_bias:
            return attention_weights

        # Create anatomical bias based on medical knowledge
        bias_matrix = self._create_anatomical_bias_matrix()

        # Apply bias to encourage attention to diagnostically important regions
        biased_weights = attention_weights + bias_matrix

        # Renormalize
        biased_weights = np.softmax(biased_weights, axis=-1)

        return biased_weights

    def _create_anatomical_bias_matrix(self) -> np.ndarray:
        """Create bias matrix based on anatomical importance."""
        # Simplified: center regions often more important in medical imaging
        bias = np.zeros((self.seq_length, self.seq_length))

        # Calculate 2D positions of patches
        patches_per_side = int(np.sqrt(self.seq_length - 1))  # -1 for CLS token

        for i in range(1, self.seq_length):  # Skip CLS token
            for j in range(1, self.seq_length):
                # Convert to 2D coordinates
                i_2d = (i - 1) // patches_per_side
                j_2d = (j - 1) // patches_per_side

                # Distance from center
                center = patches_per_side // 2
                dist_i = abs(i_2d - center)
                dist_j = abs(j_2d - center)

                # Bias towards center (where important anatomy often is)
                bias[i, j] = -0.1 * (dist_i + dist_j) / patches_per_side

        return bias


def create_medical_workload_suite() -> Dict[str, BaseWorkload]:
    """
    Create comprehensive medical workload suite.

    Returns:
        Dictionary of medical workloads for different scenarios
    """
    workloads = {}

    # CT Scan Analysis
    ct_config = MedicalWorkloadConfig(
        modality=MedicalModalityType.CT_SCAN,
        task_type=MedicalTaskType.SEGMENTATION,
        image_size=(512, 512, 64),
        batch_size=1,
        precision_level="medical",
        enable_safety_checks=True,
        regulatory_compliance=True,
    )
    workloads["ct_segmentation"] = MedicalConvolution(ct_config)

    # X-Ray Classification
    xray_config = MedicalWorkloadConfig(
        modality=MedicalModalityType.XRAY,
        task_type=MedicalTaskType.CLASSIFICATION,
        image_size=(224, 224, 1),
        batch_size=4,
        precision_level="high",
        enable_safety_checks=True,
        regulatory_compliance=True,
    )
    workloads["xray_classification"] = MedicalResNet(xray_config)

    # MRI Analysis with Transformer
    mri_config = MedicalWorkloadConfig(
        modality=MedicalModalityType.MRI,
        task_type=MedicalTaskType.DIAGNOSIS_SUPPORT,
        image_size=(256, 256, 3),
        batch_size=1,
        precision_level="medical",
        enable_safety_checks=True,
        regulatory_compliance=True,
    )
    workloads["mri_transformer"] = MedicalTransformer(mri_config)

    # Mammography Screening
    mammo_config = MedicalWorkloadConfig(
        modality=MedicalModalityType.MAMMOGRAPHY,
        task_type=MedicalTaskType.DETECTION,
        image_size=(1024, 1024, 1),
        batch_size=1,
        precision_level="medical",
        enable_safety_checks=True,
        regulatory_compliance=True,
    )
    workloads["mammography_detection"] = MedicalConvolution(mammo_config)

    logger.info(f"Created medical workload suite with {len(workloads)} workloads")
    return workloads
