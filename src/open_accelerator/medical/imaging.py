"""
Medical imaging module for Open Accelerator.

Provides comprehensive medical image processing capabilities including DICOM support,
medical image enhancement, and specialized workflows for radiology and pathology.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ImageModality(Enum):
    """Medical imaging modalities."""

    CT = "CT"
    MRI = "MRI"
    X_RAY = "X-Ray"
    ULTRASOUND = "Ultrasound"
    MAMMOGRAPHY = "Mammography"
    PET = "PET"
    SPECT = "SPECT"
    OCT = "OCT"
    PATHOLOGY = "Pathology"
    HISTOLOGY = "Histology"
    ENDOSCOPY = "Endoscopy"
    DERMOSCOPY = "Dermoscopy"
    FUNDUS = "Fundus"
    FLUOROSCOPY = "Fluoroscopy"


class ImageFormat(Enum):
    """Supported image formats."""

    DICOM = "DICOM"
    NIFTI = "NIFTI"
    PNG = "PNG"
    JPEG = "JPEG"
    TIFF = "TIFF"
    NUMPY = "NUMPY"
    HDF5 = "HDF5"
    ZARR = "ZARR"


@dataclass
class MedicalImageMetadata:
    """Medical image metadata."""

    patient_id: Optional[str] = None
    study_date: Optional[datetime] = None
    modality: str = "CT"
    series_description: str = ""
    image_dimensions: Tuple[int, ...] = (512, 512)
    pixel_spacing: Tuple[float, float] = (1.0, 1.0)
    slice_thickness: float = 1.0
    acquisition_parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.acquisition_parameters is None:
            self.acquisition_parameters = {}


class MedicalImageProcessor:
    """
    Advanced medical image processor with healthcare-specific optimizations.

    Supports DICOM processing, medical image enhancement, and integration
    with accelerator simulation for medical AI workloads.
    """

    def __init__(self, enable_gpu: bool = False, compliance_mode: bool = True):
        """Initialize medical image processor."""
        self.enable_gpu = enable_gpu
        self.compliance_mode = compliance_mode
        self.processing_history: List[Dict[str, Any]] = []

        # Initialize processing capabilities
        self._initialize_processors()
        logger.info("Medical image processor initialized")

    def _initialize_processors(self):
        """Initialize image processing capabilities."""
        self.supported_modalities = {
            "CT": {"bit_depth": 16, "default_window": (400, 40)},
            "MRI": {"bit_depth": 16, "default_window": (1000, 500)},
            "X-Ray": {"bit_depth": 16, "default_window": (4096, 2048)},
            "Ultrasound": {"bit_depth": 8, "default_window": (255, 128)},
            "OCT": {"bit_depth": 16, "default_window": (1000, 500)},
            "Mammography": {"bit_depth": 16, "default_window": (4096, 2048)},
        }

    def process_image(
        self, image: np.ndarray, metadata: MedicalImageMetadata
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process medical image with appropriate enhancements.

        Args:
            image: Input medical image
            metadata: Medical image metadata

        Returns:
            Processed image and processing statistics
        """
        start_time = datetime.now()

        # Validate input
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        # Apply modality-specific processing
        processed_image = self._apply_modality_processing(image, metadata)

        # Apply medical-grade enhancement
        enhanced_image = self._apply_medical_enhancement(processed_image, metadata)

        # Generate processing statistics
        processing_stats = {
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "input_shape": image.shape,
            "output_shape": enhanced_image.shape,
            "modality": metadata.modality,
            "enhancement_applied": True,
        }

        # Log processing for compliance
        if self.compliance_mode:
            self._log_processing_event(metadata, processing_stats)

        return enhanced_image, processing_stats

    def _apply_modality_processing(
        self, image: np.ndarray, metadata: MedicalImageMetadata
    ) -> np.ndarray:
        """Apply modality-specific processing."""
        modality = metadata.modality

        if modality not in self.supported_modalities:
            logger.warning(f"Unsupported modality {modality}, using default processing")
            return image

        config = self.supported_modalities[modality]

        # Apply windowing
        window_width, window_center = config["default_window"]
        windowed_image = self._apply_windowing(image, window_center, window_width)

        # Normalize to appropriate bit depth
        bit_depth = config["bit_depth"]
        normalized_image = self._normalize_to_bit_depth(windowed_image, bit_depth)

        return normalized_image

    def _apply_windowing(
        self, image: np.ndarray, center: float, width: float
    ) -> np.ndarray:
        """Apply medical imaging windowing."""
        min_val = center - width / 2
        max_val = center + width / 2

        windowed = np.clip(image, min_val, max_val)
        windowed = (windowed - min_val) / (max_val - min_val)

        return windowed

    def _normalize_to_bit_depth(self, image: np.ndarray, bit_depth: int) -> np.ndarray:
        """Normalize image to specific bit depth."""
        max_val = 2**bit_depth - 1
        normalized = (image * max_val).astype(np.uint16 if bit_depth > 8 else np.uint8)
        return normalized

    def _apply_medical_enhancement(
        self, image: np.ndarray, metadata: MedicalImageMetadata
    ) -> np.ndarray:
        """Apply medical-grade image enhancement."""
        enhanced = image.copy().astype(np.float32)

        # Apply noise reduction
        enhanced = self._reduce_noise(enhanced)

        # Apply contrast enhancement
        enhanced = self._enhance_contrast(enhanced)

        # Apply edge enhancement for better feature visibility
        enhanced = self._enhance_edges(enhanced)

        return enhanced

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply medical-grade noise reduction."""
        # Simple noise reduction using gaussian filtering
        try:
            from scipy import ndimage

            return ndimage.gaussian_filter(image, sigma=0.5)
        except ImportError:
            # Fallback to simple averaging
            kernel = np.ones((3, 3)) / 9
            return np.convolve(image.flatten(), kernel.flatten(), mode="same").reshape(
                image.shape
            )

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive contrast enhancement."""
        # Apply histogram equalization
        hist, bins = np.histogram(image.flatten(), 256, density=True)
        cdf = hist.cumsum()
        cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

        # Map original values to equalized values
        image_eq = np.interp(image.flatten(), bins[:-1], cdf)
        return image_eq.reshape(image.shape)

    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Apply edge enhancement for medical imaging."""
        # Simple edge enhancement using Laplacian
        try:
            from scipy import ndimage

            laplacian = ndimage.laplace(image)
            enhanced = image - 0.1 * laplacian
            return np.clip(enhanced, 0, 255)
        except ImportError:
            return image

    def _log_processing_event(
        self, metadata: MedicalImageMetadata, stats: Dict[str, Any]
    ):
        """Log processing event for medical compliance."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "patient_id": metadata.patient_id,
            "modality": metadata.modality,
            "processing_stats": stats,
            "processor_version": "1.0.0",
        }
        self.processing_history.append(event)


class DICOMProcessor:
    """
    DICOM image processor for medical imaging standards compliance.

    Handles DICOM file parsing, metadata extraction, and image conversion
    for use with the accelerator simulation framework.
    """

    def __init__(self):
        """Initialize DICOM processor."""
        self.supported_transfer_syntaxes = [
            "1.2.840.10008.1.2",  # Implicit VR Little Endian
            "1.2.840.10008.1.2.1",  # Explicit VR Little Endian
            "1.2.840.10008.1.2.2",  # Explicit VR Big Endian
        ]
        logger.info("DICOM processor initialized")

    def load_dicom(self, file_path: str) -> Tuple[np.ndarray, MedicalImageMetadata]:
        """
        Load DICOM file and extract image data and metadata.

        Args:
            file_path: Path to DICOM file

        Returns:
            Image array and metadata
        """
        try:
            # Try to use pydicom if available
            import pydicom

            ds = pydicom.dcmread(file_path)

            # Extract image data
            image = ds.pixel_array

            # Extract metadata
            metadata = MedicalImageMetadata(
                patient_id=getattr(ds, "PatientID", None),
                study_date=self._parse_dicom_date(getattr(ds, "StudyDate", None)),
                modality=getattr(ds, "Modality", "CT"),
                series_description=getattr(ds, "SeriesDescription", ""),
                image_dimensions=image.shape,
                pixel_spacing=getattr(ds, "PixelSpacing", [1.0, 1.0]),
                slice_thickness=getattr(ds, "SliceThickness", 1.0),
            )

            logger.info(f"Loaded DICOM image: {metadata.modality} {image.shape}")
            return image, metadata

        except ImportError:
            logger.warning("pydicom not available, using mock DICOM loader")
            return self._mock_dicom_load(file_path)

    def _parse_dicom_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse DICOM date string."""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            return None

    def _mock_dicom_load(
        self, file_path: str
    ) -> Tuple[np.ndarray, MedicalImageMetadata]:
        """Mock DICOM loader for testing when pydicom not available."""
        # Generate synthetic medical image
        image = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)

        metadata = MedicalImageMetadata(
            patient_id="MOCK_PATIENT_001",
            study_date=datetime.now(),
            modality="CT",
            series_description="Mock DICOM Series",
            image_dimensions=image.shape,
            pixel_spacing=(0.5, 0.5),
            slice_thickness=1.0,
        )

        logger.info("Generated mock DICOM data for testing")
        return image, metadata


class MedicalImageEnhancer:
    """
    Advanced medical image enhancement for improved diagnostic quality.

    Provides specialized enhancement algorithms optimized for different
    medical imaging modalities and diagnostic requirements.
    """

    def __init__(self):
        """Initialize medical image enhancer."""
        self.enhancement_profiles = {
            "CT": {"sharpen": 0.3, "contrast": 1.2, "noise_reduction": 0.5},
            "MRI": {"sharpen": 0.2, "contrast": 1.1, "noise_reduction": 0.7},
            "X-Ray": {"sharpen": 0.4, "contrast": 1.3, "noise_reduction": 0.3},
            "Ultrasound": {"sharpen": 0.1, "contrast": 1.0, "noise_reduction": 0.8},
        }
        logger.info("Medical image enhancer initialized")

    def enhance_image(self, image: np.ndarray, modality: str = "CT") -> np.ndarray:
        """
        Apply modality-specific enhancement to medical image.

        Args:
            image: Input medical image
            modality: Medical imaging modality

        Returns:
            Enhanced image
        """
        if modality not in self.enhancement_profiles:
            modality = "CT"  # Default fallback

        profile = self.enhancement_profiles[modality]
        enhanced = image.copy().astype(np.float32)

        # Apply noise reduction
        enhanced = self._apply_noise_reduction(enhanced, profile["noise_reduction"])

        # Apply contrast enhancement
        enhanced = self._apply_contrast_enhancement(enhanced, profile["contrast"])

        # Apply sharpening
        enhanced = self._apply_sharpening(enhanced, profile["sharpen"])

        # Normalize to original range
        enhanced = np.clip(enhanced, 0, np.max(image))

        return enhanced.astype(image.dtype)

    def _apply_noise_reduction(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply adaptive noise reduction."""
        # Simple bilateral-like filtering
        if strength <= 0:
            return image

        # Mock bilateral filtering
        smoothed = image.copy()
        for _ in range(int(strength * 3)):
            smoothed = 0.7 * smoothed + 0.3 * np.roll(smoothed, 1, axis=0)
            smoothed = 0.7 * smoothed + 0.3 * np.roll(smoothed, 1, axis=1)

        return smoothed

    def _apply_contrast_enhancement(
        self, image: np.ndarray, factor: float
    ) -> np.ndarray:
        """Apply adaptive contrast enhancement."""
        if factor == 1.0:
            return image

        mean_val = np.mean(image)
        enhanced = mean_val + factor * (image - mean_val)
        return enhanced

    def _apply_sharpening(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply adaptive sharpening."""
        if strength <= 0:
            return image

        # Simple unsharp mask
        blurred = self._apply_noise_reduction(image, 0.5)
        sharpened = image + strength * (image - blurred)
        return sharpened


class RadiologyWorkflow:
    """Specialized workflow for radiology image processing."""

    def __init__(self):
        """Initialize radiology workflow."""
        self.image_processor = MedicalImageProcessor()
        self.enhancer = MedicalImageEnhancer()
        logger.info("Radiology workflow initialized")

    def process_radiology_series(
        self, images: List[np.ndarray], metadata: MedicalImageMetadata
    ) -> Dict[str, Any]:
        """Process a complete radiology series."""
        results = {
            "processed_images": [],
            "processing_stats": [],
            "series_metadata": metadata,
            "workflow_version": "1.0.0",
        }

        for i, image in enumerate(images):
            # Process each image in the series
            processed_image, stats = self.image_processor.process_image(image, metadata)

            # Apply radiology-specific enhancement
            enhanced_image = self.enhancer.enhance_image(
                processed_image, metadata.modality
            )

            results["processed_images"].append(enhanced_image)
            results["processing_stats"].append(stats)

            logger.debug(f"Processed radiology image {i+1}/{len(images)}")

        return results


class PathologyWorkflow:
    """Specialized workflow for pathology image processing."""

    def __init__(self):
        """Initialize pathology workflow."""
        self.image_processor = MedicalImageProcessor()
        self.enhancer = MedicalImageEnhancer()
        logger.info("Pathology workflow initialized")

    def process_pathology_slide(
        self, image: np.ndarray, metadata: MedicalImageMetadata
    ) -> Dict[str, Any]:
        """Process pathology slide with specialized algorithms."""
        # Apply pathology-specific processing
        processed_image, stats = self.image_processor.process_image(image, metadata)

        # Apply color normalization for pathology
        color_normalized = self._normalize_pathology_colors(processed_image)

        # Apply pathology-specific enhancement
        enhanced_image = self.enhancer.enhance_image(color_normalized, "Pathology")

        results = {
            "processed_image": enhanced_image,
            "processing_stats": stats,
            "color_normalized": True,
            "workflow_version": "1.0.0",
        }

        return results

    def _normalize_pathology_colors(self, image: np.ndarray) -> np.ndarray:
        """Apply color normalization specific to pathology slides."""
        # Simplified color normalization
        if len(image.shape) == 3:  # Color image
            # Normalize each channel
            normalized = image.copy().astype(np.float32)
            for channel in range(image.shape[2]):
                channel_data = normalized[:, :, channel]
                channel_data = (channel_data - np.mean(channel_data)) / (
                    np.std(channel_data) + 1e-8
                )
                normalized[:, :, channel] = channel_data

            # Rescale to 0-255 range
            normalized = (
                (normalized - np.min(normalized))
                / (np.max(normalized) - np.min(normalized))
                * 255
            )
            return normalized.astype(np.uint8)

        return image


class DicomProcessor:
    """DICOM processor (alias for DICOMProcessor)."""

    def __init__(self):
        """Initialize DICOM processor."""
        self.dicom_processor = DICOMProcessor()

    def load_dicom(self, file_path: str) -> Tuple[np.ndarray, MedicalImageMetadata]:
        """Load DICOM file."""
        return self.dicom_processor.load_dicom(file_path)

    def save_dicom(
        self, image: np.ndarray, metadata: MedicalImageMetadata, file_path: str
    ) -> bool:
        """Save DICOM file."""
        return self.dicom_processor.save_dicom(image, metadata, file_path)


class NiftiProcessor:
    """NIfTI format processor for neuroimaging data."""

    def __init__(self):
        """Initialize NIfTI processor."""
        self.supported_extensions = [".nii", ".nii.gz"]
        logger.info("NIfTI processor initialized")

    def load_nifti(self, file_path: str) -> Tuple[np.ndarray, MedicalImageMetadata]:
        """
        Load NIfTI file and extract image data and metadata.

        Args:
            file_path: Path to NIfTI file

        Returns:
            Image array and metadata
        """
        try:
            # Try to use nibabel if available
            import nibabel as nib

            img = nib.load(file_path)

            # Extract image data
            image = img.get_fdata()

            # Extract metadata
            header = img.header
            metadata = MedicalImageMetadata(
                modality="MRI",  # NIfTI is typically MRI
                image_dimensions=image.shape,
                pixel_spacing=header.get_zooms()[:2]
                if len(header.get_zooms()) >= 2
                else (1.0, 1.0),
                slice_thickness=header.get_zooms()[2]
                if len(header.get_zooms()) >= 3
                else 1.0,
            )

            logger.info(f"Loaded NIfTI image: {image.shape}")
            return image, metadata

        except ImportError:
            logger.warning("nibabel not available, using mock NIfTI loader")
            return self._mock_nifti_load(file_path)

    def save_nifti(
        self, image: np.ndarray, metadata: MedicalImageMetadata, file_path: str
    ) -> bool:
        """
        Save image as NIfTI file.

        Args:
            image: Image array to save
            metadata: Medical image metadata
            file_path: Output file path

        Returns:
            Success status
        """
        try:
            import nibabel as nib

            # Create NIfTI image
            nifti_img = nib.Nifti1Image(image, affine=np.eye(4))

            # Save to file
            nib.save(nifti_img, file_path)
            logger.info(f"Saved NIfTI image to {file_path}")
            return True

        except ImportError:
            logger.warning("nibabel not available, NIfTI save not supported")
            return False
        except Exception as e:
            logger.error(f"Failed to save NIfTI file: {e}")
            return False

    def _mock_nifti_load(
        self, file_path: str
    ) -> Tuple[np.ndarray, MedicalImageMetadata]:
        """Mock NIfTI loader when nibabel is not available."""
        # Create mock 3D brain image
        image = np.random.rand(256, 256, 180).astype(np.float32)

        metadata = MedicalImageMetadata(
            modality="MRI",
            image_dimensions=image.shape,
            pixel_spacing=(1.0, 1.0),
            slice_thickness=1.0,
        )

        logger.info(f"Mock loaded NIfTI image: {image.shape}")
        return image, metadata


def create_medical_image_processor(
    modality: str = "CT", **kwargs
) -> MedicalImageProcessor:
    """
    Create a medical image processor configured for specific modality.

    Args:
        modality: Medical imaging modality
        **kwargs: Additional configuration parameters

    Returns:
        Configured medical image processor
    """
    config = {
        "enable_gpu": kwargs.get("enable_gpu", False),
        "compliance_mode": kwargs.get("compliance_mode", True),
        "modality": modality,
    }

    processor = MedicalImageProcessor(**config)
    logger.info(f"Created medical image processor for {modality}")
    return processor


def create_medical_imaging_pipeline(
    modality: str = "CT", enhancement_level: str = "standard"
) -> Dict[str, Any]:
    """
    Create a complete medical imaging pipeline.

    Args:
        modality: Medical imaging modality
        enhancement_level: Enhancement level (basic, standard, advanced)

    Returns:
        Configured medical imaging pipeline
    """
    pipeline = {
        "processor": MedicalImageProcessor(compliance_mode=True),
        "enhancer": MedicalImageEnhancer(),
        "dicom_processor": DICOMProcessor(),
        "modality": modality,
        "enhancement_level": enhancement_level,
    }

    # Add specialized workflows based on modality
    if modality in ["CT", "MRI", "X-Ray"]:
        pipeline["workflow"] = RadiologyWorkflow()
    elif modality in ["Pathology", "Histology"]:
        pipeline["workflow"] = PathologyWorkflow()

    logger.info(f"Created medical imaging pipeline for {modality}")
    return pipeline
