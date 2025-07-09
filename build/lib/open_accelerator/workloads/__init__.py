"""
Workload definitions and generators for the Open Accelerator simulator.

This module provides various ML workloads for evaluating accelerator performance
including computer vision, NLP, and autonomous driving workloads.
"""

from .autonomous_driving import *
from .base_workload import *
from .computer_vision import *
from .natural_language import *

__version__ = "0.1.0"

__all__ = [
    # Base workload exports
    "BaseWorkload",
    "WorkloadConfig",
    "WorkloadMetrics",
    "WorkloadGenerator",
    # Computer Vision workloads
    "ConvolutionWorkload",
    "ResNetWorkload",
    "EfficientNetWorkload",
    "VisionTransformerWorkload",
    "ImageClassificationSuite",
    "ObjectDetectionSuite",
    "SemanticSegmentationSuite",
    # Autonomous Driving workloads
    "PerceptionWorkload",
    "SensorFusionWorkload",
    "PathPlanningWorkload",
    "AutonomousDrivingSuite",
    "LidarProcessingWorkload",
    "CameraProcessingWorkload",
    # Natural Language workloads
    "TransformerWorkload",
    "BERTWorkload",
    "GPTWorkload",
    "AttentionWorkload",
    "LanguageModelSuite",
    "create_bert_workload",
    "create_gpt_workload",
]
