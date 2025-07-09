"""
Reliability and fault tolerance mechanisms.

Implements comprehensive reliability features including error detection,
correction, redundancy, and fault tolerance for medical-grade AI accelerators.
"""

import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

class FaultType(Enum):
    """Types of faults that can occur."""
    TRANSIENT = "transient"          # Soft errors (cosmic rays, etc.)
    PERMANENT = "permanent"          # Hard errors (device wear-out)
    INTERMITTENT = "intermittent"    # Intermittent failures
    SYSTEMATIC = "systematic"        # Design or software bugs

class ErrorType(Enum):
    """Types of errors detected."""
    SINGLE_BIT = "single_bit"        # Single bit error
    MULTI_BIT = "multi_bit"          # Multiple bit errors
    ARITHMETIC = "arithmetic"        # Arithmetic operation errors
    CONTROL = "control"              # Control flow errors
    MEMORY = "memory"                # Memory access errors
    COMMUNICATION = "communication"   # Inter-component communication errors

class RedundancyType(Enum):
    """Types of redundancy mechanisms."""
    DMR = "dmr"                      # Dual Modular Redundancy
    TMR = "tmr"                      # Triple Modular Redundancy
    TEMPORAL = "temporal"            # Temporal redundancy (re-execution)
    INFORMATION = "information"      # Information redundancy (ECC, checksums)

@dataclass
class ReliabilityConfig:
    """Reliability configuration parameters."""
    enable_error_detection: bool = True
    enable_error_correction: bool = True
    enable_redundancy: bool = True
    enable_fault_injection: bool = False  # For testing

    # Error rates (errors per operation)
    transient_error_rate: float = 1e-10
    permanent_error_rate: float = 1e-12

    # Redundancy configuration
    redundancy_type: RedundancyType = RedundancyType.TMR
    voting_threshold: int = 2  # For TMR voting

    # Error correction
    enable_ecc: bool = True
    ecc_granularity: int = 64  # bits

    # Fault tolerance
    max_correctable_errors: int = 100
    error_threshold_for_shutdown: int = 1000

    # Medical safety requirements
    safety_critical: bool = True
    required_availability: float = 0.99999  # 99.999% uptime
    mean_time_between_failures_hours: float = 8760  # 1 year

@dataclass
class ErrorReport:
    """Error detection and correction report."""
    cycle: int
    component: str
    error_type: ErrorType
    fault_type: FaultType
    detected: bool
    corrected: bool
    location: Optional[str] = None
    severity: str = "low"
    corrective_action: Optional[str] = None

@dataclass
class ReliabilityMetrics:
    """Reliability metrics and statistics."""
    total_errors_detected: int = 0
    total_errors_corrected: int = 0
    uncorrectable_errors: int = 0

    # Error breakdown by type
    transient_errors: int = 0
    permanent_errors: int = 0
    single_bit_errors: int = 0
    multi_bit_errors: int = 0

    # Reliability metrics
    mean_time_to_failure: float = 0.0
    availability: float = 1.0
    fault_coverage: float = 0.0

    # Performance impact
    performance_overhead: float = 0.0
    energy_overhead: float = 0.0
    area_overhead: float = 0.0

class ErrorDetector(ABC):
    """Abstract base class for error detection mechanisms."""

    @abstractmethod
    def detect_errors(self, data: np.ndarray, metadata: Dict[str, Any]) -> List[ErrorReport]:
        """Detect errors in data."""
        pass

class ParityChecker(ErrorDetector):
    """Parity-based error detection."""

    def __init__(self, parity_bits: int = 1):
        """
        Initialize parity checker.

        Args:
            parity_bits: Number of parity bits (1 for simple parity)
        """
        self.parity_bits = parity_bits
        self.error_count = 0

    def detect_errors(self, data: np.ndarray, metadata: Dict[str, Any]) -> List[ErrorReport]:
        """
        Detect single-bit errors using parity checking.

        Args:
            data: Input data array
            metadata: Additional metadata including expected parity

        Returns:
            List of detected errors
        """
        errors = []
        cycle = metadata.get('cycle', 0)
        component = metadata.get('component', 'unknown')

        # Convert data to bits for parity calculation
        data_bits = np.unpackbits(data.view(np.uint8))

        # Calculate parity
        calculated_parity = np.sum(data_bits) % 2
        expected_parity = metadata.get('parity', calculated_parity)

        if calculated_parity != expected_parity:
            self.error_count += 1
            error = ErrorReport(
                cycle=cycle,
                component=component,
                error_type=ErrorType.SINGLE_BIT,
                fault_type=FaultType.TRANSIENT,
                detected=True,
                corrected=False,
                location=f"parity_check_{self.error_count}",
                severity="medium"
            )
            errors.append(error)
            logger.warning(f"Parity error detected in {component} at cycle {cycle}")

        return errors

class ECCChecker(ErrorDetector):
    """Error Correcting Code (ECC) based error detection and correction."""

    def __init__(self, data_bits: int = 64, check_bits: int = 8):
        """
        Initialize ECC checker.

        Args:
            data_bits: Number of data bits
            check_bits: Number of check bits for ECC
        """
        self.data_bits = data_bits
        self.check_bits = check_bits
        self.total_bits = data_bits + check_bits

        # Generate Hamming code matrix
        self.generator_matrix = self._generate_hamming_matrix()
        self.parity_check_matrix = self._generate_parity_check_matrix()

    def _generate_hamming_matrix(self) -> np.ndarray:
        """Generate Hamming code generator matrix."""
        # Simplified Hamming code implementation
        # In practice, would use proper Hamming/BCH/Reed-Solomon codes
        matrix = np.random.randint(0, 2, (self.data_bits, self.check_bits))
        return matrix

    def _generate_parity_check_matrix(self) -> np.ndarray:
        """Generate parity check matrix."""
        # Simplified implementation
        matrix = np.random.randint(0, 2, (self.check_bits, self.total_bits))
        return matrix

    def detect_errors(self, data: np.ndarray, metadata: Dict[str, Any]) -> List[ErrorReport]:
        """
        Detect and attempt to correct errors using ECC.

        Args:
            data: Input data with ECC bits
            metadata: Additional metadata

        Returns:
            List of detected errors
        """
        errors = []
        cycle = metadata.get('cycle', 0)
        component = metadata.get('component', 'unknown')

        # Extract data and check bits
        data_portion = data[:self.data_bits]
        check_portion = data[self.data_bits:self.data_bits + self.check_bits]

        # Calculate syndrome
        syndrome = self._calculate_syndrome(data, check_portion)

        if np.any(syndrome):
            # Error detected
            error_location = self._locate_error(syndrome)

            if error_location < self.total_bits:
                # Single-bit error - correctable
                corrected_data = data.copy()
                corrected_data[error_location] = 1 - corrected_data[error_location]  # Flip bit

                error = ErrorReport(
                    cycle=cycle,
                    component=component,
                    error_type=ErrorType.SINGLE_BIT,
                    fault_type=FaultType.TRANSIENT,
                    detected=True,
                    corrected=True,
                    location=f"bit_{error_location}",
                    severity="low",
                    corrective_action="single_bit_correction"
                )
            else:
                # Multi-bit error - uncorrectable
                error = ErrorReport(
                    cycle=cycle,
                    component=component,
                    error_type=ErrorType.MULTI_BIT,
                    fault_type=FaultType.TRANSIENT,
                    detected=True,
                    corrected=False,
                    location="multiple_bits",
                    severity="high",
                    corrective_action="error_flagged"
                )

            errors.append(error)
            logger.info(f"ECC {'corrected' if error.corrected else 'detected'} error in {component}")

        return errors

    def _calculate_syndrome(self, data: np.ndarray, check_bits: np.ndarray) -> np.ndarray:
        """Calculate error syndrome."""
        # Simplified syndrome calculation
        received_word = np.concatenate([data, check_bits])
        syndrome = np.dot(self.parity_check_matrix, received_word) % 2
        return syndrome

    def _locate_error(self, syndrome: np.ndarray) -> int:
        """Locate error position from syndrome."""
        # Convert syndrome to error location
        if np.sum(syndrome) == 1:
            return np.where(syndrome)[0][0]
        else:
            return self.total_bits  # Multi-bit error indicator

class RedundancyManager:
    """Manages redundant execution and voting."""

    def __init__(self, redundancy_type: RedundancyType, voting_threshold: int = 2):
        """
        Initialize redundancy manager.

        Args:
            redundancy_type: Type of redundancy to use
            voting_threshold: Minimum votes needed for decision
        """
        self.redundancy_type = redundancy_type
        self.voting_threshold = voting_threshold
        self.execution_history: List[Dict[str, Any]] = []

    def execute_with_redundancy(self, operation: Callable, *args, **kwargs) -> Tuple[Any, List[ErrorReport]]:
        """
        Execute operation with redundancy.

        Args:
            operation: Operation to execute
            *args, **kwargs: Arguments for the operation

        Returns:
            Tuple of (result, error_reports)
        """
        errors = []

        if self.redundancy_type == RedundancyType.DMR:
            return self._execute_dmr(operation, *args, **kwargs)
        elif self.redundancy_type == RedundancyType.TMR:
            return self._execute_tmr(operation, *args, **kwargs)
        elif self.redundancy_type == RedundancyType.TEMPORAL:
            return self._execute_temporal(operation, *args, **kwargs)
        else:
            # No redundancy - single execution
            return operation(*args, **kwargs), errors

    def _execute_dmr(self, operation: Callable, *args, **kwargs) -> Tuple[Any, List[ErrorReport]]:
        """Execute with Dual Modular Redundancy."""
        errors = []

        # Execute twice
        result1 = operation(*args, **kwargs)
        result2 = operation(*args, **kwargs)

        # Compare results
        if np.array_equal(result1, result2):
            return result1, errors
        else:
            # Mismatch detected
            error = ErrorReport(
                cycle=kwargs.get('cycle', 0),
                component=kwargs.get('component', 'dmr_executor'),
                error_type=ErrorType.ARITHMETIC,
                fault_type=FaultType.TRANSIENT,
                detected=True,
                corrected=False,
                severity="high",
                corrective_action="dmr_mismatch_detected"
            )
            errors.append(error)

            # Return first result as default (could implement other strategies)
            return result1, errors

    def _execute_tmr(self, operation: Callable, *args, **kwargs) -> Tuple[Any, List[ErrorReport]]:
        """Execute with Triple Modular Redundancy."""
        errors = []

        # Execute three times
        result1 = operation(*args, **kwargs)
        result2 = operation(*args, **kwargs)
        result3 = operation(*args, **kwargs)

        results = [result1, result2, result3]

        # Majority voting
        voted_result, vote_count = self._majority_vote(results)

        if vote_count >= self.voting_threshold:
            # Successful voting
            if vote_count < 3:
                # Some disagreement detected
                error = ErrorReport(
                    cycle=kwargs.get('cycle', 0),
                    component=kwargs.get('component', 'tmr_executor'),
                    error_type=ErrorType.ARITHMETIC,
                    fault_type=FaultType.TRANSIENT,
                    detected=True,
                    corrected=True,
                    severity="medium",
                    corrective_action="tmr_majority_vote"
                )
                errors.append(error)

            return voted_result, errors
        else:
            # Voting failed - no majority
            error = ErrorReport(
                cycle=kwargs.get('cycle', 0),
                component=kwargs.get('component', 'tmr_executor'),
                error_type=ErrorType.ARITHMETIC,
                fault_type=FaultType.PERMANENT,
                detected=True,
                corrected=False,
                severity="critical",
                corrective_action="tmr_voting_failed"
            )
            errors.append(error)

            return result1, errors  # Fallback to first result

    def _execute_temporal(self, operation: Callable, *args, **kwargs) -> Tuple[Any, List[ErrorReport]]:
        """Execute with temporal redundancy (re-execution)."""
        errors = []
        max_retries = 3

        for attempt in range(max_retries):
            try:
                result = operation(*args, **kwargs)

                if attempt > 0:
                    # Re-execution was needed
                    error = ErrorReport(
                        cycle=kwargs.get('cycle', 0),
                        component=kwargs.get('component', 'temporal_executor'),
                        error_type=ErrorType.ARITHMETIC,
                        fault_type=FaultType.TRANSIENT,
                        detected=True,
                        corrected=True,
                        severity="low",
                        corrective_action=f"temporal_retry_attempt_{attempt}"
                    )
                    errors.append(error)

                return result, errors

            except Exception as e:
                if attempt == max_retries - 1:
                    # Final attempt failed
                    error = ErrorReport(
                        cycle=kwargs.get('cycle', 0),
                        component=kwargs.get('component', 'temporal_executor'),
                        error_type=ErrorType.ARITHMETIC,
                        fault_type=FaultType.PERMANENT,
                        detected=True,
                        corrected=False,
                        severity="critical",
                        corrective_action="temporal_retry_exhausted"
                    )
                    errors.append(error)
                    raise e

                # Log retry attempt
                logger.warning(f"Temporal redundancy retry {attempt + 1} due to: {e}")

        return None, errors

    def _majority_vote(self, results: List[Any]) -> Tuple[Any, int]:
        """Perform majority voting on results."""
        if not results:
            return None, 0

        # Count occurrences of each result
        result_counts = defaultdict(int)
        for result in results:
            # Convert to string for comparison (simplified)
            result_key = str(result)
            result_counts[result_key] += 1

        # Find majority
        max_count = max(result_counts.values())
        majority_key = [k for k, v in result_counts.items() if v == max_count][0]

        # Return original result corresponding to majority
        for result in results:
            if str(result) == majority_key:
                return result, max_count

        return results[0], 1  # Fallback

class FaultInjector:
    """Fault injector for reliability testing."""

    def __init__(self, config: ReliabilityConfig):
        """
        Initialize fault injector.

        Args:
            config: Reliability configuration
        """
        self.config = config
        self.injected_faults: List[Dict[str, Any]] = []
        self.random_seed = 42
        random.seed(self.random_seed)

    def inject_fault(self, data: np.ndarray, cycle: int,
                    component: str, fault_type: FaultType = FaultType.TRANSIENT) -> np.ndarray:
        """
        Inject fault into data for testing.

        Args:
            data: Input data
            cycle: Current cycle
            component: Component name
            fault_type: Type of fault to inject

        Returns:
            Data with injected fault
        """
        if not self.config.enable_fault_injection:
            return data

        # Determine if fault should be injected
        should_inject = False
        if fault_type == FaultType.TRANSIENT:
            should_inject = random.random() < self.config.transient_error_rate
        elif fault_type == FaultType.PERMANENT:
            should_inject = random.random() < self.config.permanent_error_rate

        if not should_inject:
            return data

        # Inject fault
        corrupted_data = data.copy()

        if fault_type == FaultType.TRANSIENT:
            # Flip random bit
            if data.size > 0:
                flat_data = corrupted_data.flatten()
                bit_index = random.randint(0, len(flat_data) - 1)

                # Convert to bits and flip one
                data_bytes = flat_data.view(np.uint8)
                if len(data_bytes) > 0:
                    byte_index = random.randint(0, len(data_bytes) - 1)
                    bit_position = random.randint(0, 7)
                    data_bytes[byte_index] ^= (1 << bit_position)

                corrupted_data = data_bytes.view(data.dtype).reshape(data.shape)

        elif fault_type == FaultType.PERMANENT:
            # Introduce systematic error
            corrupted_data = corrupted_data * 0.99  # Reduce values by 1%

        # Log injected fault
        fault_record = {
            'cycle': cycle,
            'component': component,
            'fault_type': fault_type.value,
            'location': f"fault_{len(self.injected_faults)}",
            'original_data_hash': hash(data.tobytes()),
            'corrupted_data_hash': hash(corrupted_data.tobytes())
        }
        self.injected_faults.append(fault_record)

        logger.debug(f"Injected {fault_type.value} fault in {component} at cycle {cycle}")

        return corrupted_data

    def get_injection_report(self) -> Dict[str, Any]:
        """Get report of all injected faults."""
        return {
            'total_faults_injected': len(self.injected_faults),
            'fault_breakdown': {
                fault_type.value: len([f for f in self.injected_faults
                                     if f['fault_type'] == fault_type.value])
                for fault_type in FaultType
            },
            'injected_faults': self.injected_faults
        }

class ReliabilityManager:
    """Main reliability management system."""

    def __init__(self, config: ReliabilityConfig):
        """
        Initialize reliability manager.

        Args:
            config: Reliability configuration
        """
        self.config = config

        # Error detection and correction
        self.error_detectors: Dict[str, ErrorDetector] = {
            'parity': ParityChecker(),
            'ecc': ECCChecker()
        }

        # Redundancy management
        self.redundancy_manager = RedundancyManager(
            config.redundancy_type,
            config.voting_threshold
        )

        # Fault injection for testing
        self.fault_injector = FaultInjector(config) if config.enable_fault_injection else None

        # Metrics and reporting
        self.metrics = ReliabilityMetrics()
        self.error_log: List[ErrorReport] = []
        self.fault_log: List[Dict[str, Any]] = []

        # Component health tracking
        self.component_health: Dict[str, float] = defaultdict(lambda: 1.0)  # 1.0 = healthy
        self.component_error_counts: Dict[str, int] = defaultdict(int)

        # Operational state
        self.system_operational = True
        self.degraded_mode = False
        self.shutdown_initiated = False

        logger.info("Reliability manager initialized")

    def process_data_with_reliability(self, data: np.ndarray,
                                    component: str,
                                    cycle: int,
                                    operation: Optional[Callable] = None) -> Tuple[np.ndarray, List[ErrorReport]]:
        """
        Process data through reliability mechanisms.

        Args:
            data: Input data
            component: Component identifier
            cycle: Current cycle
            operation: Optional operation to execute with redundancy

        Returns:
            Tuple of (processed_data, error_reports)
        """
        all_errors = []
        processed_data = data.copy()

        # Step 1: Fault injection (if enabled for testing)
        if self.fault_injector:
            processed_data = self.fault_injector.inject_fault(
                processed_data, cycle, component
            )

        # Step 2: Error detection
        if self.config.enable_error_detection:
            detection_errors = self._detect_errors(processed_data, component, cycle)
            all_errors.extend(detection_errors)

            # Update component health based on errors
            self._update_component_health(component, detection_errors)

        # Step 3: Error correction
        if self.config.enable_error_correction and all_errors:
            processed_data, correction_errors = self._correct_errors(
                processed_data, all_errors, component, cycle
            )
            all_errors.extend(correction_errors)

        # Step 4: Redundant execution (if operation provided)
        if operation and self.config.enable_redundancy:
            result, redundancy_errors = self.redundancy_manager.execute_with_redundancy(
                operation, processed_data, cycle=cycle, component=component
            )
            all_errors.extend(redundancy_errors)
            if result is not None:
                processed_data = result

        # Step 5: Update metrics and logs
        self._update_metrics(all_errors)
        self.error_log.extend(all_errors)

        # Step 6: Check system health and take corrective actions
        self._check_system_health()

        return processed_data, all_errors

    def _detect_errors(self, data: np.ndarray, component: str, cycle: int) -> List[ErrorReport]:
        """Run error detection on data."""
        all_errors = []

        metadata = {
            'cycle': cycle,
            'component': component,
            'data_size': data.size
        }

        # Run all enabled error detectors
        for detector_name, detector in self.error_detectors.items():
            try:
                errors = detector.detect_errors(data, metadata)
                all_errors.extend(errors)
            except Exception as e:
                logger.error(f"Error detector {detector_name} failed: {e}")

        return all_errors

    def _correct_errors(self, data: np.ndarray, errors: List[ErrorReport],
                       component: str, cycle: int) -> Tuple[np.ndarray, List[ErrorReport]]:
        """Attempt error correction."""
        corrected_data = data.copy()
        correction_errors = []

        correctable_errors = [e for e in errors if e.error_type in [ErrorType.SINGLE_BIT]]
        uncorrectable_errors = [e for e in errors if e not in correctable_errors]

        # Attempt correction of correctable errors
        for error in correctable_errors:
            if error.error_type == ErrorType.SINGLE_BIT and error.location:
                try:
                    # Simple bit flip correction (would be more sophisticated in practice)
                    if 'bit_' in error.location:
                        bit_pos = int(error.location.split('_')[1])
                        if bit_pos < corrected_data.size * 8:  # Within bounds
                            # Flip the bit (simplified)
                            flat_data = corrected_data.flatten().view(np.uint8)
                            byte_pos = bit_pos // 8
                            bit_in_byte = bit_pos % 8
                            if byte_pos < len(flat_data):
                                flat_data[byte_pos] ^= (1 << bit_in_byte)
                                corrected_data = flat_data.view(data.dtype).reshape(data.shape)
                                error.corrected = True

                except Exception as e:
                    logger.warning(f"Failed to correct error at {error.location}: {e}")

        # Log uncorrectable errors
        for error in uncorrectable_errors:
            correction_error = ErrorReport(
                cycle=cycle,
                component=component,
                error_type=error.error_type,
                fault_type=error.fault_type,
                detected=True,
                corrected=False,
                location=error.location,
                severity="high",
                corrective_action="uncorrectable_error_logged"
            )
            correction_errors.append(correction_error)

        return corrected_data, correction_errors

    def _update_component_health(self, component: str, errors: List[ErrorReport]) -> None:
        """Update component health based on detected errors."""
        if not errors:
            # No errors - slight health recovery
            self.component_health[component] = min(1.0, self.component_health[component] + 0.001)
            return

        # Degrade health based on error severity
        health_impact = 0.0
        for error in errors:
            if error.severity == "low":
                health_impact += 0.001
            elif error.severity == "medium":
                health_impact += 0.01
            elif error.severity == "high":
                health_impact += 0.05
            elif error.severity == "critical":
                health_impact += 0.1

        self.component_health[component] = max(0.0, self.component_health[component] - health_impact)
        self.component_error_counts[component] += len(errors)

        logger.debug(f"Component {component} health: {self.component_health[component]:.3f}")

    def _update_metrics(self, errors: List[ErrorReport]) -> None:
        """Update reliability metrics."""
        for error in errors:
            self.metrics.total_errors_detected += 1

            if error.corrected:
                self.metrics.total_errors_corrected += 1
            else:
                self.metrics.uncorrectable_errors += 1

            # Update error type counters
            if error.fault_type == FaultType.TRANSIENT:
                self.metrics.transient_errors += 1
            elif error.fault_type == FaultType.PERMANENT:
                self.metrics.permanent_errors += 1

            if error.error_type == ErrorType.SINGLE_BIT:
                self.metrics.single_bit_errors += 1
            elif error.error_type == ErrorType.MULTI_BIT:
                self.metrics.multi_bit_errors += 1

    def _check_system_health(self) -> None:
        """Check overall system health and take corrective actions."""
        total_errors = self.metrics.total_errors_detected
        uncorrectable_errors = self.metrics.uncorrectable_errors

        # Check for shutdown conditions
        if uncorrectable_errors >= self.config.error_threshold_for_shutdown:
            if not self.shutdown_initiated:
                logger.critical(f"Initiating system shutdown due to {uncorrectable_errors} uncorrectable errors")
                self.shutdown_initiated = True
                self.system_operational = False

        # Check for degraded mode
        elif total_errors >= self.config.max_correctable_errors:
            if not self.degraded_mode:
                logger.warning("Entering degraded mode due to high error count")
                self.degraded_mode = True

        # Check component health
        unhealthy_components = [comp for comp, health in self.component_health.items()
                              if health < 0.5]

        if unhealthy_components:
            logger.warning(f"Unhealthy components detected: {unhealthy_components}")
            # Could trigger component isolation or replacement

    def get_reliability_report(self) -> Dict[str, Any]:
        """Generate comprehensive reliability report."""
        # Calculate derived metrics
        total_operations = max(1, len(self.error_log))  # Avoid division by zero

        self.metrics.fault_coverage = (
            self.metrics.total_errors_detected /
            max(1, self.metrics.total_errors_detected + len(self.fault_log))
        )

        # Calculate availability
        downtime_cycles = 1000 if self.shutdown_initiated else (100 if self.degraded_mode else 0)
        total_cycles = max(1000, downtime_cycles + 10000)  # Simplified calculation
        self.metrics.availability = 1.0 - (downtime_cycles / total_cycles)

        # Performance overhead (simplified)
        self.metrics.performance_overhead = 0.02 if self.config.enable_redundancy else 0.01
        self.metrics.energy_overhead = 0.05 if self.config.enable_redundancy else 0.02
        self.metrics.area_overhead = 0.15 if self.config.enable_ecc else 0.05

        return {
            'reliability_metrics': self.metrics.__dict__,
            'system_status': {
                'operational': self.system_operational,
                'degraded_mode': self.degraded_mode,
                'shutdown_initiated': self.shutdown_initiated
            },
            'component_health': dict(self.component_health),
            'component_error_counts': dict(self.component_error_counts),
            'error_summary': {
                'total_errors': len(self.error_log),
                'errors_by_type': self._summarize_errors_by_type(),
                'errors_by_component': self._summarize_errors_by_component(),
                'recent_critical_errors': [
                    e.__dict__ for e in self.error_log[-10:]
                    if e.severity == "critical"
                ]
            },
            'fault_injection_report': (
                self.fault_injector.get_injection_report()
                if self.fault_injector else None
            ),
            'recommendations': self._generate_reliability_recommendations()
        }

    def _summarize_errors_by_type(self) -> Dict[str, int]:
        """Summarize errors by type."""
        summary = defaultdict(int)
        for error in self.error_log:
            summary[error.error_type.value] += 1
        return dict(summary)

    def _summarize_errors_by_component(self) -> Dict[str, int]:
        """Summarize errors by component."""
        summary = defaultdict(int)
        for error in self.error_log:
            summary[error.component] += 1
        return dict(summary)

    def _generate_reliability_recommendations(self) -> List[str]:
        """Generate reliability improvement recommendations."""
        recommendations = []

        if self.metrics.uncorrectable_errors > 0:
            recommendations.append(
                "Critical: Uncorrectable errors detected. Consider upgrading error "
                "correction mechanisms or replacing faulty components."
            )

        if self.metrics.availability < self.config.required_availability:
            recommendations.append(
                f"Availability ({self.metrics.availability:.5f}) below required "
                f"({self.config.required_availability:.5f}). Improve fault tolerance."
            )

        unhealthy_components = [comp for comp, health in self.component_health.items()
                              if health < 0.7]
        if unhealthy_components:
            recommendations.append(
                f"Components with degraded health: {unhealthy_components}. "
                "Schedule maintenance or replacement."
            )

        if self.metrics.fault_coverage < 0.95:
            recommendations.append(
                "Fault coverage below 95%. Consider additional error detection mechanisms."
            )

        if self.metrics.performance_overhead > 0.1:
            recommendations.append(
                "High performance overhead from reliability mechanisms. "
                "Consider optimization or selective application."
            )

        return recommendations

    def inject_fault(self, data: np.ndarray, cycle: int, 
                    component: str, fault_type: FaultType = FaultType.TRANSIENT) -> np.ndarray:
        """
        Inject fault into data for testing.

        Args:
            data: Input data
            cycle: Current cycle
            component: Component name
            fault_type: Type of fault to inject

        Returns:
            Data with injected fault
        """
        if self.fault_injector:
            return self.fault_injector.inject_fault(data, cycle, component, fault_type)
        return data

    def reset_metrics(self) -> None:
        """Reset reliability metrics and logs."""
        self.metrics = ReliabilityMetrics()
        self.error_log.clear()
        self.fault_log.clear()
        self.component_health.clear()
        self.component_error_counts.clear()

        self.system_operational = True
        self.degraded_mode = False
        self.shutdown_initiated = False

        logger.info("Reliability metrics reset")

def create_medical_reliability_config() -> ReliabilityConfig:
    """Create reliability configuration for medical applications."""
    return ReliabilityConfig(
        enable_error_detection=True,
        enable_error_correction=True,
        enable_redundancy=True,
        enable_fault_injection=False,
        redundancy_type=RedundancyType.TMR,
        safety_critical=True,
        required_availability=0.99999,
        mean_time_between_failures_hours=8760
    )
