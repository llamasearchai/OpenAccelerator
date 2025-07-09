"""
Advanced power management and energy optimization.

Implements sophisticated power modeling, DVFS (Dynamic Voltage and Frequency Scaling),
power gating, and energy optimization strategies for medical AI accelerators.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class PowerState(Enum):
    """Power states for components."""

    ACTIVE = "active"
    IDLE = "idle"
    SLEEP = "sleep"
    POWER_GATED = "power_gated"
    RETENTION = "retention"


class VoltageLevel(Enum):
    """Voltage levels for DVFS."""

    HIGH = "high"  # 1.2V - High performance
    NOMINAL = "nominal"  # 1.0V - Nominal operation
    LOW = "low"  # 0.8V - Low power
    ULTRA_LOW = "ultra_low"  # 0.6V - Ultra low power


class FrequencyLevel(Enum):
    """Frequency levels for DVFS."""

    HIGH = "high"  # 2.0 GHz
    NOMINAL = "nominal"  # 1.5 GHz
    LOW = "low"  # 1.0 GHz
    ULTRA_LOW = "ultra_low"  # 500 MHz


@dataclass
class PowerConfig:
    """Power management configuration."""

    enable_dvfs: bool = True
    enable_power_gating: bool = True
    enable_clock_gating: bool = True
    voltage_scaling_enabled: bool = True
    temperature_monitoring: bool = True

    # Power limits
    max_power_w: float = 300.0  # 300W TDP
    thermal_limit_c: float = 85.0  # 85°C thermal limit

    # DVFS parameters
    voltage_levels: dict[VoltageLevel, float] = field(
        default_factory=lambda: {
            VoltageLevel.HIGH: 1.2,
            VoltageLevel.NOMINAL: 1.0,
            VoltageLevel.LOW: 0.8,
            VoltageLevel.ULTRA_LOW: 0.6,
        }
    )

    frequency_levels: dict[FrequencyLevel, float] = field(
        default_factory=lambda: {
            FrequencyLevel.HIGH: 2.0e9,  # 2.0 GHz
            FrequencyLevel.NOMINAL: 1.5e9,  # 1.5 GHz
            FrequencyLevel.LOW: 1.0e9,  # 1.0 GHz
            FrequencyLevel.ULTRA_LOW: 0.5e9,  # 500 MHz
        }
    )

    # Power gating thresholds
    idle_threshold_cycles: int = 100
    sleep_threshold_cycles: int = 1000
    power_gate_threshold_cycles: int = 10000


@dataclass
class PowerMetrics:
    """Power consumption metrics."""

    dynamic_power_w: float = 0.0
    static_power_w: float = 0.0
    total_power_w: float = 0.0
    energy_consumed_j: float = 0.0
    peak_power_w: float = 0.0
    average_power_w: float = 0.0
    temperature_c: float = 25.0

    # Power breakdown by component
    compute_power_w: float = 0.0
    memory_power_w: float = 0.0
    interconnect_power_w: float = 0.0
    leakage_power_w: float = 0.0

    # Efficiency metrics
    performance_per_watt: float = 0.0
    energy_per_operation: float = 0.0
    thermal_design_power_utilized: float = 0.0


class PowerModel(ABC):
    """Abstract base class for power models."""

    @abstractmethod
    def calculate_dynamic_power(
        self, activity_factor: float, voltage: float, frequency: float
    ) -> float:
        """Calculate dynamic power consumption."""
        pass

    @abstractmethod
    def calculate_static_power(self, voltage: float, temperature: float) -> float:
        """Calculate static power consumption."""
        pass


class CMOSPowerModel(PowerModel):
    """CMOS power model for silicon accelerators."""

    def __init__(self, technology_node: int = 7):
        """
        Initialize CMOS power model.

        Args:
            technology_node: Technology node in nm (7, 14, 28, etc.)
        """
        self.technology_node = technology_node

        # Technology-dependent parameters
        self.capacitance_factor = self._get_capacitance_factor()
        self.leakage_coefficient = self._get_leakage_coefficient()
        self.temperature_coefficient = 0.02  # 2%/°C increase

    def _get_capacitance_factor(self) -> float:
        """Get capacitance factor based on technology node."""
        capacitance_map = {
            7: 1.0,  # 7nm baseline
            14: 1.5,  # 14nm
            28: 2.5,  # 28nm
            45: 4.0,  # 45nm
        }
        return capacitance_map.get(self.technology_node, 1.0)

    def _get_leakage_coefficient(self) -> float:
        """Get leakage coefficient based on technology node."""
        leakage_map = {
            7: 0.15,  # 7nm - higher leakage
            14: 0.10,  # 14nm
            28: 0.05,  # 28nm
            45: 0.02,  # 45nm - lower leakage
        }
        return leakage_map.get(self.technology_node, 0.10)

    def calculate_dynamic_power(
        self, activity_factor: float, voltage: float, frequency: float
    ) -> float:
        """
        Calculate dynamic power using P = α * C * V² * f

        Args:
            activity_factor: Switching activity factor (0-1)
            voltage: Supply voltage (V)
            frequency: Operating frequency (Hz)

        Returns:
            Dynamic power in watts
        """
        # Base capacitance in femtofarads (technology dependent)
        base_capacitance = 1000.0 * self.capacitance_factor  # fF

        # Convert to farads
        capacitance = base_capacitance * 1e-15

        # Dynamic power calculation
        dynamic_power = activity_factor * capacitance * (voltage**2) * frequency

        return dynamic_power

    def calculate_static_power(self, voltage: float, temperature: float) -> float:
        """
        Calculate static (leakage) power.

        Args:
            voltage: Supply voltage (V)
            temperature: Temperature in Celsius

        Returns:
            Static power in watts
        """
        # Base leakage current at nominal conditions
        base_leakage = 10e-9  # 10 nA base leakage

        # Voltage dependence (exponential)
        voltage_factor = np.exp((voltage - 1.0) / 0.1)  # Exponential voltage dependence

        # Temperature dependence
        temp_factor = np.exp((temperature - 25) * self.temperature_coefficient)

        # Technology scaling
        tech_factor = self.leakage_coefficient

        static_power = (
            base_leakage * voltage * voltage_factor * temp_factor * tech_factor
        )

        return static_power


class ComponentPowerManager:
    """Power manager for individual components."""

    def __init__(
        self, component_id: str, power_model: PowerModel, power_config: PowerConfig
    ):
        """
        Initialize component power manager.

        Args:
            component_id: Unique component identifier
            power_model: Power model to use
            power_config: Power configuration
        """
        self.component_id = component_id
        self.power_model = power_model
        self.config = power_config

        # Current state
        self.power_state = PowerState.ACTIVE
        self.voltage_level = VoltageLevel.NOMINAL
        self.frequency_level = FrequencyLevel.NOMINAL

        # Activity tracking
        self.activity_history: list[float] = []
        self.idle_cycles = 0
        self.total_cycles = 0

        # Power metrics
        self.power_metrics = PowerMetrics()
        self.power_trace: list[tuple[int, float]] = []  # (cycle, power)

        logger.debug(f"Initialized power manager for {component_id}")

    def update_activity(self, cycle: int, activity_factor: float) -> None:
        """
        Update component activity for power calculation.

        Args:
            cycle: Current simulation cycle
            activity_factor: Activity factor (0.0 = idle, 1.0 = full activity)
        """
        self.activity_history.append(activity_factor)
        self.total_cycles += 1

        # Track idle cycles for power gating decisions
        if activity_factor == 0.0:
            self.idle_cycles += 1
        else:
            self.idle_cycles = 0

        # Calculate current power
        current_voltage = self.config.voltage_levels[self.voltage_level]
        current_frequency = self.config.frequency_levels[self.frequency_level]

        # Only calculate power if component is active
        if self.power_state == PowerState.ACTIVE:
            dynamic_power = self.power_model.calculate_dynamic_power(
                activity_factor, current_voltage, current_frequency
            )
            static_power = self.power_model.calculate_static_power(
                current_voltage, self.power_metrics.temperature_c
            )
        elif self.power_state == PowerState.IDLE:
            dynamic_power = 0.0
            static_power = (
                self.power_model.calculate_static_power(
                    current_voltage, self.power_metrics.temperature_c
                )
                * 0.5
            )  # Reduced static power when idle
        else:  # SLEEP or POWER_GATED
            dynamic_power = 0.0
            static_power = (
                0.0
                if self.power_state == PowerState.POWER_GATED
                else self.power_model.calculate_static_power(
                    current_voltage, self.power_metrics.temperature_c
                )
                * 0.1
            )  # Minimal retention power

        total_power = dynamic_power + static_power

        # Update metrics
        self.power_metrics.dynamic_power_w = dynamic_power
        self.power_metrics.static_power_w = static_power
        self.power_metrics.total_power_w = total_power
        self.power_metrics.peak_power_w = max(
            self.power_metrics.peak_power_w, total_power
        )

        # Store power trace
        self.power_trace.append((cycle, total_power))

        # Make power management decisions
        self._make_power_decisions()

    def _make_power_decisions(self) -> None:
        """Make power management decisions based on activity."""
        if not self.config.enable_power_gating:
            return

        # Power gating decisions
        if self.idle_cycles >= self.config.power_gate_threshold_cycles:
            self._transition_to_power_state(PowerState.POWER_GATED)
        elif self.idle_cycles >= self.config.sleep_threshold_cycles:
            self._transition_to_power_state(PowerState.SLEEP)
        elif self.idle_cycles >= self.config.idle_threshold_cycles:
            self._transition_to_power_state(PowerState.IDLE)
        elif self.idle_cycles == 0:  # Active again
            self._transition_to_power_state(PowerState.ACTIVE)

    def _transition_to_power_state(self, new_state: PowerState) -> None:
        """Transition to new power state."""
        if self.power_state != new_state:
            logger.debug(
                f"{self.component_id}: {self.power_state.value} -> {new_state.value}"
            )
            self.power_state = new_state

    def set_dvfs_level(
        self, voltage_level: VoltageLevel, frequency_level: FrequencyLevel
    ) -> None:
        """
        Set DVFS (Dynamic Voltage and Frequency Scaling) level.

        Args:
            voltage_level: Target voltage level
            frequency_level: Target frequency level
        """
        if self.config.enable_dvfs:
            self.voltage_level = voltage_level
            self.frequency_level = frequency_level
            logger.debug(
                f"{self.component_id}: DVFS set to {voltage_level.value}/{frequency_level.value}"
            )

    def get_power_metrics(self) -> PowerMetrics:
        """Get current power metrics."""
        if len(self.power_trace) > 0:
            # Calculate average power
            total_energy = sum(power for _, power in self.power_trace)
            self.power_metrics.average_power_w = total_energy / len(self.power_trace)
            self.power_metrics.energy_consumed_j = (
                total_energy * 1e-9
            )  # Assuming 1ns cycles

        return self.power_metrics


class SystemPowerManager:
    """System-level power management."""

    def __init__(self, power_config: PowerConfig):
        """
        Initialize system power manager.

        Args:
            power_config: System power configuration
        """
        self.config = power_config
        self.component_managers: dict[str, ComponentPowerManager] = {}
        self.system_metrics = PowerMetrics()
        self.thermal_model = ThermalModel()  # Will be defined below

        # Power optimization
        self.workload_predictor = WorkloadPredictor()
        self.dvfs_controller = DVFSController(power_config)

        # System state
        self.current_cycle = 0
        self.power_budget_w = power_config.max_power_w
        self.thermal_throttling_active = False

        logger.info("Initialized system power manager")

    def register_component(
        self, component_id: str, power_model: PowerModel
    ) -> ComponentPowerManager:
        """
        Register a component for power management.

        Args:
            component_id: Unique component identifier
            power_model: Power model for the component

        Returns:
            Component power manager
        """
        manager = ComponentPowerManager(component_id, power_model, self.config)
        self.component_managers[component_id] = manager
        logger.debug(f"Registered component: {component_id}")
        return manager

    def update_system_power(
        self, cycle: int, component_activities: dict[str, float]
    ) -> None:
        """
        Update system-wide power consumption.

        Args:
            cycle: Current simulation cycle
            component_activities: Activity factors for each component
        """
        self.current_cycle = cycle

        # Update component power
        for component_id, activity in component_activities.items():
            if component_id in self.component_managers:
                self.component_managers[component_id].update_activity(cycle, activity)

        # Calculate system totals
        self._calculate_system_metrics()

        # Thermal management
        self._update_thermal_model()

        # Power optimization
        self._optimize_power_performance()

    def _calculate_system_metrics(self) -> None:
        """Calculate system-wide power metrics."""
        total_dynamic = 0.0
        total_static = 0.0
        total_power = 0.0
        peak_power = 0.0

        for manager in self.component_managers.values():
            metrics = manager.get_power_metrics()
            total_dynamic += metrics.dynamic_power_w
            total_static += metrics.static_power_w
            total_power += metrics.total_power_w
            peak_power = max(peak_power, metrics.peak_power_w)

        self.system_metrics.dynamic_power_w = total_dynamic
        self.system_metrics.static_power_w = total_static
        self.system_metrics.total_power_w = total_power
        self.system_metrics.peak_power_w = max(
            self.system_metrics.peak_power_w, peak_power
        )

        # Update running average
        if self.current_cycle > 0:
            alpha = 0.01  # Exponential moving average factor
            self.system_metrics.average_power_w = (
                (1 - alpha) * self.system_metrics.average_power_w + 
                alpha * total_power
            )

    def _update_thermal_model(self) -> None:
        """Update thermal model based on current power consumption."""
        # Simple thermal model - in reality this would be more complex
        ambient_temp = 25.0  # Celsius
        thermal_resistance = 0.1  # K/W
        
        # Calculate temperature rise
        temp_rise = self.system_metrics.total_power_w * thermal_resistance
        current_temp = ambient_temp + temp_rise
        
        # Update system metrics
        self.system_metrics.temperature_c = current_temp
        
        # Check for thermal throttling
        if current_temp > self.config.thermal_limit_c:
            self.thermal_throttling_active = True
            logger.warning(f"Thermal throttling activated at {current_temp:.1f}°C")
        else:
            self.thermal_throttling_active = False

    def _optimize_power_performance(self) -> None:
        """Optimize power and performance based on current conditions."""
        # Simple power optimization - in reality this would be more sophisticated
        if self.thermal_throttling_active:
            # Reduce power by lowering voltage/frequency
            for manager in self.component_managers.values():
                manager.set_dvfs_level(VoltageLevel.LOW, FrequencyLevel.LOW)
        elif self.system_metrics.total_power_w < self.power_budget_w * 0.8:
            # Increase performance if we have power budget
            for manager in self.component_managers.values():
                manager.set_dvfs_level(VoltageLevel.NOMINAL, FrequencyLevel.NOMINAL)

    def get_system_metrics(self) -> PowerMetrics:
        """Get system-wide power metrics."""
        return self.system_metrics


class ThermalModel:
    """Simple thermal model for temperature simulation."""
    
    def __init__(self):
        self.ambient_temperature = 25.0  # Celsius
        self.thermal_resistance = 0.1  # K/W
        self.thermal_capacitance = 100.0  # J/K
        self.current_temperature = self.ambient_temperature
        
    def update_temperature(self, power_w: float, dt: float = 1e-9) -> float:
        """Update temperature based on power consumption."""
        # Simple RC thermal model
        temp_rise = power_w * self.thermal_resistance
        target_temp = self.ambient_temperature + temp_rise
        
        # Exponential approach to target temperature
        time_constant = self.thermal_resistance * self.thermal_capacitance
        alpha = dt / time_constant
        
        self.current_temperature = (
            (1 - alpha) * self.current_temperature + 
            alpha * target_temp
        )
        
        return self.current_temperature


class WorkloadPredictor:
    """Workload predictor for power management."""
    
    def __init__(self):
        self.activity_history = []
        self.prediction_horizon = 100  # cycles
        
    def predict_activity(self, current_activity: float) -> float:
        """Predict future activity based on history."""
        self.activity_history.append(current_activity)
        
        # Keep only recent history
        if len(self.activity_history) > self.prediction_horizon:
            self.activity_history.pop(0)
            
        # Simple moving average prediction
        if len(self.activity_history) >= 10:
            return sum(self.activity_history[-10:]) / 10
        else:
            return current_activity


class DVFSController:
    """Dynamic Voltage and Frequency Scaling controller."""
    
    def __init__(self, power_config: PowerConfig):
        self.config = power_config
        self.current_voltage = VoltageLevel.NOMINAL
        self.current_frequency = FrequencyLevel.NOMINAL
        
    def adjust_dvfs(self, utilization: float, temperature: float) -> tuple[VoltageLevel, FrequencyLevel]:
        """Adjust DVFS based on utilization and temperature."""
        # Simple policy: reduce voltage/frequency if temperature is high
        if temperature > self.config.thermal_limit_c * 0.9:
            voltage = VoltageLevel.LOW
            frequency = FrequencyLevel.LOW
        elif utilization > 0.8:
            voltage = VoltageLevel.HIGH
            frequency = FrequencyLevel.HIGH
        elif utilization > 0.5:
            voltage = VoltageLevel.NOMINAL
            frequency = FrequencyLevel.NOMINAL
        else:
            voltage = VoltageLevel.LOW
            frequency = FrequencyLevel.LOW
            
        self.current_voltage = voltage
        self.current_frequency = frequency
        
        return voltage, frequency
