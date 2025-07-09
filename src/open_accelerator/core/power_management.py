"""
Advanced power management for AI accelerators.

Implements dynamic voltage and frequency scaling (DVFS), power gating,
clock gating, and thermal management for energy-efficient operation.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path  # <- needed for type annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from ..utils.config import AcceleratorConfig

logger = logging.getLogger(__name__)


class PowerState(Enum):
    """Power states for accelerator components."""

    ACTIVE = "active"
    IDLE = "idle"
    SLEEP = "sleep"
    DEEP_SLEEP = "deep_sleep"
    POWER_OFF = "power_off"


class VoltageLevel(Enum):
    """Voltage levels for DVFS."""

    HIGH = "high"  # 1.2V - High performance
    NOMINAL = "nominal"  # 1.0V - Standard operation
    LOW = "low"  # 0.8V - Low power
    ULTRA_LOW = "ultra_low"  # 0.6V - Minimum operation


class FrequencyLevel(Enum):
    """Frequency levels for DVFS."""

    HIGH = "high"  # 1000 MHz
    NOMINAL = "nominal"  # 800 MHz
    LOW = "low"  # 600 MHz
    ULTRA_LOW = "ultra_low"  # 400 MHz


class ThermalState(Enum):
    """Thermal management states."""

    NORMAL = "normal"  # < 70°C
    WARM = "warm"  # 70-80°C
    HOT = "hot"  # 80-90°C
    CRITICAL = "critical"  # > 90°C


@dataclass
class PowerConfig:
    """Power management configuration."""

    enable_dvfs: bool = True
    enable_power_gating: bool = True
    enable_clock_gating: bool = True
    enable_thermal_management: bool = True

    # DVFS settings
    default_voltage: VoltageLevel = VoltageLevel.NOMINAL
    default_frequency: FrequencyLevel = FrequencyLevel.NOMINAL
    dvfs_reaction_time_ms: float = 10.0

    # Power gating settings
    power_gate_threshold_idle_cycles: int = 1000
    power_gate_wake_penalty_cycles: int = 100

    # Clock gating settings
    clock_gate_threshold_idle_cycles: int = 10
    clock_gate_wake_penalty_cycles: int = 1

    # Thermal settings
    thermal_target_temp_c: float = 75.0
    thermal_critical_temp_c: float = 90.0
    thermal_throttle_temp_c: float = 85.0
    thermal_polling_interval_ms: float = 100.0

    # Power budget settings
    max_power_budget_watts: float = 300.0
    power_budget_margin_percent: float = 10.0

    # Energy efficiency targets
    target_energy_efficiency_tops_per_watt: float = 50.0
    min_performance_threshold_percent: float = 80.0


@dataclass
class PowerMetrics:
    """Power consumption and efficiency metrics."""

    # Current measurements
    current_power_watts: float = 0.0
    current_voltage: float = 1.0
    current_frequency_mhz: float = 800.0
    current_temperature_c: float = 25.0

    # Cumulative measurements
    total_energy_joules: float = 0.0
    active_time_seconds: float = 0.0
    idle_time_seconds: float = 0.0
    sleep_time_seconds: float = 0.0

    # Efficiency metrics
    energy_efficiency_tops_per_watt: float = 0.0
    power_utilization_percent: float = 0.0
    thermal_utilization_percent: float = 0.0

    # DVFS statistics
    dvfs_transitions: int = 0
    power_gate_events: int = 0
    clock_gate_events: int = 0
    thermal_throttle_events: int = 0

    # Component power breakdown
    systolic_array_power_watts: float = 0.0
    memory_power_watts: float = 0.0
    control_power_watts: float = 0.0
    io_power_watts: float = 0.0


@dataclass
class ComponentPowerProfile:
    """Power profile for an accelerator component."""

    component_name: str
    base_power_watts: float  # Static power consumption
    dynamic_power_factor: float  # Power scaling with activity
    voltage_sensitivity: float  # Power scaling with voltage
    frequency_sensitivity: float  # Power scaling with frequency
    max_power_watts: float  # Maximum power consumption

    # Power states
    active_power_multiplier: float = 1.0
    idle_power_multiplier: float = 0.3
    sleep_power_multiplier: float = 0.1
    deep_sleep_power_multiplier: float = 0.05


class PowerModel(ABC):
    """Abstract base class for power modeling."""

    @abstractmethod
    def calculate_power(
        self, utilization: float, voltage: float, frequency: float, state: PowerState
    ) -> float:
        """Calculate power consumption for given parameters."""
        pass


class SimplePowerModel(PowerModel):
    """Simple power model based on utilization and DVFS parameters."""

    def __init__(self, profile: ComponentPowerProfile):
        """
        Initialize simple power model.

        Args:
            profile: Component power profile
        """
        self.profile = profile

    def calculate_power(
        self, utilization: float, voltage: float, frequency: float, state: PowerState
    ) -> float:
        """
        Calculate power using simple model: P = P_static + P_dynamic * utilization * V² * f

        Args:
            utilization: Component utilization (0.0 to 1.0)
            voltage: Operating voltage
            frequency: Operating frequency in MHz
            state: Current power state

        Returns:
            Power consumption in watts
        """
        # Base static power
        static_power = self.profile.base_power_watts

        # Dynamic power based on utilization
        dynamic_power = (
            self.profile.dynamic_power_factor
            * utilization
            * (voltage**self.profile.voltage_sensitivity)
            * (frequency / 1000.0) ** self.profile.frequency_sensitivity
        )

        total_power = static_power + dynamic_power

        # Apply power state multiplier
        if state == PowerState.ACTIVE:
            multiplier = self.profile.active_power_multiplier
        elif state == PowerState.IDLE:
            multiplier = self.profile.idle_power_multiplier
        elif state == PowerState.SLEEP:
            multiplier = self.profile.sleep_power_multiplier
        elif state == PowerState.DEEP_SLEEP:
            multiplier = self.profile.deep_sleep_power_multiplier
        else:  # POWER_OFF
            return 0.0

        final_power = total_power * multiplier

        # Clamp to maximum power
        return min(final_power, self.profile.max_power_watts)


class DVFSController:
    """Dynamic Voltage and Frequency Scaling controller."""

    def __init__(self, config: PowerConfig):
        """
        Initialize DVFS controller.

        Args:
            config: Power management configuration
        """
        self.config = config

        # Voltage and frequency mappings
        self.voltage_levels = {
            VoltageLevel.HIGH: 1.2,
            VoltageLevel.NOMINAL: 1.0,
            VoltageLevel.LOW: 0.8,
            VoltageLevel.ULTRA_LOW: 0.6,
        }

        self.frequency_levels = {
            FrequencyLevel.HIGH: 1000.0,
            FrequencyLevel.NOMINAL: 800.0,
            FrequencyLevel.LOW: 600.0,
            FrequencyLevel.ULTRA_LOW: 400.0,
        }

        # Current operating point
        self.current_voltage_level = config.default_voltage
        self.current_frequency_level = config.default_frequency

        # Performance tracking
        self.utilization_history: List[float] = []
        self.power_history: List[float] = []
        self.performance_history: List[float] = []

        # DVFS decision parameters
        self.utilization_high_threshold = 0.8
        self.utilization_low_threshold = 0.3
        self.power_high_threshold = 0.9  # 90% of power budget
        self.performance_low_threshold = 0.7  # 70% of target performance

        self.last_dvfs_time = 0.0
        self.transition_count = 0

        logger.info("DVFS controller initialized")

    def update_operating_point(
        self,
        utilization: float,
        power_consumption: float,
        performance: float,
        temperature: float,
    ) -> Tuple[VoltageLevel, FrequencyLevel]:
        """
        Update voltage and frequency based on system metrics.

        Args:
            utilization: Current system utilization (0.0 to 1.0)
            power_consumption: Current power consumption as fraction of budget
            performance: Current performance as fraction of target
            temperature: Current temperature in Celsius

        Returns:
            Tuple of (new_voltage_level, new_frequency_level)
        """
        current_time = time.time() * 1000  # Convert to milliseconds

        # Check if enough time has passed since last DVFS transition
        if (current_time - self.last_dvfs_time) < self.config.dvfs_reaction_time_ms:
            return self.current_voltage_level, self.current_frequency_level

        # Record metrics
        self.utilization_history.append(utilization)
        self.power_history.append(power_consumption)
        self.performance_history.append(performance)

        # Keep only recent history (last 10 samples)
        if len(self.utilization_history) > 10:
            self.utilization_history.pop(0)
            self.power_history.pop(0)
            self.performance_history.pop(0)

        # Calculate average metrics
        avg_utilization = np.mean(self.utilization_history)
        avg_power = np.mean(self.power_history)
        avg_performance = np.mean(self.performance_history)

        # Determine new operating point
        new_voltage = self.current_voltage_level
        new_frequency = self.current_frequency_level

        # Scale up if high utilization or low performance
        if (
            avg_utilization > self.utilization_high_threshold
            or avg_performance < self.performance_low_threshold
        ):
            # Check if we can increase without exceeding thermal or power limits
            if (
                temperature < self.config.thermal_throttle_temp_c
                and avg_power < self.power_high_threshold
            ):
                new_voltage, new_frequency = self._scale_up(new_voltage, new_frequency)

        # Scale down if low utilization and power/thermal pressure
        elif (
            avg_utilization < self.utilization_low_threshold
            or avg_power > self.power_high_threshold
            or temperature > self.config.thermal_target_temp_c
        ):
            new_voltage, new_frequency = self._scale_down(new_voltage, new_frequency)

        # Apply changes if different from current
        if (
            new_voltage != self.current_voltage_level
            or new_frequency != self.current_frequency_level
        ):
            self.current_voltage_level = new_voltage
            self.current_frequency_level = new_frequency
            self.last_dvfs_time = current_time
            self.transition_count += 1

            logger.debug(
                f"DVFS transition to {new_voltage.value}V, {new_frequency.value}MHz"
            )

        return new_voltage, new_frequency

    def _scale_up(
        self, voltage: VoltageLevel, frequency: FrequencyLevel
    ) -> Tuple[VoltageLevel, FrequencyLevel]:
        """Scale up voltage/frequency for higher performance."""
        # Prioritize frequency scaling first (better energy efficiency)
        if frequency == FrequencyLevel.ULTRA_LOW:
            return voltage, FrequencyLevel.LOW
        elif frequency == FrequencyLevel.LOW:
            return voltage, FrequencyLevel.NOMINAL
        elif frequency == FrequencyLevel.NOMINAL:
            return voltage, FrequencyLevel.HIGH

        # Then scale voltage if frequency is already at maximum
        elif voltage == VoltageLevel.ULTRA_LOW:
            return VoltageLevel.LOW, frequency
        elif voltage == VoltageLevel.LOW:
            return VoltageLevel.NOMINAL, frequency
        elif voltage == VoltageLevel.NOMINAL:
            return VoltageLevel.HIGH, frequency

        # Already at maximum
        return voltage, frequency

    def _scale_down(
        self, voltage: VoltageLevel, frequency: FrequencyLevel
    ) -> Tuple[VoltageLevel, FrequencyLevel]:
        """Scale down voltage/frequency for lower power."""
        # Prioritize voltage scaling first (quadratic power reduction)
        if voltage == VoltageLevel.HIGH:
            return VoltageLevel.NOMINAL, frequency
        elif voltage == VoltageLevel.NOMINAL:
            return VoltageLevel.LOW, frequency
        elif voltage == VoltageLevel.LOW:
            return VoltageLevel.ULTRA_LOW, frequency

        # Then scale frequency if voltage is already at minimum
        elif frequency == FrequencyLevel.HIGH:
            return voltage, FrequencyLevel.NOMINAL
        elif frequency == FrequencyLevel.NOMINAL:
            return voltage, FrequencyLevel.LOW
        elif frequency == FrequencyLevel.LOW:
            return voltage, FrequencyLevel.ULTRA_LOW

        # Already at minimum
        return voltage, frequency

    def get_current_voltage(self) -> float:
        """Get current voltage in volts."""
        return self.voltage_levels[self.current_voltage_level]

    def get_current_frequency(self) -> float:
        """Get current frequency in MHz."""
        return self.frequency_levels[self.current_frequency_level]

    def get_dvfs_metrics(self) -> Dict[str, Any]:
        """Get DVFS performance metrics."""
        return {
            "current_voltage_level": self.current_voltage_level.value,
            "current_frequency_level": self.current_frequency_level.value,
            "current_voltage_v": self.get_current_voltage(),
            "current_frequency_mhz": self.get_current_frequency(),
            "transition_count": self.transition_count,
            "avg_utilization": (
                np.mean(self.utilization_history) if self.utilization_history else 0.0
            ),
            "avg_power_fraction": (
                np.mean(self.power_history) if self.power_history else 0.0
            ),
            "avg_performance_fraction": (
                np.mean(self.performance_history) if self.performance_history else 0.0
            ),
        }


class ThermalManager:
    """Thermal management and throttling system."""

    def __init__(self, config: PowerConfig):
        """
        Initialize thermal manager.

        Args:
            config: Power management configuration
        """
        self.config = config
        self.current_temperature = 25.0  # Ambient temperature
        self.thermal_state = ThermalState.NORMAL

        # Thermal control parameters
        self.thermal_time_constant = 5.0  # Seconds for thermal response
        self.ambient_temperature = 25.0
        self.thermal_resistance = 0.5  # °C/W

        # Throttling history
        self.throttle_events = []
        self.temperature_history = []

        logger.info("Thermal manager initialized")

    def update_temperature(
        self, power_consumption: float, ambient_temp: Optional[float] = None
    ) -> float:
        """
        Update chip temperature based on power consumption.

        Args:
            power_consumption: Current power consumption in watts
            ambient_temp: Ambient temperature (optional)

        Returns:
            Updated temperature in Celsius
        """
        if ambient_temp is not None:
            self.ambient_temperature = ambient_temp

        # Simple thermal model: T = T_ambient + P * R_thermal
        target_temperature = self.ambient_temperature + (
            power_consumption * self.thermal_resistance
        )

        # Apply thermal time constant (first-order response)
        dt = 0.1  # Assume 100ms update interval
        alpha = dt / (self.thermal_time_constant + dt)
        self.current_temperature = (
            alpha * target_temperature + (1 - alpha) * self.current_temperature
        )

        # Update thermal state
        self._update_thermal_state()

        # Record temperature
        self.temperature_history.append(self.current_temperature)
        if len(self.temperature_history) > 100:  # Keep last 100 samples
            self.temperature_history.pop(0)

        return self.current_temperature

    def _update_thermal_state(self):
        """Update thermal state based on current temperature."""
        temp = self.current_temperature

        if temp >= self.config.thermal_critical_temp_c:
            self.thermal_state = ThermalState.CRITICAL
        elif temp >= self.config.thermal_throttle_temp_c:
            self.thermal_state = ThermalState.HOT
        elif temp >= self.config.thermal_target_temp_c:
            self.thermal_state = ThermalState.WARM
        else:
            self.thermal_state = ThermalState.NORMAL

    def should_throttle(self) -> bool:
        """Check if thermal throttling should be applied."""
        return self.thermal_state in [ThermalState.HOT, ThermalState.CRITICAL]

    def get_throttle_factor(self) -> float:
        """
        Get throttling factor for performance reduction.

        Returns:
            Throttling factor (0.0 to 1.0, where 1.0 = no throttling)
        """
        if self.thermal_state == ThermalState.CRITICAL:
            # Aggressive throttling for critical temperature
            throttle_factor = 0.5
        elif self.thermal_state == ThermalState.HOT:
            # Moderate throttling for hot temperature
            temp_excess = self.current_temperature - self.config.thermal_throttle_temp_c
            temp_range = (
                self.config.thermal_critical_temp_c
                - self.config.thermal_throttle_temp_c
            )
            throttle_factor = max(0.5, 1.0 - (temp_excess / temp_range) * 0.5)
        else:
            throttle_factor = 1.0

        # Record throttle event if throttling is active
        if throttle_factor < 1.0:
            self.throttle_events.append(
                {
                    "timestamp": time.time(),
                    "temperature": self.current_temperature,
                    "throttle_factor": throttle_factor,
                    "thermal_state": self.thermal_state.value,
                }
            )

        return throttle_factor

    def get_thermal_metrics(self) -> Dict[str, Any]:
        """Get thermal management metrics."""
        return {
            "current_temperature_c": self.current_temperature,
            "thermal_state": self.thermal_state.value,
            "ambient_temperature_c": self.ambient_temperature,
            "thermal_utilization_percent": min(
                100.0,
                (self.current_temperature - self.ambient_temperature)
                / (self.config.thermal_critical_temp_c - self.ambient_temperature)
                * 100,
            ),
            "throttle_events_count": len(self.throttle_events),
            "recent_throttle_events": (
                self.throttle_events[-5:] if self.throttle_events else []
            ),
            "avg_temperature_c": (
                np.mean(self.temperature_history)
                if self.temperature_history
                else self.current_temperature
            ),
            "max_temperature_c": (
                max(self.temperature_history)
                if self.temperature_history
                else self.current_temperature
            ),
        }


class PowerGatingController:
    """Power gating controller for unused components."""

    def __init__(self, config: PowerConfig):
        """
        Initialize power gating controller.

        Args:
            config: Power management configuration
        """
        self.config = config
        self.component_states: Dict[str, PowerState] = {}
        self.idle_counters: Dict[str, int] = {}
        self.gate_events: List[Dict[str, Any]] = []

        logger.info("Power gating controller initialized")

    def register_component(self, component_name: str):
        """Register a component for power gating control."""
        self.component_states[component_name] = PowerState.ACTIVE
        self.idle_counters[component_name] = 0
        logger.debug(f"Registered component {component_name} for power gating")

    def update_component_activity(self, component_name: str, is_active: bool):
        """
        Update component activity status.

        Args:
            component_name: Name of the component
            is_active: Whether the component is currently active
        """
        if component_name not in self.component_states:
            self.register_component(component_name)

        current_state = self.component_states[component_name]

        if is_active:
            # Component is active
            self.idle_counters[component_name] = 0

            # Wake up component if it was gated
            if current_state in [PowerState.SLEEP, PowerState.DEEP_SLEEP]:
                self._wake_component(component_name)
        else:
            # Component is idle
            self.idle_counters[component_name] += 1

            # Check if we should gate the component
            if (
                current_state == PowerState.ACTIVE
                and self.idle_counters[component_name]
                >= self.config.power_gate_threshold_idle_cycles
            ):
                self._gate_component(component_name)

    def _gate_component(self, component_name: str):
        """Gate power to an idle component."""
        if not self.config.enable_power_gating:
            return

        # Determine gating level based on idle time
        idle_cycles = self.idle_counters[component_name]

        if idle_cycles >= self.config.power_gate_threshold_idle_cycles * 10:
            # Very long idle - deep sleep
            new_state = PowerState.DEEP_SLEEP
        else:
            # Moderate idle - light sleep
            new_state = PowerState.SLEEP

        old_state = self.component_states[component_name]
        self.component_states[component_name] = new_state

        # Record gate event
        self.gate_events.append(
            {
                "timestamp": time.time(),
                "component": component_name,
                "action": "gate",
                "old_state": old_state.value,
                "new_state": new_state.value,
                "idle_cycles": idle_cycles,
            }
        )

        logger.debug(f"Power gated component {component_name} to {new_state.value}")

    def _wake_component(self, component_name: str):
        """Wake up a power-gated component."""
        old_state = self.component_states[component_name]
        self.component_states[component_name] = PowerState.ACTIVE

        # Record wake event
        self.gate_events.append(
            {
                "timestamp": time.time(),
                "component": component_name,
                "action": "wake",
                "old_state": old_state.value,
                "new_state": PowerState.ACTIVE.value,
                "wake_penalty_cycles": self.config.power_gate_wake_penalty_cycles,
            }
        )

        logger.debug(f"Woke up component {component_name} from {old_state.value}")

    def get_component_state(self, component_name: str) -> PowerState:
        """Get current power state of a component."""
        return self.component_states.get(component_name, PowerState.ACTIVE)

    def get_wake_penalty(self, component_name: str) -> int:
        """Get wake penalty cycles for a component."""
        state = self.get_component_state(component_name)
        if state == PowerState.DEEP_SLEEP:
            return self.config.power_gate_wake_penalty_cycles * 2
        elif state == PowerState.SLEEP:
            return self.config.power_gate_wake_penalty_cycles
        else:
            return 0

    def get_power_gating_metrics(self) -> Dict[str, Any]:
        """Get power gating metrics."""
        total_events = len(self.gate_events)
        gate_events = [e for e in self.gate_events if e["action"] == "gate"]
        wake_events = [e for e in self.gate_events if e["action"] == "wake"]

        return {
            "total_events": total_events,
            "gate_events": len(gate_events),
            "wake_events": len(wake_events),
            "component_states": {
                name: state.value for name, state in self.component_states.items()
            },
            "idle_counters": self.idle_counters.copy(),
            "recent_events": self.gate_events[-10:] if self.gate_events else [],
        }


class ClockGatingController:
    """Clock gating controller for fine-grained power management."""

    def __init__(self, config: PowerConfig):
        """
        Initialize clock gating controller.

        Args:
            config: Power management configuration
        """
        self.config = config
        self.clock_enabled: Dict[str, bool] = {}
        self.idle_counters: Dict[str, int] = {}
        self.gate_events: List[Dict[str, Any]] = []

        logger.info("Clock gating controller initialized")

    def register_clock_domain(self, domain_name: str):
        """Register a clock domain for gating control."""
        self.clock_enabled[domain_name] = True
        self.idle_counters[domain_name] = 0
        logger.debug(f"Registered clock domain {domain_name}")

    def update_domain_activity(self, domain_name: str, is_active: bool):
        """
        Update clock domain activity.

        Args:
            domain_name: Name of the clock domain
            is_active: Whether the domain is currently active
        """
        if domain_name not in self.clock_enabled:
            self.register_clock_domain(domain_name)

        if is_active:
            # Domain is active
            self.idle_counters[domain_name] = 0

            # Enable clock if it was gated
            if not self.clock_enabled[domain_name]:
                self._enable_clock(domain_name)
        else:
            # Domain is idle
            self.idle_counters[domain_name] += 1

            # Check if we should gate the clock
            if (
                self.clock_enabled[domain_name]
                and self.idle_counters[domain_name]
                >= self.config.clock_gate_threshold_idle_cycles
            ):
                self._gate_clock(domain_name)

    def _gate_clock(self, domain_name: str):
        """Gate clock to an idle domain."""
        if not self.config.enable_clock_gating:
            return

        self.clock_enabled[domain_name] = False

        # Record gate event
        self.gate_events.append(
            {
                "timestamp": time.time(),
                "domain": domain_name,
                "action": "gate",
                "idle_cycles": self.idle_counters[domain_name],
            }
        )

        logger.debug(f"Clock gated domain {domain_name}")

    def _enable_clock(self, domain_name: str):
        """Enable clock for a domain."""
        self.clock_enabled[domain_name] = True

        # Record enable event
        self.gate_events.append(
            {
                "timestamp": time.time(),
                "domain": domain_name,
                "action": "enable",
                "wake_penalty_cycles": self.config.clock_gate_wake_penalty_cycles,
            }
        )

        logger.debug(f"Clock enabled for domain {domain_name}")

    def is_clock_enabled(self, domain_name: str) -> bool:
        """Check if clock is enabled for a domain."""
        return self.clock_enabled.get(domain_name, True)

    def get_clock_gating_metrics(self) -> Dict[str, Any]:
        """Get clock gating metrics."""
        total_events = len(self.gate_events)
        gate_events = [e for e in self.gate_events if e["action"] == "gate"]
        enable_events = [e for e in self.gate_events if e["action"] == "enable"]

        return {
            "total_events": total_events,
            "gate_events": len(gate_events),
            "enable_events": len(enable_events),
            "domain_states": self.clock_enabled.copy(),
            "idle_counters": self.idle_counters.copy(),
            "recent_events": self.gate_events[-10:] if self.gate_events else [],
        }


class PowerManager:
    """Main power management system coordinating all power management components."""

    def __init__(self, config: PowerConfig):
        """
        Initialize power manager.

        Args:
            config: Power management configuration
        """
        self.config = config
        self.metrics = PowerMetrics()

        # Initialize power management components
        self.dvfs_controller = DVFSController(config)
        self.thermal_manager = ThermalManager(config)
        self.power_gating_controller = PowerGatingController(config)
        self.clock_gating_controller = ClockGatingController(config)

        # Component power models
        self.component_models: Dict[str, PowerModel] = {}
        self.component_utilizations: Dict[str, float] = {}
        self.component_states: Dict[str, PowerState] = {}

        # Simulation state
        self.current_cycle = 0
        self.last_update_time = time.time()
        self.simulation_start_time = time.time()

        # Initialize default components
        self._initialize_default_components()

        logger.info("Power manager initialized")

    def _initialize_default_components(self):
        """Initialize default accelerator components."""
        # Systolic array
        systolic_profile = ComponentPowerProfile(
            component_name="systolic_array",
            base_power_watts=50.0,
            dynamic_power_factor=100.0,
            voltage_sensitivity=2.0,  # Quadratic voltage scaling
            frequency_sensitivity=1.0,  # Linear frequency scaling
            max_power_watts=200.0,
        )
        self.add_component("systolic_array", SimplePowerModel(systolic_profile))

        # Memory subsystem
        memory_profile = ComponentPowerProfile(
            component_name="memory",
            base_power_watts=20.0,
            dynamic_power_factor=30.0,
            voltage_sensitivity=1.5,
            frequency_sensitivity=0.8,
            max_power_watts=60.0,
        )
        self.add_component("memory", SimplePowerModel(memory_profile))

        # Control unit
        control_profile = ComponentPowerProfile(
            component_name="control",
            base_power_watts=10.0,
            dynamic_power_factor=15.0,
            voltage_sensitivity=1.8,
            frequency_sensitivity=1.2,
            max_power_watts=30.0,
        )
        self.add_component("control", SimplePowerModel(control_profile))

        # I/O subsystem
        io_profile = ComponentPowerProfile(
            component_name="io",
            base_power_watts=5.0,
            dynamic_power_factor=10.0,
            voltage_sensitivity=1.0,
            frequency_sensitivity=0.5,
            max_power_watts=20.0,
        )
        self.add_component("io", SimplePowerModel(io_profile))

    def add_component(self, component_name: str, power_model: PowerModel):
        """Add a component to power management."""
        self.component_models[component_name] = power_model
        self.component_utilizations[component_name] = 0.0
        self.component_states[component_name] = PowerState.ACTIVE

        # Register with power gating and clock gating controllers
        self.power_gating_controller.register_component(component_name)
        self.clock_gating_controller.register_clock_domain(component_name)

        logger.debug(f"Added component {component_name} to power management")

    def update_component_utilization(self, component_name: str, utilization: float):
        """Update component utilization for power calculation."""
        if component_name in self.component_utilizations:
            self.component_utilizations[component_name] = max(
                0.0, min(1.0, utilization)
            )

            # Update power gating controller
            is_active = utilization > 0.01  # Consider active if > 1% utilization
            self.power_gating_controller.update_component_activity(
                component_name, is_active
            )
            self.clock_gating_controller.update_domain_activity(
                component_name, is_active
            )

    def cycle_update(self):
        """Update power management state for one simulation cycle."""
        current_time = time.time()
        dt = current_time - self.last_update_time

        # Calculate current power consumption
        total_power = self._calculate_total_power()

        # Update thermal state
        self.metrics.current_temperature_c = self.thermal_manager.update_temperature(
            total_power, ambient_temp=25.0
        )

        # Get system metrics for DVFS
        system_utilization = np.mean(list(self.component_utilizations.values()))
        power_fraction = total_power / self.config.max_power_budget_watts
        performance_fraction = (
            1.0  # Default performance fraction for thermal calculation
        )

        # Update DVFS operating point
        voltage_level, frequency_level = self.dvfs_controller.update_operating_point(
            float(system_utilization),
            power_fraction,
            performance_fraction,
            self.metrics.current_temperature_c,
        )

        # Update metrics
        self.metrics.current_power_watts = total_power
        self.metrics.current_voltage = self.dvfs_controller.get_current_voltage()
        self.metrics.current_frequency_mhz = (
            self.dvfs_controller.get_current_frequency()
        )

        # Update energy consumption
        self.metrics.total_energy_joules += total_power * dt

        # Update time tracking
        if system_utilization > 0.1:
            self.metrics.active_time_seconds += dt
        elif system_utilization > 0.01:
            self.metrics.idle_time_seconds += dt
        else:
            self.metrics.sleep_time_seconds += dt

        # Update efficiency metrics
        if total_power > 0:
            # TOPS calculation based on theoretical peak performance
            estimated_tops = system_utilization * 100.0  # Assume 100 TOPS peak
            self.metrics.energy_efficiency_tops_per_watt = float(
                estimated_tops / total_power
            )

        self.metrics.power_utilization_percent = (
            total_power / self.config.max_power_budget_watts
        ) * 100
        self.metrics.thermal_utilization_percent = (
            self.thermal_manager.get_thermal_metrics()["thermal_utilization_percent"]
        )

        # Update component power breakdown
        self._update_component_power_breakdown()

        # Increment cycle counter
        self.current_cycle += 1
        self.last_update_time = current_time

    def _calculate_total_power(self) -> float:
        """Calculate total system power consumption."""
        total_power = 0.0

        # Get current operating parameters
        voltage = self.dvfs_controller.get_current_voltage()
        frequency = self.dvfs_controller.get_current_frequency()

        # Calculate power for each component
        for component_name, power_model in self.component_models.items():
            utilization = self.component_utilizations[component_name]
            state = self.power_gating_controller.get_component_state(component_name)

            # Apply thermal throttling if needed
            if self.thermal_manager.should_throttle():
                throttle_factor = self.thermal_manager.get_throttle_factor()
                utilization *= throttle_factor

            # Calculate component power
            component_power = power_model.calculate_power(
                utilization, voltage, frequency, state
            )
            total_power += component_power

        return total_power

    def _update_component_power_breakdown(self):
        """Update component-level power breakdown."""
        voltage = self.dvfs_controller.get_current_voltage()
        frequency = self.dvfs_controller.get_current_frequency()

        for component_name, power_model in self.component_models.items():
            utilization = self.component_utilizations[component_name]
            state = self.power_gating_controller.get_component_state(component_name)

            # Apply thermal throttling
            if self.thermal_manager.should_throttle():
                throttle_factor = self.thermal_manager.get_throttle_factor()
                utilization *= throttle_factor

            component_power = power_model.calculate_power(
                utilization, voltage, frequency, state
            )

            # Update metrics based on component type
            if component_name == "systolic_array":
                self.metrics.systolic_array_power_watts = component_power
            elif component_name == "memory":
                self.metrics.memory_power_watts = component_power
            elif component_name == "control":
                self.metrics.control_power_watts = component_power
            elif component_name == "io":
                self.metrics.io_power_watts = component_power

    def get_throttle_factor(self) -> float:
        """Get current system throttle factor due to thermal constraints."""
        return self.thermal_manager.get_throttle_factor()

    def is_component_active(self, component_name: str) -> bool:
        """Check if a component is in active power state."""
        state = self.power_gating_controller.get_component_state(component_name)
        return state == PowerState.ACTIVE

    def get_component_wake_penalty(self, component_name: str) -> int:
        """Get wake penalty cycles for accessing a component."""
        return self.power_gating_controller.get_wake_penalty(component_name)

    def get_power_metrics(self) -> PowerMetrics:
        """Get current power metrics."""
        return self.metrics

    def get_power_status(self) -> Dict[str, Any]:
        """Get comprehensive power management status."""
        return {
            "current_metrics": {
                "power_watts": self.metrics.current_power_watts,
                "voltage": self.metrics.current_voltage,
                "frequency_mhz": self.metrics.current_frequency_mhz,
                "temperature_c": self.metrics.current_temperature_c,
                "energy_joules": self.metrics.total_energy_joules,
                "energy_efficiency_tops_per_watt": self.metrics.energy_efficiency_tops_per_watt,
            },
            "component_breakdown": {
                "systolic_array_watts": self.metrics.systolic_array_power_watts,
                "memory_watts": self.metrics.memory_power_watts,
                "control_watts": self.metrics.control_power_watts,
                "io_watts": self.metrics.io_power_watts,
            },
            "utilization_metrics": {
                "power_utilization_percent": self.metrics.power_utilization_percent,
                "thermal_utilization_percent": self.metrics.thermal_utilization_percent,
                "component_utilizations": self.component_utilizations.copy(),
            },
            "time_breakdown": {
                "active_time_seconds": self.metrics.active_time_seconds,
                "idle_time_seconds": self.metrics.idle_time_seconds,
                "sleep_time_seconds": self.metrics.sleep_time_seconds,
                "total_simulation_time": time.time() - self.simulation_start_time,
            },
            "dvfs_status": self.dvfs_controller.get_dvfs_metrics(),
            "thermal_status": self.thermal_manager.get_thermal_metrics(),
            "power_gating_status": self.power_gating_controller.get_power_gating_metrics(),
            "clock_gating_status": self.clock_gating_controller.get_clock_gating_metrics(),
            "budget_status": {
                "max_power_budget_watts": self.config.max_power_budget_watts,
                "current_power_watts": self.metrics.current_power_watts,
                "budget_utilization_percent": (
                    self.metrics.current_power_watts
                    / self.config.max_power_budget_watts
                )
                * 100,
                "power_budget_margin_watts": self.config.max_power_budget_watts
                - self.metrics.current_power_watts,
            },
        }

    def reset(self):
        """Reset power management state."""
        self.metrics = PowerMetrics()
        self.current_cycle = 0
        self.last_update_time = time.time()
        self.simulation_start_time = time.time()

        # Reset component utilizations
        for component_name in self.component_utilizations:
            self.component_utilizations[component_name] = 0.0
            self.component_states[component_name] = PowerState.ACTIVE

        # Reset controllers
        self.dvfs_controller.utilization_history.clear()
        self.dvfs_controller.power_history.clear()
        self.dvfs_controller.performance_history.clear()
        self.dvfs_controller.transition_count = 0

        self.thermal_manager.current_temperature = 25.0
        self.thermal_manager.thermal_state = ThermalState.NORMAL
        self.thermal_manager.throttle_events.clear()
        self.thermal_manager.temperature_history.clear()

        # Reset gating controllers
        for component_name in self.power_gating_controller.component_states:
            self.power_gating_controller.component_states[
                component_name
            ] = PowerState.ACTIVE
            self.power_gating_controller.idle_counters[component_name] = 0
        self.power_gating_controller.gate_events.clear()

        for domain_name in self.clock_gating_controller.clock_enabled:
            self.clock_gating_controller.clock_enabled[domain_name] = True
            self.clock_gating_controller.idle_counters[domain_name] = 0
        self.clock_gating_controller.gate_events.clear()

        logger.info("Power manager reset")


class PowerOptimizer:
    """Power optimization algorithms and strategies."""

    def __init__(self, config: PowerConfig):
        """
        Initialize power optimizer.

        Args:
            config: Power management configuration
        """
        self.config = config
        self.optimization_history: List[Dict[str, Any]] = []

    def optimize_for_energy_efficiency(
        self, workload_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize system configuration for maximum energy efficiency.

        Args:
            workload_characteristics: Workload analysis results

        Returns:
            Recommended power configuration
        """
        recommendations = {}

        # Analyze workload characteristics
        compute_intensity = workload_characteristics.get("compute_intensity", 0.5)
        memory_intensity = workload_characteristics.get("memory_intensity", 0.5)
        parallelism = workload_characteristics.get("parallelism", 0.5)
        duration = workload_characteristics.get("duration_seconds", 1.0)

        # Recommend DVFS settings based on workload
        if compute_intensity > 0.8:
            # High compute workload - optimize for performance
            recommendations["voltage_level"] = VoltageLevel.HIGH
            recommendations["frequency_level"] = FrequencyLevel.HIGH
        elif compute_intensity < 0.3:
            # Low compute workload - optimize for energy
            recommendations["voltage_level"] = VoltageLevel.LOW
            recommendations["frequency_level"] = FrequencyLevel.LOW
        else:
            # Balanced workload
            recommendations["voltage_level"] = VoltageLevel.NOMINAL
            recommendations["frequency_level"] = FrequencyLevel.NOMINAL

        # Recommend power gating strategy
        if parallelism < 0.5:
            # Low parallelism - aggressive power gating
            recommendations["power_gate_threshold"] = (
                self.config.power_gate_threshold_idle_cycles // 2
            )
        else:
            # High parallelism - conservative power gating
            recommendations["power_gate_threshold"] = (
                self.config.power_gate_threshold_idle_cycles * 2
            )

        # Recommend thermal management
        if duration > 10.0:
            # Long-running workload - conservative thermal management
            recommendations["thermal_target"] = self.config.thermal_target_temp_c - 5.0
        else:
            # Short workload - allow higher temperatures
            recommendations["thermal_target"] = self.config.thermal_target_temp_c + 5.0

        # Record optimization decision
        optimization_record = {
            "timestamp": time.time(),
            "workload_characteristics": workload_characteristics,
            "recommendations": recommendations,
            "optimization_type": "energy_efficiency",
        }
        self.optimization_history.append(optimization_record)

        return recommendations

    def optimize_for_performance(
        self, performance_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize system configuration for maximum performance.

        Args:
            performance_requirements: Performance requirements and constraints

        Returns:
            Recommended power configuration
        """
        recommendations = {}

        target_throughput = performance_requirements.get(
            "target_throughput_tops", 100.0
        )
        latency_constraint = performance_requirements.get("max_latency_ms", 10.0)
        power_budget = performance_requirements.get(
            "power_budget_watts", self.config.max_power_budget_watts
        )

        # Maximize performance within power budget
        if power_budget >= self.config.max_power_budget_watts * 0.9:
            # High power budget - maximize performance
            recommendations["voltage_level"] = VoltageLevel.HIGH
            recommendations["frequency_level"] = FrequencyLevel.HIGH
            recommendations["power_gate_threshold"] = (
                self.config.power_gate_threshold_idle_cycles * 4
            )
        elif power_budget >= self.config.max_power_budget_watts * 0.7:
            # Medium power budget - balanced performance
            recommendations["voltage_level"] = VoltageLevel.NOMINAL
            recommendations["frequency_level"] = FrequencyLevel.HIGH
            recommendations["power_gate_threshold"] = (
                self.config.power_gate_threshold_idle_cycles * 2
            )
        else:
            # Low power budget - optimize efficiency
            recommendations["voltage_level"] = VoltageLevel.LOW
            recommendations["frequency_level"] = FrequencyLevel.NOMINAL
            recommendations[
                "power_gate_threshold"
            ] = self.config.power_gate_threshold_idle_cycles

        # Thermal considerations for performance
        recommendations["thermal_target"] = min(
            self.config.thermal_target_temp_c
            + 10.0,  # Allow higher temps for performance
            self.config.thermal_critical_temp_c - 5.0,  # But stay safe
        )

        # Record optimization decision
        optimization_record = {
            "timestamp": time.time(),
            "performance_requirements": performance_requirements,
            "recommendations": recommendations,
            "optimization_type": "performance",
        }
        self.optimization_history.append(optimization_record)

        return recommendations

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of optimization decisions."""
        return self.optimization_history.copy()


# Factory functions for common power management configurations


def create_medical_power_config() -> PowerConfig:
    """Create power configuration optimized for medical AI applications."""
    return PowerConfig(
        enable_dvfs=True,
        enable_power_gating=True,
        enable_clock_gating=True,
        enable_thermal_management=True,
        # Conservative DVFS for reliability
        default_voltage=VoltageLevel.NOMINAL,
        default_frequency=FrequencyLevel.NOMINAL,
        dvfs_reaction_time_ms=50.0,  # Slower reaction for stability
        # Conservative power gating for predictable performance
        power_gate_threshold_idle_cycles=2000,
        power_gate_wake_penalty_cycles=200,
        # Aggressive thermal management for safety
        thermal_target_temp_c=70.0,
        thermal_critical_temp_c=85.0,
        thermal_throttle_temp_c=80.0,
        # Moderate power budget for medical devices
        max_power_budget_watts=200.0,
        target_energy_efficiency_tops_per_watt=60.0,
    )


def create_automotive_power_config() -> PowerConfig:
    """Create power configuration optimized for automotive AI applications."""
    return PowerConfig(
        enable_dvfs=True,
        enable_power_gating=True,
        enable_clock_gating=True,
        enable_thermal_management=True,
        # Responsive DVFS for dynamic workloads
        default_voltage=VoltageLevel.NOMINAL,
        default_frequency=FrequencyLevel.NOMINAL,
        dvfs_reaction_time_ms=5.0,  # Fast reaction for automotive
        # Moderate power gating for real-time performance
        power_gate_threshold_idle_cycles=500,
        power_gate_wake_penalty_cycles=50,
        # Automotive thermal management
        thermal_target_temp_c=80.0,
        thermal_critical_temp_c=95.0,
        thermal_throttle_temp_c=90.0,
        # High power budget for automotive performance
        max_power_budget_watts=500.0,
        target_energy_efficiency_tops_per_watt=40.0,
    )


def create_edge_power_config() -> PowerConfig:
    """Create power configuration optimized for edge AI applications."""
    return PowerConfig(
        enable_dvfs=True,
        enable_power_gating=True,
        enable_clock_gating=True,
        enable_thermal_management=True,
        # Aggressive DVFS for energy efficiency
        default_voltage=VoltageLevel.LOW,
        default_frequency=FrequencyLevel.NOMINAL,
        dvfs_reaction_time_ms=20.0,
        # Aggressive power gating for edge efficiency
        power_gate_threshold_idle_cycles=100,
        power_gate_wake_penalty_cycles=20,
        # Edge thermal management
        thermal_target_temp_c=65.0,
        thermal_critical_temp_c=80.0,
        thermal_throttle_temp_c=75.0,
        # Low power budget for edge devices
        max_power_budget_watts=50.0,
        target_energy_efficiency_tops_per_watt=100.0,
    )


def create_datacenter_power_config() -> PowerConfig:
    """Create power configuration optimized for datacenter AI applications."""
    return PowerConfig(
        enable_dvfs=True,
        enable_power_gating=False,  # Less power gating in datacenter for predictability
        enable_clock_gating=True,
        enable_thermal_management=True,
        # Performance-oriented DVFS
        default_voltage=VoltageLevel.HIGH,
        default_frequency=FrequencyLevel.HIGH,
        dvfs_reaction_time_ms=1.0,  # Very fast reaction
        # Minimal power gating
        power_gate_threshold_idle_cycles=5000,
        power_gate_wake_penalty_cycles=500,
        # Datacenter thermal management with better cooling
        thermal_target_temp_c=85.0,
        thermal_critical_temp_c=100.0,
        thermal_throttle_temp_c=95.0,
        # High power budget for datacenter performance
        max_power_budget_watts=1000.0,
        target_energy_efficiency_tops_per_watt=30.0,  # Lower efficiency for max performance
    )


# Power analysis and reporting utilities


class PowerAnalyzer:
    """Analyzes power consumption patterns and provides insights."""

    def __init__(self):
        """Initialize power analyzer."""
        self.power_samples: List[float] = []
        self.utilization_samples: List[float] = []
        self.temperature_samples: List[float] = []
        self.timestamps: List[float] = []

    def add_sample(self, power: float, utilization: float, temperature: float):
        """Add a power measurement sample."""
        self.power_samples.append(power)
        self.utilization_samples.append(utilization)
        self.temperature_samples.append(temperature)
        self.timestamps.append(time.time())

        # Keep only recent samples (last 1000)
        if len(self.power_samples) > 1000:
            self.power_samples.pop(0)
            self.utilization_samples.pop(0)
            self.temperature_samples.pop(0)
            self.timestamps.pop(0)

    def get_power_statistics(self) -> Dict[str, Any]:
        """Get statistical analysis of power consumption."""
        if not self.power_samples:
            return {}

        power_array = np.array(self.power_samples)
        utilization_array = np.array(self.utilization_samples)
        temperature_array = np.array(self.temperature_samples)

        return {
            "power_stats": {
                "mean_watts": float(np.mean(power_array)),
                "median_watts": float(np.median(power_array)),
                "std_watts": float(np.std(power_array)),
                "min_watts": float(np.min(power_array)),
                "max_watts": float(np.max(power_array)),
                "p95_watts": float(np.percentile(power_array, 95)),
                "p99_watts": float(np.percentile(power_array, 99)),
            },
            "utilization_stats": {
                "mean_utilization": float(np.mean(utilization_array)),
                "median_utilization": float(np.median(utilization_array)),
                "std_utilization": float(np.std(utilization_array)),
                "min_utilization": float(np.min(utilization_array)),
                "max_utilization": float(np.max(utilization_array)),
            },
            "temperature_stats": {
                "mean_temp_c": float(np.mean(temperature_array)),
                "median_temp_c": float(np.median(temperature_array)),
                "std_temp_c": float(np.std(temperature_array)),
                "min_temp_c": float(np.min(temperature_array)),
                "max_temp_c": float(np.max(temperature_array)),
                "p95_temp_c": float(np.percentile(temperature_array, 95)),
                "p99_temp_c": float(np.percentile(temperature_array, 99)),
            },
            "efficiency_stats": {
                "power_utilization_correlation": (
                    float(np.corrcoef(power_array, utilization_array)[0, 1])
                    if len(power_array) > 1
                    else 0.0
                ),
                "power_temperature_correlation": (
                    float(np.corrcoef(power_array, temperature_array)[0, 1])
                    if len(power_array) > 1
                    else 0.0
                ),
            },
            "sample_count": len(self.power_samples),
            "analysis_duration_seconds": (
                self.timestamps[-1] - self.timestamps[0]
                if len(self.timestamps) > 1
                else 0.0
            ),
        }

    def detect_power_anomalies(
        self, threshold_std_devs: float = 3.0
    ) -> List[Dict[str, Any]]:
        """Detect power consumption anomalies."""
        if len(self.power_samples) < 10:
            return []

        power_array = np.array(self.power_samples)
        mean_power = np.mean(power_array)
        std_power = np.std(power_array)

        anomalies = []
        for i, (power, timestamp) in enumerate(
            zip(self.power_samples, self.timestamps)
        ):
            z_score = abs(power - mean_power) / std_power if std_power > 0 else 0

            if z_score > threshold_std_devs:
                anomalies.append(
                    {
                        "timestamp": timestamp,
                        "sample_index": i,
                        "power_watts": power,
                        "z_score": z_score,
                        "deviation_from_mean": power - mean_power,
                        "utilization": self.utilization_samples[i],
                        "temperature_c": self.temperature_samples[i],
                    }
                )

        return anomalies

    def get_power_efficiency_report(self) -> Dict[str, Any]:
        """Generate power efficiency analysis report."""
        if not self.power_samples:
            return {}

        stats = self.get_power_statistics()
        anomalies = self.detect_power_anomalies()

        # Calculate efficiency metrics
        power_array = np.array(self.power_samples)
        utilization_array = np.array(self.utilization_samples)

        # Power efficiency = Utilization / Power (higher is better)
        efficiency_ratios = []
        for power, util in zip(power_array, utilization_array):
            if power > 0:
                efficiency_ratios.append(util / power)

        efficiency_array = (
            np.array(efficiency_ratios) if efficiency_ratios else np.array([0])
        )

        return {
            "summary": {
                "total_samples": len(self.power_samples),
                "analysis_duration_hours": (
                    (self.timestamps[-1] - self.timestamps[0]) / 3600
                    if len(self.timestamps) > 1
                    else 0.0
                ),
                "average_power_watts": stats["power_stats"]["mean_watts"],
                "peak_power_watts": stats["power_stats"]["max_watts"],
                "average_utilization": stats["utilization_stats"]["mean_utilization"],
                "average_temperature_c": stats["temperature_stats"]["mean_temp_c"],
            },
            "efficiency_metrics": {
                "mean_efficiency_ratio": float(np.mean(efficiency_array)),
                "median_efficiency_ratio": float(np.median(efficiency_array)),
                "best_efficiency_ratio": float(np.max(efficiency_array)),
                "worst_efficiency_ratio": float(np.min(efficiency_array)),
            },
            "detailed_stats": stats,
            "anomalies": {
                "count": len(anomalies),
                "anomaly_rate_percent": (
                    (len(anomalies) / len(self.power_samples)) * 100
                    if self.power_samples
                    else 0.0
                ),
                "recent_anomalies": anomalies[-5:] if anomalies else [],
            },
            "recommendations": self._generate_efficiency_recommendations(
                stats, anomalies
            ),
        }

    def _generate_efficiency_recommendations(
        self, stats: Dict[str, Any], anomalies: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate power efficiency improvement recommendations."""
        recommendations = []

        power_stats = stats.get("power_stats", {})
        util_stats = stats.get("utilization_stats", {})
        temp_stats = stats.get("temperature_stats", {})

        # High power variability
        if power_stats.get("std_watts", 0) > power_stats.get("mean_watts", 1) * 0.3:
            recommendations.append(
                "High power variability detected. Consider DVFS tuning for more stable power consumption."
            )

        # Low utilization with high power
        if (
            util_stats.get("mean_utilization", 0) < 0.5
            and power_stats.get("mean_watts", 0) > 100
        ):
            recommendations.append(
                "Low utilization with high power consumption. Enable more aggressive power gating."
            )

        # High temperature
        if temp_stats.get("mean_temp_c", 0) > 80:
            recommendations.append(
                "High operating temperature detected. Improve cooling or reduce power consumption."
            )

        # Frequent anomalies
        if len(anomalies) > len(self.power_samples) * 0.1:
            recommendations.append(
                "Frequent power anomalies detected. Check for workload irregularities or hardware issues."
            )

        # Low efficiency
        if (
            util_stats.get("mean_utilization", 0) > 0.7
            and power_stats.get("mean_watts", 0) > 200
        ):
            recommendations.append(
                "High power consumption with good utilization. Consider optimizing algorithms or hardware."
            )

        return recommendations


# ---------------------------------------------------------------------------
# Convenience helpers – *simple* glue for high-level examples
# ---------------------------------------------------------------------------


def integrate_power_management(
    accelerator_config: "AcceleratorConfig",
    power_config: Optional[PowerConfig] = None,
) -> "PowerManager":
    """Create and attach a :class:`PowerManager` to the given accelerator.

    The helper returns a fully-initialised :class:`PowerManager` instance that
    is **independent** from the accelerator model so that examples can inspect
    its state after a simulation run.  It currently performs the following
    actions:

    1.   Instantiate a :class:`PowerManager` with *power_config* or the one
         embedded in *accelerator_config*.
    2.   Register the default component set (array, memory, control, IO).

    The function purposefully stays extremely lightweight: the advanced power
    integration with live utilisation tracking is handled elsewhere in the code
    base.  For demos we merely need a *stand-alone* object that provides power
    metrics via :pymeth:`PowerManager.get_power_status`.
    """

    # Local import to avoid a hard dependency if utils is trimmed down
    try:
        from ..utils.config import AcceleratorConfig as _AccelCfg  # noqa: F401
    except ImportError:  # Fallback for legacy flat layout
        _AccelCfg = None  # type: ignore

    # Use provided config or fallback to accelerator's embedded config
    cfg = power_config or getattr(accelerator_config, "power", None) or PowerConfig()

    pm = PowerManager(cfg)

    # NOTE: We deliberately *do not* attempt to hook the power manager into the
    #       simulator update loop because the advanced models already support
    #       that via dedicated callbacks.  The examples only rely on static
    #       reporting after the run has finished which works without coupling.

    logger.info(
        "Power manager integrated with accelerator config '%s'",
        getattr(accelerator_config, "name", "Unnamed"),
    )
    return pm


def create_power_report(
    power_manager: "PowerManager", output_path: Optional[Union[str, Path]] = None
) -> Path:
    """Generate a minimal power report and write it to *output_path*.

    The human-readable report contains current as well as cumulative power
    statistics.  It is intentionally formatted as plain text so that it can be
    viewed easily in any environment (including CI logs).
    """
    from pathlib import Path as _Path

    # Determine output location
    if output_path is None:
        _out = _Path("power_report.txt")
    else:
        _out = _Path(output_path)

    # Collect metrics
    metrics = power_manager.get_power_status()

    lines = [
        "# OpenAccelerator – Power Report",
        f"Report generated at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for section, values in metrics.items():
        lines.append(f"[{section.capitalize()}]")
        for key, val in values.items():
            lines.append(f"{key.replace('_', ' ').title():30s}: {val}")
        lines.append("")

    _out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Power report written to %s", _out)
    return _out


# ---------------------------------------------------------------------------
# Ensure helper symbols are available for *import *
# ---------------------------------------------------------------------------

__all__ = [
    "PowerState",
    "VoltageLevel",
    "FrequencyLevel",
    "ThermalState",
    "PowerConfig",
    "PowerMetrics",
    "ComponentPowerProfile",
    "PowerModel",
    "SimplePowerModel",
    "DVFSController",
    "ThermalManager",
    "PowerGatingController",
    "ClockGatingController",
    "PowerManager",
    "PowerOptimizer",
    "PowerAnalyzer",
    "create_medical_power_config",
    "create_automotive_power_config",
    "create_edge_power_config",
    "create_datacenter_power_config",
    "integrate_power_management",
    "create_power_report",
]
