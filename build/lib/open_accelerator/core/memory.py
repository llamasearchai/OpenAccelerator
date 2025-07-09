"""
Advanced memory hierarchy implementation.

Features:
- Multi-level memory hierarchy (L1/L2/DRAM)
- Bandwidth and latency modeling
- Medical-grade data integrity checks
- Power-aware memory management
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils.config import AcceleratorConfig, MemoryConfig, MemoryType

logger = logging.getLogger(__name__)


class MemoryState(Enum):
    """Memory subsystem states."""

    IDLE = "idle"
    READING = "reading"
    WRITING = "writing"
    REFRESHING = "refreshing"
    POWER_DOWN = "power_down"


@dataclass
class MemoryMetrics:
    """Memory performance and power metrics."""

    total_reads: int = 0
    total_writes: int = 0
    read_latency_cycles: int = 0
    write_latency_cycles: int = 0
    bandwidth_utilization: float = 0.0
    energy_consumed: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    refresh_cycles: int = 0


class MemoryBuffer:
    """
    Advanced memory buffer with ECC and power management.

    Supports multiple memory types with appropriate timing and power models.
    """

    def __init__(self, name: str, config: MemoryConfig, medical_mode: bool = False):
        """
        Initialize memory buffer.

        Args:
            name: Buffer identifier
            config: Memory configuration
            medical_mode: Enable medical-grade features
        """
        self.name = name
        self.config = config
        self.medical_mode = medical_mode

        # Memory storage
        self.capacity = config.buffer_size
        self.data: deque = deque(maxlen=self.capacity)
        self.metadata: Dict[int, Dict] = {}  # Address -> metadata mapping

        # Performance parameters
        self.bandwidth = config.bandwidth
        self.latency = config.latency
        self.energy_per_access = config.energy_per_access

        # State tracking
        self.state = MemoryState.IDLE
        self.metrics = MemoryMetrics()
        self.current_cycle = 0

        # Medical mode features
        if medical_mode:
            self.enable_ecc = True
            self.enable_integrity_checks = True
            self.redundant_storage = True
        else:
            self.enable_ecc = False
            self.enable_integrity_checks = False
            self.redundant_storage = False

        # Memory type specific initialization
        self._setup_memory_type()

        logger.debug(
            f"Initialized {name} buffer: {config.buffer_size} elements, "
            f"{config.bandwidth} BW, {config.memory_type.value}"
        )

    def _setup_memory_type(self):
        """Configure memory type specific parameters."""
        if self.config.memory_type == MemoryType.SRAM:
            self.access_energy = 0.1  # pJ per bit
            self.static_power = 1e-6  # μW per bit
            self.refresh_required = False

        elif self.config.memory_type == MemoryType.DRAM:
            self.access_energy = 1.0  # pJ per bit
            self.static_power = 1e-8  # μW per bit
            self.refresh_required = True
            self.refresh_period = 64000  # cycles (64ms at 1MHz)

        elif self.config.memory_type == MemoryType.HBM:
            self.access_energy = 2.0  # pJ per bit
            self.static_power = 1e-7  # μW per bit
            self.refresh_required = True
            self.refresh_period = 32000  # cycles

        else:  # Cache
            self.access_energy = 0.05  # pJ per bit
            self.static_power = 1e-5  # μW per bit
            self.refresh_required = False

    def read(
        self, num_elements: int, address: Optional[int] = None
    ) -> Tuple[List[Any], bool]:
        """
        Read data from buffer.

        Args:
            num_elements: Number of elements to read
            address: Optional specific address (for random access)

        Returns:
            Tuple of (data_list, success_flag)
        """
        if self.state == MemoryState.POWER_DOWN:
            return [], False

        # Bandwidth limitation
        actual_elements = min(num_elements, self.bandwidth)

        # Latency simulation
        if self.latency > 1:
            self.state = MemoryState.READING
            # In real implementation, would need cycle-accurate modeling

        data = []
        success = True

        try:
            for _ in range(actual_elements):
                if self.data:
                    element = self.data.popleft()

                    # Medical mode: integrity check
                    if self.medical_mode and self.enable_integrity_checks:
                        if not self._verify_data_integrity(element):
                            logger.error(f"{self.name}: Data integrity check failed")
                            success = False
                            continue

                    data.append(element)
                else:
                    break

            # Update metrics
            self.metrics.total_reads += len(data)
            self.metrics.read_latency_cycles += self.latency
            self.metrics.energy_consumed += len(data) * self.access_energy

            self.state = MemoryState.IDLE

        except Exception as e:
            logger.error(f"{self.name}: Read operation failed: {e}")
            success = False

        return data, success

    def write(self, data_elements: List[Any]) -> Tuple[int, bool]:
        """
        Write data to buffer.

        Args:
            data_elements: List of elements to write

        Returns:
            Tuple of (elements_written, success_flag)
        """
        if self.state == MemoryState.POWER_DOWN:
            return 0, False

        # Bandwidth limitation
        actual_elements = min(len(data_elements), self.bandwidth)
        elements_written = 0
        success = True

        try:
            self.state = MemoryState.WRITING

            for i in range(actual_elements):
                if len(self.data) < self.capacity:
                    element = data_elements[i]

                    # Medical mode: add ECC
                    if self.medical_mode and self.enable_ecc:
                        element = self._add_ecc(element)

                    self.data.append(element)
                    elements_written += 1

                    # Medical mode: duplicate storage
                    if self.medical_mode and self.redundant_storage:
                        self._store_redundant_copy(element, len(self.data) - 1)

                else:
                    logger.warning(
                        f"{self.name}: Buffer full, cannot write more elements"
                    )
                    break

            # Update metrics
            self.metrics.total_writes += elements_written
            self.metrics.write_latency_cycles += self.latency
            self.metrics.energy_consumed += elements_written * self.access_energy

            self.state = MemoryState.IDLE

        except Exception as e:
            logger.error(f"{self.name}: Write operation failed: {e}")
            success = False

        return elements_written, success

    def _verify_data_integrity(self, element: Any) -> bool:
        """Verify data integrity using checksums or ECC."""
        # Simplified integrity check
        if hasattr(element, "__dict__") and "checksum" in element.__dict__:
            # Verify checksum
            calculated = hash(str(element)) & 0xFFFF
            return calculated == element.checksum
        return True  # Assume OK if no checksum

    def _add_ecc(self, element: Any) -> Any:
        """Add Error Correcting Code to element."""
        # Simplified ECC addition
        if hasattr(element, "__dict__"):
            element.checksum = hash(str(element)) & 0xFFFF
        return element

    def _store_redundant_copy(self, element: Any, address: int) -> None:
        """Store redundant copy for medical-grade reliability."""
        self.metadata[address] = {
            "redundant_copy": element,
            "timestamp": self.current_cycle,
            "access_count": 0,
        }

    def cycle(self) -> None:
        """Execute one memory cycle (refresh, power management, etc.)."""
        self.current_cycle += 1

        # Handle DRAM refresh
        if self.refresh_required and self.current_cycle % self.refresh_period == 0:
            self._perform_refresh()

        # Update bandwidth utilization
        self._update_bandwidth_utilization()

    def _perform_refresh(self) -> None:
        """Perform DRAM refresh cycle."""
        self.state = MemoryState.REFRESHING
        self.metrics.refresh_cycles += 1
        # Refresh energy cost
        self.metrics.energy_consumed += self.capacity * 0.01  # 0.01pJ per element
        self.state = MemoryState.IDLE

    def _update_bandwidth_utilization(self) -> None:
        """Update bandwidth utilization metrics."""
        # Simplified: based on recent activity
        recent_accesses = self.metrics.total_reads + self.metrics.total_writes
        if self.current_cycle > 0:
            self.metrics.bandwidth_utilization = min(
                1.0, recent_accesses / self.current_cycle
            )

    def power_down(self) -> None:
        """Enter power-down mode."""
        if self.config.memory_type in [MemoryType.DRAM, MemoryType.HBM]:
            self.state = MemoryState.POWER_DOWN
            logger.debug(f"{self.name}: Entered power-down mode")

    def wake_up(self) -> None:
        """Wake up from power-down mode."""
        if self.state == MemoryState.POWER_DOWN:
            self.state = MemoryState.IDLE
            logger.debug(f"{self.name}: Woke up from power-down mode")

    def get_status(self) -> Dict[str, Any]:
        """Get current buffer status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "occupancy": len(self.data),
            "capacity": self.capacity,
            "utilization": len(self.data) / self.capacity,
            "metrics": self.metrics,
        }

    def reset(self) -> None:
        """Reset buffer state."""
        self.data.clear()
        self.metadata.clear()
        self.metrics = MemoryMetrics()
        self.current_cycle = 0
        self.state = MemoryState.IDLE


class CacheBuffer(MemoryBuffer):
    """
    Cache buffer with associativity and replacement policies.
    """

    def __init__(
        self,
        name: str,
        config: MemoryConfig,
        associativity: int = 4,
        replacement_policy: str = "LRU",
    ):
        super().__init__(name, config)

        self.associativity = associativity
        self.replacement_policy = replacement_policy
        self.cache_lines = config.buffer_size // associativity

        # Cache organization
        self.cache_sets: List[List[Dict]] = []
        for _ in range(self.cache_lines):
            cache_set = []
            for _ in range(associativity):
                cache_set.append(
                    {
                        "valid": False,
                        "tag": None,
                        "data": None,
                        "lru_counter": 0,
                        "dirty": False,
                    }
                )
            self.cache_sets.append(cache_set)

        self.lru_global_counter = 0

    def cache_lookup(self, address: int) -> Tuple[bool, Any]:
        """
        Perform cache lookup.

        Args:
            address: Memory address

        Returns:
            Tuple of (hit, data)
        """
        set_index = address % self.cache_lines
        tag = address // self.cache_lines

        cache_set = self.cache_sets[set_index]

        # Check for hit
        for way in cache_set:
            if way["valid"] and way["tag"] == tag:
                # Cache hit
                way["lru_counter"] = self.lru_global_counter
                self.lru_global_counter += 1
                self.metrics.cache_hits += 1
                return True, way["data"]

        # Cache miss
        self.metrics.cache_misses += 1
        return False, None

    def cache_insert(self, address: int, data: Any) -> bool:
        """
        Insert data into cache.

        Args:
            address: Memory address
            data: Data to insert

        Returns:
            True if inserted successfully
        """
        set_index = address % self.cache_lines
        tag = address // self.cache_lines

        cache_set = self.cache_sets[set_index]

        # Find empty way or LRU way
        victim_way = None
        min_lru = float("inf")

        for way in cache_set:
            if not way["valid"]:
                victim_way = way
                break
            elif way["lru_counter"] < min_lru:
                min_lru = way["lru_counter"]
                victim_way = way

        if victim_way:
            victim_way["valid"] = True
            victim_way["tag"] = tag
            victim_way["data"] = data
            victim_way["lru_counter"] = self.lru_global_counter
            victim_way["dirty"] = False
            self.lru_global_counter += 1
            return True

        return False


class MemoryHierarchy:
    """
    Complete memory hierarchy management.

    Manages L1 cache, L2 cache, and main memory with coherence and power optimization.
    """

    def __init__(self, config: AcceleratorConfig):
        """
        Initialize memory hierarchy.

        Args:
            config: Complete accelerator configuration
        """
        self.config = config
        self.medical_mode = config.medical_mode

        # Initialize memory levels
        self.l1_cache = self._create_l1_cache()
        self.l2_cache = self._create_l2_cache()
        self.main_memory = self._create_main_memory()

        # Memory controller
        self.outstanding_requests: Dict[int, Dict] = {}
        self.request_queue: deque = deque()

        # Metrics
        self.total_accesses = 0
        self.l1_hit_rate = 0.0
        self.l2_hit_rate = 0.0
        self.average_latency = 0.0

        logger.info("Initialized memory hierarchy")

    def _create_l1_cache(self) -> CacheBuffer:
        """Create L1 cache configuration."""
        l1_config = MemoryConfig(
            memory_type=MemoryType.CACHE,
            buffer_size=self.config.memory.l1_size,
            bandwidth=self.config.memory.l1_bandwidth,
            latency=1,  # 1 cycle for L1
            energy_per_access=0.1,
        )
        return CacheBuffer(
            "L1_Cache", l1_config, associativity=4, replacement_policy="LRU"
        )

    def _create_l2_cache(self) -> CacheBuffer:
        """Create L2 cache configuration."""
        l2_config = MemoryConfig(
            memory_type=MemoryType.CACHE,
            buffer_size=self.config.memory.l2_size,
            bandwidth=self.config.memory.l2_bandwidth,
            latency=8,  # 8 cycles for L2
            energy_per_access=1.0,
        )
        return CacheBuffer(
            "L2_Cache", l2_config, associativity=8, replacement_policy="LRU"
        )

    def _create_main_memory(self) -> MemoryBuffer:
        """Create main memory configuration."""
        main_config = MemoryConfig(
            memory_type=MemoryType.HBM
            if self.config.memory.enable_hbm
            else MemoryType.DRAM,
            buffer_size=self.config.memory.main_memory_size,
            bandwidth=self.config.memory.main_memory_bandwidth,
            latency=100,  # 100 cycles for main memory
            energy_per_access=20.0,
        )
        return MemoryBuffer("MainMemory", main_config, self.medical_mode)

    def read_request(self, address: int, size: int) -> Tuple[List[Any], int]:
        """
        Process memory read request through hierarchy.

        Args:
            address: Memory address
            size: Number of elements to read

        Returns:
            Tuple of (data, latency_cycles)
        """
        self.total_accesses += 1
        total_latency = 0
        data = []

        # Try L1 cache first
        hit, cache_data = self.l1_cache.cache_lookup(address)
        if hit:
            data = cache_data[:size] if isinstance(cache_data, list) else [cache_data]
            total_latency = 1
            self._update_hit_rates(1, 0)
            return data, total_latency

        # Try L2 cache
        hit, cache_data = self.l2_cache.cache_lookup(address)
        if hit:
            data = cache_data[:size] if isinstance(cache_data, list) else [cache_data]
            total_latency = 8

            # Install in L1
            self.l1_cache.cache_insert(address, cache_data)
            self._update_hit_rates(2, 0)
            return data, total_latency

        # Access main memory
        main_data, success = self.main_memory.read(size, address)
        if success:
            data = main_data
            total_latency = 100

            # Install in caches
            if data:
                self.l2_cache.cache_insert(address, data)
                self.l1_cache.cache_insert(address, data)

        self._update_hit_rates(3, 0)  # Main memory access
        return data, total_latency

    def write_request(self, address: int, data: List[Any]) -> Tuple[bool, int]:
        """
        Process memory write request through hierarchy.

        Args:
            address: Memory address
            data: Data to write

        Returns:
            Tuple of (success, latency_cycles)
        """
        self.total_accesses += 1

        # Write-through policy: update all levels
        success = True
        total_latency = 0

        # Update L1 if present
        hit, _ = self.l1_cache.cache_lookup(address)
        if hit:
            self.l1_cache.cache_insert(address, data)
            total_latency += 1

        # Update L2 if present
        hit, _ = self.l2_cache.cache_lookup(address)
        if hit:
            self.l2_cache.cache_insert(address, data)
            total_latency += 8

        # Always write to main memory
        written, mem_success = self.main_memory.write(data)
        success = success and mem_success
        total_latency += 100

        return success, total_latency

    def _update_hit_rates(self, level: int, miss_type: int) -> None:
        """Update cache hit rate statistics."""
        # Simplified hit rate tracking
        if level == 1:  # L1 hit
            self.l1_hit_rate = (
                self.l1_hit_rate * (self.total_accesses - 1) + 1
            ) / self.total_accesses
        elif level == 2:  # L2 hit
            self.l2_hit_rate = (
                self.l2_hit_rate * (self.total_accesses - 1) + 1
            ) / self.total_accesses

    def cycle(self) -> None:
        """Execute one memory hierarchy cycle."""
        # Cycle all memory components
        self.l1_cache.cycle()
        self.l2_cache.cycle()
        self.main_memory.cycle()

        # Process pending requests
        self._process_request_queue()

    def _process_request_queue(self) -> None:
        """Process queued memory requests."""
        # Simplified request processing
        if self.request_queue:
            request = self.request_queue.popleft()
            # Process request based on type
            pass

    def get_hierarchy_metrics(self) -> Dict[str, Any]:
        """Get comprehensive memory hierarchy metrics."""
        return {
            "l1_metrics": self.l1_cache.get_status(),
            "l2_metrics": self.l2_cache.get_status(),
            "main_memory_metrics": self.main_memory.get_status(),
            "hierarchy_metrics": {
                "total_accesses": self.total_accesses,
                "l1_hit_rate": self.l1_hit_rate,
                "l2_hit_rate": self.l2_hit_rate,
                "average_latency": self.average_latency,
            },
        }

    def power_management(self, utilization: float) -> None:
        """Intelligent power management based on utilization."""
        # Power down unused memory when utilization is low
        if utilization < 0.1:  # Less than 10% utilization
            if hasattr(self.main_memory, "power_down"):
                self.main_memory.power_down()
        elif utilization > 0.5:  # High utilization
            if hasattr(self.main_memory, "wake_up"):
                self.main_memory.wake_up()

    def reset(self) -> None:
        """Reset entire memory hierarchy."""
        self.l1_cache.reset()
        self.l2_cache.reset()
        self.main_memory.reset()

        self.outstanding_requests.clear()
        self.request_queue.clear()
        self.total_accesses = 0
        self.l1_hit_rate = 0.0
        self.l2_hit_rate = 0.0
        self.average_latency = 0.0

        logger.info("Memory hierarchy reset completed")
