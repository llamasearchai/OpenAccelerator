"""
Interconnect module for Open Accelerator simulator.

This module provides comprehensive network-on-chip (NoC) and interconnect
functionality for the accelerator simulator.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class NetworkTopology(Enum):
    """Supported network topologies."""

    MESH = "mesh"
    TORUS = "torus"
    TREE = "tree"
    CROSSBAR = "crossbar"
    RING = "ring"
    BUTTERFLY = "butterfly"


class RoutingAlgorithm(Enum):
    """Routing algorithms for NoC."""

    XY = "xy"
    YX = "yx"
    ADAPTIVE = "adaptive"
    MINIMAL = "minimal"
    DETERMINISTIC = "deterministic"


class FlowControl(Enum):
    """Flow control mechanisms."""

    STORE_AND_FORWARD = "store_and_forward"
    WORMHOLE = "wormhole"
    VIRTUAL_CUT_THROUGH = "virtual_cut_through"
    CREDIT_BASED = "credit_based"


@dataclass
class InterconnectConfig:
    """Configuration for interconnect systems."""

    topology: NetworkTopology = NetworkTopology.MESH
    routing_algorithm: RoutingAlgorithm = RoutingAlgorithm.XY
    flow_control: FlowControl = FlowControl.WORMHOLE
    buffer_size: int = 16
    flit_size: int = 64
    num_virtual_channels: int = 2
    bandwidth: float = 1e9  # bytes per second
    latency: float = 1e-9  # seconds
    enable_congestion_control: bool = True
    enable_quality_of_service: bool = False
    medical_mode: bool = False


@dataclass
class Packet:
    """Network packet representation."""

    source: int
    destination: int
    payload: Any
    priority: int = 0
    timestamp: float = 0.0
    packet_id: int = 0
    size: int = 64
    routing_info: Optional[Dict[str, Any]] = None


@dataclass
class NetworkMessage:
    """High-level network message."""

    message_type: str
    source_id: int
    destination_id: int
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


class Link:
    """Network link between routers."""

    def __init__(
        self,
        link_id: int,
        source: int,
        destination: int,
        bandwidth: float,
        latency: float,
    ):
        self.link_id = link_id
        self.source = source
        self.destination = destination
        self.bandwidth = bandwidth
        self.latency = latency
        self.utilization = 0.0
        self.congestion_level = 0.0
        self.packets_in_transit: List[Packet] = []

    def send_packet(self, packet: Packet) -> bool:
        """Send a packet through this link."""
        if self.congestion_level < 0.9:  # Allow if not heavily congested
            self.packets_in_transit.append(packet)
            self.utilization += packet.size / self.bandwidth
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get link status information."""
        return {
            "link_id": self.link_id,
            "utilization": self.utilization,
            "congestion_level": self.congestion_level,
            "packets_in_transit": len(self.packets_in_transit),
        }


class Router:
    """Network router for NoC."""

    def __init__(self, router_id: int, x: int, y: int, config: InterconnectConfig):
        self.router_id = router_id
        self.x = x
        self.y = y
        self.config = config
        self.input_buffers: List[List[Packet]] = [[] for _ in range(4)]  # N, S, E, W
        self.output_buffers: List[List[Packet]] = [[] for _ in range(4)]
        self.routing_table: Dict[int, int] = {}
        self.links: List[Link] = []
        self.performance_metrics = {
            "packets_routed": 0,
            "average_latency": 0.0,
            "buffer_utilization": 0.0,
        }

    def route_packet(self, packet: Packet) -> Optional[int]:
        """Route packet to appropriate output port."""
        if self.config.routing_algorithm == RoutingAlgorithm.XY:
            return self._xy_routing(packet)
        elif self.config.routing_algorithm == RoutingAlgorithm.ADAPTIVE:
            return self._adaptive_routing(packet)
        else:
            return 0  # Default port

    def _xy_routing(self, packet: Packet) -> int:
        """XY routing algorithm."""
        # Simplified XY routing
        dest_x = packet.destination % 16  # Assuming 16x16 grid
        dest_y = packet.destination // 16

        if dest_x > self.x:
            return 2  # East
        elif dest_x < self.x:
            return 3  # West
        elif dest_y > self.y:
            return 0  # North
        else:
            return 1  # South

    def _adaptive_routing(self, packet: Packet) -> int:
        """Adaptive routing based on congestion."""
        # Simplified adaptive routing
        best_port = 0
        min_congestion = float("inf")

        for i, link in enumerate(self.links):
            if link.congestion_level < min_congestion:
                min_congestion = link.congestion_level
                best_port = i

        return best_port

    def process_packet(self, packet: Packet) -> bool:
        """Process a packet through the router."""
        output_port = self.route_packet(packet)
        if output_port is not None and output_port < len(self.output_buffers):
            self.output_buffers[output_port].append(packet)
            self.performance_metrics["packets_routed"] += 1
            return True
        return False


class NetworkInterface:
    """Network interface for connecting to routers."""

    def __init__(self, interface_id: int, router: Router):
        self.interface_id = interface_id
        self.router = router
        self.send_queue: List[Packet] = []
        self.receive_queue: List[Packet] = []
        self.statistics = {
            "packets_sent": 0,
            "packets_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
        }

    def send_message(self, message: NetworkMessage) -> bool:
        """Send a message through the network."""
        packet = Packet(
            source=message.source_id,
            destination=message.destination_id,
            payload=message.data,
            timestamp=message.timestamp,
        )
        return self.send_packet(packet)

    def send_packet(self, packet: Packet) -> bool:
        """Send a packet through the network."""
        if self.router.process_packet(packet):
            self.statistics["packets_sent"] += 1
            self.statistics["bytes_sent"] += packet.size
            return True
        return False

    def receive_packet(self) -> Optional[Packet]:
        """Receive a packet from the network."""
        if self.receive_queue:
            packet = self.receive_queue.pop(0)
            self.statistics["packets_received"] += 1
            self.statistics["bytes_received"] += packet.size
            return packet
        return None


class NoC:
    """Network-on-Chip implementation."""

    def __init__(self, config: InterconnectConfig, grid_size: Tuple[int, int] = (4, 4)):
        self.config = config
        self.grid_size = grid_size
        self.routers: List[List[Router]] = []
        self.links: List[Link] = []
        self.interfaces: List[NetworkInterface] = []
        self.performance_metrics = {
            "total_packets": 0,
            "average_latency": 0.0,
            "throughput": 0.0,
            "network_utilization": 0.0,
        }
        self._initialize_network()

    def _initialize_network(self):
        """Initialize the NoC network."""
        # Create routers
        for y in range(self.grid_size[1]):
            router_row = []
            for x in range(self.grid_size[0]):
                router_id = y * self.grid_size[0] + x
                router = Router(router_id, x, y, self.config)
                router_row.append(router)
            self.routers.append(router_row)

        # Create links
        self._create_links()

        # Create network interfaces
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                interface_id = y * self.grid_size[0] + x
                interface = NetworkInterface(interface_id, self.routers[y][x])
                self.interfaces.append(interface)

    def _create_links(self):
        """Create links between routers based on topology."""
        if self.config.topology == NetworkTopology.MESH:
            self._create_mesh_links()
        elif self.config.topology == NetworkTopology.TORUS:
            self._create_torus_links()
        elif self.config.topology == NetworkTopology.TREE:
            self._create_tree_links()

    def _create_mesh_links(self):
        """Create mesh topology links."""
        link_id = 0
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                router_id = y * self.grid_size[0] + x

                # East link
                if x < self.grid_size[0] - 1:
                    neighbor_id = router_id + 1
                    link = Link(
                        link_id,
                        router_id,
                        neighbor_id,
                        self.config.bandwidth,
                        self.config.latency,
                    )
                    self.links.append(link)
                    self.routers[y][x].links.append(link)
                    link_id += 1

                # North link
                if y < self.grid_size[1] - 1:
                    neighbor_id = router_id + self.grid_size[0]
                    link = Link(
                        link_id,
                        router_id,
                        neighbor_id,
                        self.config.bandwidth,
                        self.config.latency,
                    )
                    self.links.append(link)
                    self.routers[y][x].links.append(link)
                    link_id += 1

    def _create_torus_links(self):
        """Create torus topology links."""
        self._create_mesh_links()
        # Add wrap-around links for torus
        # Implementation would add additional links for wrap-around

    def _create_tree_links(self):
        """Create tree topology links."""
        # Binary tree topology with root at (0,0)
        link_id = 0

        # Create a binary tree structure where each node has at most 2 children
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                router_id = y * self.grid_size[0] + x

                # Parent-child relationships in binary tree
                # Left child: 2*i + 1, Right child: 2*i + 2
                left_child_id = 2 * router_id + 1
                right_child_id = 2 * router_id + 2

                # Create link to left child if it exists
                if left_child_id < self.grid_size[0] * self.grid_size[1]:
                    left_child_y = left_child_id // self.grid_size[0]
                    left_child_x = left_child_id % self.grid_size[0]

                    if (
                        left_child_y < self.grid_size[1]
                        and left_child_x < self.grid_size[0]
                    ):
                        link = Link(
                            link_id,
                            router_id,
                            left_child_id,
                            self.config.bandwidth,
                            self.config.latency,
                        )
                        self.links.append(link)
                        self.routers[y][x].links.append(link)
                        link_id += 1

                # Create link to right child if it exists
                if right_child_id < self.grid_size[0] * self.grid_size[1]:
                    right_child_y = right_child_id // self.grid_size[0]
                    right_child_x = right_child_id % self.grid_size[0]

                    if (
                        right_child_y < self.grid_size[1]
                        and right_child_x < self.grid_size[0]
                    ):
                        link = Link(
                            link_id,
                            router_id,
                            right_child_id,
                            self.config.bandwidth,
                            self.config.latency,
                        )
                        self.links.append(link)
                        self.routers[y][x].links.append(link)
                        link_id += 1

    def send_message(
        self, source_id: int, destination_id: int, message: NetworkMessage
    ) -> bool:
        """Send a message through the NoC."""
        if source_id < len(self.interfaces):
            return self.interfaces[source_id].send_message(message)
        return False

    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status."""
        total_utilization = sum(link.utilization for link in self.links)
        avg_utilization = total_utilization / len(self.links) if self.links else 0

        return {
            "topology": self.config.topology.value,
            "grid_size": self.grid_size,
            "num_routers": len(self.routers) * len(self.routers[0])
            if self.routers
            else 0,
            "num_links": len(self.links),
            "average_utilization": avg_utilization,
            "performance_metrics": self.performance_metrics,
        }


class CrossbarSwitch:
    """Crossbar switch implementation."""

    def __init__(self, num_inputs: int, num_outputs: int, config: InterconnectConfig):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.config = config
        self.input_buffers: List[List[Packet]] = [[] for _ in range(num_inputs)]
        self.output_buffers: List[List[Packet]] = [[] for _ in range(num_outputs)]
        self.crossbar_matrix = [
            [False for _ in range(num_outputs)] for _ in range(num_inputs)
        ]
        self.performance_metrics = {
            "packets_switched": 0,
            "conflicts": 0,
            "utilization": 0.0,
        }

    def configure_connection(self, input_port: int, output_port: int) -> bool:
        """Configure a connection in the crossbar."""
        if (
            input_port < self.num_inputs
            and output_port < self.num_outputs
            and not self.crossbar_matrix[input_port][output_port]
        ):
            self.crossbar_matrix[input_port][output_port] = True
            return True
        return False

    def switch_packet(self, input_port: int, output_port: int, packet: Packet) -> bool:
        """Switch a packet from input to output port."""
        if (
            input_port < self.num_inputs
            and output_port < self.num_outputs
            and self.crossbar_matrix[input_port][output_port]
        ):
            self.output_buffers[output_port].append(packet)
            self.performance_metrics["packets_switched"] += 1
            return True
        return False


# Factory functions for creating different interconnect configurations


def create_mesh_noc(
    grid_size: Tuple[int, int] = (4, 4), config: Optional[InterconnectConfig] = None
) -> NoC:
    """Create a mesh NoC configuration."""
    if config is None:
        config = InterconnectConfig(topology=NetworkTopology.MESH)
    return NoC(config, grid_size)


def create_torus_noc(
    grid_size: Tuple[int, int] = (4, 4), config: Optional[InterconnectConfig] = None
) -> NoC:
    """Create a torus NoC configuration."""
    if config is None:
        config = InterconnectConfig(topology=NetworkTopology.TORUS)
    return NoC(config, grid_size)


def create_tree_noc(
    grid_size: Tuple[int, int] = (4, 4), config: Optional[InterconnectConfig] = None
) -> NoC:
    """Create a tree NoC configuration."""
    if config is None:
        config = InterconnectConfig(topology=NetworkTopology.TREE)
    return NoC(config, grid_size)


def create_crossbar_interconnect(
    num_inputs: int = 16,
    num_outputs: int = 16,
    config: Optional[InterconnectConfig] = None,
) -> CrossbarSwitch:
    """Create a crossbar interconnect configuration."""
    if config is None:
        config = InterconnectConfig(topology=NetworkTopology.CROSSBAR)
    return CrossbarSwitch(num_inputs, num_outputs, config)


# Export all the classes and functions
__all__ = [
    "InterconnectConfig",
    "NetworkTopology",
    "RoutingAlgorithm",
    "FlowControl",
    "Router",
    "NetworkInterface",
    "Link",
    "Packet",
    "NetworkMessage",
    "NoC",
    "CrossbarSwitch",
    "create_mesh_noc",
    "create_torus_noc",
    "create_tree_noc",
    "create_crossbar_interconnect",
]
