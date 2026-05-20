"""
Die-to-die interconnect model.

Models three topologies:
  - Ring: unidirectional or bidirectional ring
  - Mesh: 2D grid with nearest-neighbor links
  - All-to-all: full crossbar (every chiplet has a direct link to every other)

Each topology is parameterized by:
  - num_chiplets: number of nodes
  - link_bw: bandwidth per link in bytes/cycle
  - link_latency: per-hop latency in cycles

Supports standard collective operations:
  - broadcast: one node sends data to all others
  - scatter: one node sends distinct chunks to each other node
  - gather: all nodes send distinct chunks to one node
  - allreduce: all nodes contribute partial sums, all receive the result

Transfer times include serialization delay (data_size / link_bw) and
multi-hop latency where applicable. Contention is modeled as serialized
access on shared links.
"""

import math
from dataclasses import dataclass
from enum import Enum


class Topology(Enum):
    RING = "ring"
    MESH = "mesh"
    ALL_TO_ALL = "all_to_all"


@dataclass
class TransferResult:
    cycles: int
    data_bytes: int
    effective_bw: float  # bytes/cycle achieved
    description: str


class Interconnect:
    def __init__(self,
                 num_chiplets: int,
                 topology: Topology,
                 link_bw: float,
                 link_latency: int = 10,
                 element_bytes: int = 2):
        """
        Args:
            num_chiplets: number of chiplets in the system
            topology: ring, mesh, or all_to_all
            link_bw: bytes per cycle per link
            link_latency: fixed per-hop latency in cycles
            element_bytes: bytes per data element (2 for fp16, 4 for fp32)
        """
        self.P = num_chiplets
        self.topology = topology
        self.link_bw = link_bw
        self.link_latency = link_latency
        self.element_bytes = element_bytes

        if topology == Topology.MESH:
            self.mesh_dim = math.isqrt(num_chiplets)
            if self.mesh_dim * self.mesh_dim != num_chiplets:
                # Find closest rectangular mesh
                self.mesh_rows = self.mesh_dim
                self.mesh_cols = math.ceil(num_chiplets / self.mesh_dim)
            else:
                self.mesh_rows = self.mesh_dim
                self.mesh_cols = self.mesh_dim

    def _serial_time(self, data_bytes: int) -> float:
        """Time to push data_bytes through a single link."""
        return data_bytes / self.link_bw

    def broadcast(self, num_elements: int) -> TransferResult:
        """One node sends identical data to all other nodes."""
        data_bytes = num_elements * self.element_bytes
        P = self.P

        if P <= 1:
            return TransferResult(0, 0, 0, "single chiplet, no transfer")

        if self.topology == Topology.ALL_TO_ALL:
            # Direct link to every node, all transfers happen in parallel
            cycles = self._serial_time(data_bytes) + self.link_latency

        elif self.topology == Topology.RING:
            # Pipelined broadcast around the ring.
            # Split data into P-1 chunks, pipeline around.
            # Time = (P-1) * (chunk_size/bw + latency)
            # With large data, serialization dominates:
            # Total ~= data_bytes/bw + (P-1)*latency
            # But without pipelining (store-and-forward):
            # Total = (P-1) * (data_bytes/bw + latency)
            # Use the pipelined model since it's standard.
            chunk = data_bytes / (P - 1) if P > 1 else data_bytes
            cycles = (self._serial_time(chunk) + self.link_latency) * (P - 1)

        elif self.topology == Topology.MESH:
            # Spanning tree broadcast on mesh.
            # Diameter = (mesh_rows - 1) + (mesh_cols - 1)
            # Store-and-forward at each hop.
            diameter = (self.mesh_rows - 1) + (self.mesh_cols - 1)
            cycles = diameter * (self._serial_time(data_bytes) + self.link_latency)

        else:
            raise ValueError(f"Unknown topology: {self.topology}")

        cycles = int(math.ceil(cycles))
        eff_bw = data_bytes / cycles if cycles > 0 else 0
        return TransferResult(cycles, data_bytes, eff_bw,
                              f"broadcast {num_elements} elements via {self.topology.value}")

    def scatter(self, total_elements: int) -> TransferResult:
        """One node sends a distinct chunk of total_elements/P to each other node."""
        P = self.P
        if P <= 1:
            return TransferResult(0, 0, 0, "single chiplet")

        chunk_elements = total_elements // P
        chunk_bytes = chunk_elements * self.element_bytes
        data_bytes = chunk_bytes * (P - 1)  # total data moved

        if self.topology == Topology.ALL_TO_ALL:
            # All chunks sent in parallel
            cycles = self._serial_time(chunk_bytes) + self.link_latency

        elif self.topology == Topology.RING:
            # Each hop forwards (P-1-i) chunks at step i
            # Total time = sum over i of (P-1-i)*chunk_bytes/bw + latency
            # = chunk_bytes/bw * P*(P-1)/2 + (P-1)*latency
            # Optimized scatter-reduce: (P-1) * (chunk_bytes/bw + latency)
            cycles = (P - 1) * (self._serial_time(chunk_bytes) + self.link_latency)

        elif self.topology == Topology.MESH:
            diameter = (self.mesh_rows - 1) + (self.mesh_cols - 1)
            # Worst case: diameter hops, each forwarding a chunk
            cycles = diameter * (self._serial_time(chunk_bytes) * P + self.link_latency)

        cycles = int(math.ceil(cycles))
        eff_bw = data_bytes / cycles if cycles > 0 else 0
        return TransferResult(cycles, data_bytes, eff_bw,
                              f"scatter {total_elements} elements across {P} chiplets")

    def allreduce(self, num_elements: int) -> TransferResult:
        """
        All chiplets contribute num_elements of partial sums.
        All chiplets receive the fully reduced result.

        Uses ring allreduce where applicable:
          Phase 1 (reduce-scatter): P-1 steps, each sending num_elements/P
          Phase 2 (allgather): P-1 steps, each sending num_elements/P
          Total data per link = 2 * (P-1)/P * num_elements
        """
        P = self.P
        data_bytes = num_elements * self.element_bytes

        if P <= 1:
            return TransferResult(0, 0, 0, "single chiplet")

        if self.topology == Topology.ALL_TO_ALL:
            # Reduce to one node, broadcast back
            # Each node sends data_bytes to root: serialized = data_bytes * (P-1) / bw
            # But with all-to-all links, all sends are parallel
            # Reduce: data_bytes/bw + latency
            # Broadcast: data_bytes/bw + latency
            cycles = 2 * (self._serial_time(data_bytes) + self.link_latency)

        elif self.topology == Topology.RING:
            # Ring allreduce: optimal bandwidth utilization
            chunk_bytes = data_bytes / P
            # Phase 1: reduce-scatter, P-1 steps
            # Phase 2: allgather, P-1 steps
            # Each step: chunk_bytes / bw + latency
            steps = 2 * (P - 1)
            cycles = steps * (self._serial_time(chunk_bytes) + self.link_latency)

        elif self.topology == Topology.MESH:
            # Hierarchical: reduce along rows, then along cols, then broadcast back
            # Row reduce: (mesh_cols - 1) steps
            # Col reduce: (mesh_rows - 1) steps
            # Reverse broadcast: same cost
            row_steps = self.mesh_cols - 1
            col_steps = self.mesh_rows - 1
            total_steps = 2 * (row_steps + col_steps)
            chunk_bytes = data_bytes / max(self.mesh_rows, self.mesh_cols)
            cycles = total_steps * (self._serial_time(chunk_bytes) + self.link_latency)

        cycles = int(math.ceil(cycles))
        total_moved = int(2 * (P - 1) / P * data_bytes)
        eff_bw = total_moved / cycles if cycles > 0 else 0
        return TransferResult(cycles, total_moved, eff_bw,
                              f"allreduce {num_elements} elements across {P} chiplets")

    def point_to_point(self, num_elements: int, hops: int = 1) -> TransferResult:
        """Direct transfer between two chiplets separated by `hops` links."""
        data_bytes = num_elements * self.element_bytes
        cycles = int(math.ceil(
            hops * (self._serial_time(data_bytes) + self.link_latency)
        ))
        eff_bw = data_bytes / cycles if cycles > 0 else 0
        return TransferResult(cycles, data_bytes, eff_bw,
                              f"p2p {num_elements} elements, {hops} hops")
