"""
Die-to-die interconnect model.

Models four topologies:
  - Ring:         bidirectional ring, O(P) links
  - Mesh:         2-D grid with nearest-neighbour links, O(P) links
  - All-to-all:   full crossbar, every chiplet has a direct link, O(P²) links
  - Hierarchical: chiplets grouped into clusters; intra-cluster links are
                  faster than inter-cluster links (models real chiplet packages
                  such as AMD MI300X where chiplets share a local bus but reach
                  remote chiplets over a slower die-to-die fabric)

Each topology is parameterised by:
  - num_chiplets:   number of nodes
  - link_bw:        bandwidth per link in bytes/cycle
  - link_latency:   per-hop latency in cycles

Hierarchical-only parameters:
  - cluster_size:        chiplets per cluster (default: isqrt(P))
  - intra_bw_factor:     link_bw multiplier for intra-cluster links (default 2.0)
  - intra_latency_factor: link_latency multiplier for intra-cluster (default 0.5)

Supported collectives:
  broadcast  – one node → all others (identical data)
  scatter    – one node → each other node (distinct chunks)
  allreduce  – partial sums from all nodes → fully reduced result at all nodes

Transfer times include serialisation delay (bytes / bw) and multi-hop latency.
"""

import math
from dataclasses import dataclass
from enum import Enum


class Topology(Enum):
    RING         = "ring"
    MESH         = "mesh"
    ALL_TO_ALL   = "all_to_all"
    HIERARCHICAL = "hierarchical"


@dataclass
class TransferResult:
    cycles: int
    data_bytes: int
    effective_bw: float   # bytes/cycle achieved
    description: str


class Interconnect:
    def __init__(self,
                 num_chiplets: int,
                 topology: Topology,
                 link_bw: float,
                 link_latency: int = 10,
                 element_bytes: int = 2,
                 # ── hierarchical parameters ───────────────────────────
                 cluster_size: int = None,
                 intra_bw_factor: float = 2.0,
                 intra_latency_factor: float = 0.5):
        """
        Args:
            num_chiplets:        total number of chiplets
            topology:            ring | mesh | all_to_all | hierarchical
            link_bw:             bytes per cycle per link (inter-cluster bw
                                 for hierarchical)
            link_latency:        fixed per-hop latency in cycles (inter-cluster
                                 latency for hierarchical)
            element_bytes:       bytes per data element (2 = fp16, 4 = fp32)
            cluster_size:        chiplets per cluster for hierarchical topology;
                                 defaults to isqrt(num_chiplets)
            intra_bw_factor:     intra-cluster bandwidth = link_bw × factor
            intra_latency_factor: intra-cluster latency  = link_latency × factor
        """
        self.P = num_chiplets
        self.topology = topology
        self.link_bw = link_bw
        self.link_latency = link_latency
        self.element_bytes = element_bytes

        if topology == Topology.MESH:
            self.mesh_dim = math.isqrt(num_chiplets)
            if self.mesh_dim * self.mesh_dim != num_chiplets:
                self.mesh_rows = self.mesh_dim
                self.mesh_cols = math.ceil(num_chiplets / self.mesh_dim)
            else:
                self.mesh_rows = self.mesh_dim
                self.mesh_cols = self.mesh_dim

        elif topology == Topology.HIERARCHICAL:
            if cluster_size is None:
                cluster_size = max(2, math.isqrt(num_chiplets))
            self.cluster_size = cluster_size
            self.num_clusters = math.ceil(num_chiplets / cluster_size)
            # intra-cluster links are shorter → higher bandwidth, lower latency
            self.intra_bw      = link_bw * intra_bw_factor
            self.inter_bw      = link_bw
            self.intra_latency = max(1, int(link_latency * intra_latency_factor))
            self.inter_latency = link_latency

    # ── helpers ───────────────────────────────────────────────────────────────

    def _serial(self, data_bytes: float, bw: float) -> float:
        return data_bytes / bw

    # ── collectives ───────────────────────────────────────────────────────────

    def broadcast(self, num_elements: int) -> TransferResult:
        """One node sends identical data to all other nodes."""
        data_bytes = num_elements * self.element_bytes
        P = self.P

        if P <= 1:
            return TransferResult(0, 0, 0.0, "single chiplet, no transfer")

        if self.topology == Topology.ALL_TO_ALL:
            # Direct link to every node; all transfers in parallel
            cycles = self._serial(data_bytes, self.link_bw) + self.link_latency

        elif self.topology == Topology.RING:
            # Pipelined ring broadcast: split into (P-1) chunks
            chunk = data_bytes / (P - 1)
            cycles = (self._serial(chunk, self.link_bw) + self.link_latency) * (P - 1)

        elif self.topology == Topology.MESH:
            # Spanning-tree broadcast on mesh (store-and-forward)
            diameter = (self.mesh_rows - 1) + (self.mesh_cols - 1)
            cycles = diameter * (self._serial(data_bytes, self.link_bw) + self.link_latency)

        elif self.topology == Topology.HIERARCHICAL:
            # Phase 1: source broadcasts to all other cluster leaders (inter)
            nc = self.num_clusters
            chunk_inter = data_bytes / max(nc - 1, 1)
            inter_cycles = (self._serial(chunk_inter, self.inter_bw)
                            + self.inter_latency) * (nc - 1)
            # Phase 2: each leader broadcasts within its cluster (intra)
            cs = self.cluster_size
            chunk_intra = data_bytes / max(cs - 1, 1)
            intra_cycles = (self._serial(chunk_intra, self.intra_bw)
                            + self.intra_latency) * (cs - 1)
            # Phases are pipelined: intra starts as soon as first chunk arrives
            cycles = inter_cycles + intra_cycles

        else:
            raise ValueError(f"Unknown topology: {self.topology}")

        cycles = int(math.ceil(cycles))
        eff_bw = data_bytes / cycles if cycles > 0 else 0.0
        return TransferResult(cycles, data_bytes, eff_bw,
                              f"broadcast {num_elements} elements via {self.topology.value}")

    def scatter(self, total_elements: int) -> TransferResult:
        """One node sends a distinct chunk of total_elements/P to each other node."""
        P = self.P
        if P <= 1:
            return TransferResult(0, 0, 0.0, "single chiplet")

        chunk_elements = total_elements // P
        chunk_bytes = chunk_elements * self.element_bytes
        data_bytes = chunk_bytes * (P - 1)

        if self.topology == Topology.ALL_TO_ALL:
            cycles = self._serial(chunk_bytes, self.link_bw) + self.link_latency

        elif self.topology == Topology.RING:
            cycles = (P - 1) * (self._serial(chunk_bytes, self.link_bw) + self.link_latency)

        elif self.topology == Topology.MESH:
            diameter = (self.mesh_rows - 1) + (self.mesh_cols - 1)
            cycles = diameter * (self._serial(chunk_bytes, self.link_bw) * P
                                 + self.link_latency)

        elif self.topology == Topology.HIERARCHICAL:
            nc = self.num_clusters
            cs = self.cluster_size
            # Phase 1: scatter cluster-sized chunks to each cluster leader (inter)
            cluster_chunk_bytes = chunk_bytes * cs
            inter_cycles = (nc - 1) * (self._serial(cluster_chunk_bytes, self.inter_bw)
                                       + self.inter_latency)
            # Phase 2: each leader scatters within its cluster (intra)
            intra_cycles = (cs - 1) * (self._serial(chunk_bytes, self.intra_bw)
                                       + self.intra_latency)
            cycles = inter_cycles + intra_cycles

        else:
            raise ValueError(f"Unknown topology: {self.topology}")

        cycles = int(math.ceil(cycles))
        eff_bw = data_bytes / cycles if cycles > 0 else 0.0
        return TransferResult(cycles, data_bytes, eff_bw,
                              f"scatter {total_elements} elements across {P} chiplets")

    def allreduce(self, num_elements: int) -> TransferResult:
        """
        All chiplets contribute partial sums; all receive the fully reduced result.

        Ring allreduce (2-phase reduce-scatter + allgather):
          Each phase: P-1 steps, each sending num_elements/P per step.
          Total data per link = 2(P-1)/P × num_elements × element_bytes.

        Hierarchical allreduce:
          Phase 1 – ring allreduce within each cluster (intra links).
          Phase 2 – ring allreduce across cluster representatives (inter links).
          The two phases are sequential (cluster result must be ready before
          inter-cluster reduction begins).
        """
        P = self.P
        data_bytes = num_elements * self.element_bytes

        if P <= 1:
            return TransferResult(0, 0, 0.0, "single chiplet")

        if self.topology == Topology.ALL_TO_ALL:
            # Parallel reduce to root + broadcast: 2 × (data/bw + latency)
            cycles = 2 * (self._serial(data_bytes, self.link_bw) + self.link_latency)

        elif self.topology == Topology.RING:
            chunk_bytes = data_bytes / P
            steps = 2 * (P - 1)
            cycles = steps * (self._serial(chunk_bytes, self.link_bw) + self.link_latency)

        elif self.topology == Topology.MESH:
            # Hierarchical: reduce along rows then cols, broadcast back
            row_steps = self.mesh_cols - 1
            col_steps = self.mesh_rows - 1
            total_steps = 2 * (row_steps + col_steps)
            chunk_bytes = data_bytes / max(self.mesh_rows, self.mesh_cols)
            cycles = total_steps * (self._serial(chunk_bytes, self.link_bw) + self.link_latency)

        elif self.topology == Topology.HIERARCHICAL:
            nc = self.num_clusters
            cs = self.cluster_size
            # Phase 1: ring allreduce within each cluster
            chunk_intra = data_bytes / cs
            steps_intra = 2 * (cs - 1)
            intra_cycles = steps_intra * (self._serial(chunk_intra, self.intra_bw)
                                          + self.intra_latency)
            # Phase 2: ring allreduce across num_clusters representatives
            chunk_inter = data_bytes / nc
            steps_inter = 2 * (nc - 1)
            inter_cycles = steps_inter * (self._serial(chunk_inter, self.inter_bw)
                                          + self.inter_latency)
            cycles = intra_cycles + inter_cycles

        else:
            raise ValueError(f"Unknown topology: {self.topology}")

        cycles = int(math.ceil(cycles))
        total_moved = int(2 * (P - 1) / P * data_bytes)
        eff_bw = total_moved / cycles if cycles > 0 else 0.0
        return TransferResult(cycles, total_moved, eff_bw,
                              f"allreduce {num_elements} elements across {P} chiplets")

    def point_to_point(self, num_elements: int, hops: int = 1) -> TransferResult:
        """Direct transfer between two chiplets separated by `hops` links."""
        data_bytes = num_elements * self.element_bytes
        cycles = int(math.ceil(
            hops * (self._serial(data_bytes, self.link_bw) + self.link_latency)
        ))
        eff_bw = data_bytes / cycles if cycles > 0 else 0.0
        return TransferResult(cycles, data_bytes, eff_bw,
                              f"p2p {num_elements} elements, {hops} hops")

    # ── topology introspection ────────────────────────────────────────────────

    def connection_info(self) -> dict:
        """
        Return topology connection statistics useful for fair cross-topology
        comparisons.

        Keys:
          num_links          – number of bidirectional physical links
          bisection_bandwidth – minimum bandwidth across any balanced partition
                               of chiplets (bytes/cycle)
          diameter           – maximum hop-distance between any two chiplets
          avg_hops           – average hop-distance under uniform traffic
          link_bw            – per-link bandwidth (bytes/cycle)
          total_bandwidth    – num_links × link_bw (bytes/cycle)
        """
        P = self.P

        if self.topology == Topology.RING:
            num_links   = P                       # P bidirectional links
            bisect_bw   = 2 * self.link_bw        # cut ring at 2 points
            diameter    = P // 2
            avg_hops    = P / 4.0

        elif self.topology == Topology.MESH:
            r, c = self.mesh_rows, self.mesh_cols
            num_links   = r * (c - 1) + (r - 1) * c
            bisect_bw   = min(r, c) * self.link_bw
            diameter    = (r - 1) + (c - 1)
            avg_hops    = ((r - 1) + (c - 1)) / 2.0

        elif self.topology == Topology.ALL_TO_ALL:
            num_links   = P * (P - 1) // 2
            bisect_bw   = (P // 2) * (P // 2) * self.link_bw
            diameter    = 1
            avg_hops    = 1.0

        elif self.topology == Topology.HIERARCHICAL:
            cs = self.cluster_size
            nc = self.num_clusters
            # Intra-cluster: all-to-all within each cluster
            intra_links = nc * (cs * (cs - 1) // 2)
            # Inter-cluster: ring of cluster leaders
            inter_links = nc
            num_links   = intra_links + inter_links
            bisect_bw   = self.inter_bw           # bottleneck is inter-cluster
            diameter    = 1 + nc // 2             # 1 intra hop + inter-ring hops
            avg_hops    = 1.0 + nc / 4.0

        else:
            raise ValueError(f"Unknown topology: {self.topology}")

        return {
            "num_links":           num_links,
            "bisection_bandwidth": bisect_bw,
            "diameter":            diameter,
            "avg_hops":            avg_hops,
            "link_bw":             self.link_bw,
            "total_bandwidth":     num_links * self.link_bw,
        }
