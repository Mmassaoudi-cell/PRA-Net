"""
Power-system network utilities: build pandapower IEEE case30 / case118 networks
(standard MATPOWER topologies, since MATPOWER itself is MATLAB-only) and derive
the DC (linearized) measurement Jacobian used for FDIA construction and the
bad-data-detection (BDD) chi-square / largest-normalized-residual test.

This substitutes for the source paper's ref [32] (MATPOWER case files, same
topology/parameters) and approximates ref [33]'s load/state generation, which
is not publicly distributed with the paper (documented as an assumption in
SOURCE_PAPER_AUDIT.md).
"""
from __future__ import annotations
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import networkx as nx


def build_network(system: str):
    if system == "case30":
        net = pn.case30()
    elif system == "case118":
        net = pn.case118()
    else:
        raise ValueError(f"unknown system {system}")
    pp.runpp(net, algorithm="nr", init="flat")
    return net


def adjacency_graph(net) -> nx.Graph:
    """Bus adjacency graph from branch (line + trafo) topology, used for the
    topology-based neighborhood-consistency rule in localization (Sec. IV-B)."""
    g = nx.Graph()
    g.add_nodes_from(net.bus.index.tolist())
    for _, row in net.line.iterrows():
        g.add_edge(int(row.from_bus), int(row.to_bus))
    for _, row in net.trafo.iterrows():
        g.add_edge(int(row.hv_bus), int(row.lv_bus))
    return g


def dc_jacobian(net):
    """
    Build the DC power-flow Jacobian H_dc such that z_dc = H_dc @ theta,
    where z_dc stacks [P_flow (all branches, from-side), P_inj (all buses)]
    and theta are bus voltage angles (radians), slack bus angle fixed at 0.

    This is the classical linear FDIA/BDD model (Liu et al. 2011, the source
    paper's own foundational reference [3]): an attack a = H_dc @ c is
    exactly unobservable to a WLS/chi-square BDD test on z_dc for any c
    supported on non-slack buses.
    """
    n_bus = len(net.bus)
    slack_bus = int(net.ext_grid.bus.iloc[0])
    non_slack = [b for b in net.bus.index if b != slack_bus]
    idx_map = {b: i for i, b in enumerate(non_slack)}  # bus -> reduced-angle index

    branches = []
    for _, row in net.line.iterrows():
        fb, tb = int(row.from_bus), int(row.to_bus)
        x_pu = row.x_ohm_per_km * row.length_km * net.sn_mva / (net.bus.loc[fb, "vn_kv"] ** 2)
        branches.append((fb, tb, x_pu))
    for _, row in net.trafo.iterrows():
        fb, tb = int(row.hv_bus), int(row.lv_bus)
        x_pu = row.vk_percent / 100.0 * (net.sn_mva / row.sn_mva)
        branches.append((fb, tb, max(x_pu, 1e-3)))

    n_branch = len(branches)
    n_red = len(non_slack)

    H_flow = np.zeros((n_branch, n_red))
    for k, (fb, tb, x) in enumerate(branches):
        b = 1.0 / x
        if fb in idx_map:
            H_flow[k, idx_map[fb]] += b
        if tb in idx_map:
            H_flow[k, idx_map[tb]] -= b

    H_inj = np.zeros((n_bus, n_red))
    for k, (fb, tb, x) in enumerate(branches):
        b = 1.0 / x
        if fb in idx_map:
            H_inj[fb, idx_map[fb]] += b
        if tb in idx_map:
            H_inj[tb, idx_map[tb]] += b
        if fb in idx_map and tb in idx_map:
            H_inj[fb, idx_map[tb]] -= b
            H_inj[tb, idx_map[fb]] -= b
        elif fb in idx_map:
            pass
        elif tb in idx_map:
            pass

    # H_flow/H_inj above are in per-unit (assuming |V|~1 pu); scale to MW to
    # match the AC-power-flow-derived measurement vectors, which are in MW.
    H_flow = H_flow * net.sn_mva
    H_inj = H_inj * net.sn_mva

    H_dc = np.vstack([H_flow, H_inj])
    return {
        "H_dc": H_dc,
        "H_flow": H_flow,
        "H_inj": H_inj,
        "non_slack": non_slack,
        "slack_bus": slack_bus,
        "branches": branches,
        "n_bus": n_bus,
        "n_branch": n_branch,
    }
