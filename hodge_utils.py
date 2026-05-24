import numpy as np
import pandas as pd
import networkx as nx

def build_graph_from_returns(returns_df, top_edge_fraction=0.2):
    """
    Build undirected graph from correlation distance.
    Returns:
        G: networkx Graph
        edge_list: list of (i, j, weight) where i,j are integer indices
        nodes: list of ticker symbols in order
    """
    corr = returns_df.corr().fillna(0)
    np.fill_diagonal(corr.values, 0)
    dist = 1 - np.abs(corr)          # distance = 1 - |correlation|
    n_edges_possible = len(dist) * (len(dist) - 1) // 2
    keep = int(n_edges_possible * top_edge_fraction)
    triu = np.triu_indices_from(dist, k=1)
    flat_dist = dist.values[triu]
    if keep >= len(flat_dist):
        keep = len(flat_dist) - 1
    thresh = np.sort(flat_dist)[keep] if keep > 0 else 1.0

    nodes = returns_df.columns.tolist()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    edge_list = []
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i >= j:
                continue
            d = dist.iloc[i, j]
            if d <= thresh:
                G.add_edge(u, v, weight=d)
                edge_list.append((i, j, d))
    return G, edge_list, nodes

def incidence_matrix(node_list, edge_list):
    """
    Build oriented incidence matrix B of shape (edges, nodes).
    For edge e from i to j: B[e, i] = -1, B[e, j] = +1.
    """
    n_nodes = len(node_list)
    n_edges = len(edge_list)
    B = np.zeros((n_edges, n_nodes), dtype=float)
    for e_idx, (i, j, _) in enumerate(edge_list):
        B[e_idx, i] = -1.0
        B[e_idx, j] = 1.0
    return B

def compute_edge_flow(returns_df, edge_list):
    """Edge flow = return_i - return_j on the last day."""
    last_ret = returns_df.iloc[-1].values   # numpy array in node order
    f = np.zeros(len(edge_list), dtype=float)
    for idx, (i, j, _) in enumerate(edge_list):
        f[idx] = last_ret[i] - last_ret[j]
    return f

def hodge_decomposition(B, f, eps=1e-8):
    """
    Hodge decomposition using dense linear algebra.
    B : (edges, nodes)
    f : (edges,)
    Returns (grad, curl, harmonic, potential)
    """
    n_edges = B.shape[0]
    f = f.ravel()
    assert len(f) == n_edges, f"Length mismatch: f {len(f)} vs edges {n_edges}"

    # ---- Gradient component: solve B^T B x = B^T f ----
    BtB = B.T @ B + eps * np.eye(B.shape[1])
    Btf = B.T @ f                     # (nodes,)
    potential = np.linalg.lstsq(BtB, Btf, rcond=None)[0]
    grad = B @ potential              # (edges,)

    # ---- Harmonic component: solve L1 h = L1 f, L1 = B B^T + B^T B ----
    L1 = B @ B.T + B.T @ B            # (edges, edges)
    L1_reg = L1 + eps * np.eye(n_edges)
    rhs = L1_reg @ f
    h = np.linalg.lstsq(L1_reg, rhs, rcond=None)[0]
    harmonic = f - h

    # ---- Curl component ----
    curl = f - grad - harmonic

    return grad, curl, harmonic, potential

def get_node_scores(harmonic_flow, B):
    """Divergence of harmonic flow at nodes = B^T @ harmonic_flow (size nodes)."""
    return B.T @ harmonic_flow
