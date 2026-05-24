import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

def build_graph_from_returns(returns_df, top_edge_fraction=0.2):
    """
    Build undirected graph with edge weights = 1 - |correlation|.
    Returns:
        G: networkx Graph
        edge_list: list of (i, j, weight) with integer indices
        nodes: list of ticker symbols in the order used for indices
    """
    corr = returns_df.corr().fillna(0)
    np.fill_diagonal(corr.values, 0)
    dist = 1 - np.abs(corr)   # distance = 1 - |correlation|
    n_edges = len(dist) * (len(dist)-1) // 2
    keep = int(n_edges * top_edge_fraction)
    triu = np.triu_indices_from(dist, k=1)
    flat_dist = dist.values[triu]
    if keep >= len(flat_dist):
        keep = len(flat_dist) - 1
    thresh = np.sort(flat_dist)[keep] if keep > 0 else 1.0
    G = nx.Graph()
    nodes = returns_df.columns.tolist()
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
    Build sparse incidence matrix B (nodes x edges).
    Orientation: from node i to node j (i -> j)
    """
    n_nodes = len(node_list)
    n_edges = len(edge_list)
    row = []
    col = []
    data = []
    for e_idx, (i, j, _) in enumerate(edge_list):
        # i -> j: -1 at i, +1 at j
        row.append(i)
        col.append(e_idx)
        data.append(-1.0)
        row.append(j)
        col.append(e_idx)
        data.append(1.0)
    B = sp.csr_matrix((data, (row, col)), shape=(n_nodes, n_edges))
    return B

def compute_edge_flow(returns_df, edge_list):
    """
    Edge flow = return_i - return_j for the last day in the window.
    Returns 1D numpy array of length len(edge_list).
    """
    last_ret = returns_df.iloc[-1]  # Series indexed by ticker
    f = np.zeros(len(edge_list), dtype=float)
    for idx, (i, j, _) in enumerate(edge_list):
        f[idx] = last_ret.iloc[i] - last_ret.iloc[j]
    return f

def hodge_decomposition(B, f, eps=1e-8, max_iter=100):
    """
    Decompose edge flow f into gradient, curl, and harmonic components.
    Uses the graph Helmholtzian (1-Laplacian) method.
    
    Parameters:
        B : sparse matrix (nodes x edges)
        f : 1D array (edges,)
        eps : regularization for solving linear systems
        max_iter : max iterations for LSQR
    
    Returns:
        grad : gradient component (edges,)
        curl : curl component (edges,)
        harmonic : harmonic component (edges,)
        potential : node potentials (nodes,)
    """
    # Ensure f is 1D
    f = f.ravel()
    n_edges = B.shape[1]
    assert len(f) == n_edges, f"f length {len(f)} != number of edges {n_edges}"
    
    # ---- Gradient component: solve B * x = f (least squares) ----
    # Solve for node potentials x such that Bx approximates f
    # Use LSQR on the normal equations: B^T B x = B^T f
    BtB = B.T @ B
    Btf = B.T @ f
    # Regularize
    BtB_reg = BtB + eps * sp.eye(BtB.shape[0], format='csr')
    x = lsqr(BtB_reg, Btf, atol=1e-6, btol=1e-6, iter_lim=max_iter)[0]
    grad = (B @ x).ravel()
    
    # ---- Harmonic component: projection onto kernel of L1 = B B^T + B^T B ----
    # Solve L1 * h = L1 * f, then harmonic = f - h
    L1 = B @ B.T + B.T @ B   # 1-Laplacian (edges x edges)
    L1 = L1 + eps * sp.eye(L1.shape[0], format='csr')
    rhs = L1 @ f
    h = lsqr(L1, rhs, atol=1e-6, btol=1e-6, iter_lim=max_iter)[0]
    harmonic = (f - h).ravel()
    
    # Curl component = f - grad - harmonic
    curl = f - grad - harmonic
    
    return grad, curl, harmonic, x

def get_node_scores(harmonic_flow, B):
    """
    Compute divergence of harmonic flow at each node: B^T * harmonic_flow.
    Returns 1D array of length n_nodes.
    """
    return (B.T @ harmonic_flow).ravel()
