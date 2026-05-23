import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

def build_graph_from_returns(returns_df, top_edge_fraction=0.2):
    """
    Build undirected graph with edge weights = 1 - correlation.
    Returns:
        G: networkx Graph
        edge_list: list of (i, j, weight)
        node_list: list of tickers in order
    """
    corr = returns_df.corr().fillna(0)
    np.fill_diagonal(corr.values, 0)
    # distance
    dist = 1 - abs(corr)   # or 1 - correlation? Use absolute for symmetry
    # keep top fraction of edges (smallest distance -> strongest correlation)
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

def incidence_matrix(G, node_list, edge_list):
    """Build sparse incidence matrix B1 (nodes x edges)."""
    n_nodes = len(node_list)
    n_edges = len(edge_list)
    row = []
    col = []
    data = []
    for e_idx, (i, j, w) in enumerate(edge_list):
        # orientation: i -> j
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
    Flow on edge (i,j) = return_i - return_j (on the last day of the window).
    Returns a vector f of length n_edges.
    """
    last_ret = returns_df.iloc[-1]  # last day return
    f = np.zeros(len(edge_list))
    for idx, (i, j, _) in enumerate(edge_list):
        f[idx] = last_ret.iloc[i] - last_ret.iloc[j]
    return f

def hodge_decomposition(B, f, eps=1e-8, max_iter=100):
    """
    Decompose edge flow f into gradient, curl, and harmonic components.
    Returns:
        grad: gradient component (B * x)
        curl: curl component (f - grad - harmonic)
        harmonic: harmonic component
        potential: node potential x
    """
    # Solve for potential x: minimize ||B x - f||^2
    # Normal equations: B^T B x = B^T f
    BtB = B.T @ B
    Btf = B.T @ f
    # Regularise
    BtB_reg = BtB + eps * sp.eye(BtB.shape[0], format='csr')
    # Use LSQR (or sparse solver)
    x = lsqr(BtB_reg, Btf, atol=1e-6, btol=1e-6, iter_lim=max_iter)[0]
    grad = B @ x
    # Harmonic = residual after removing gradient from the nullspace of B^T and B?
    # Standard approach: harmonic = f - grad - curl, where curl is projection onto cycle space.
    # Simplify: curl = (I - B (B^T B)^+ B^T) f, harmonic = f - grad - curl.
    # Instead we compute harmonic as projection onto kernel of Laplacian L1 = B B^T + B^T B? 
    # For graph Helmholtzian: L1 = B B^T (0-form Laplacian) + B^T B (1-form Laplacian).
    # We'll compute curl by solving for edge curl potential? Use pseudoinverse.
    # A robust method: 
    #   grad = B x
    #   harmonic = f - B x - curl, with curl = B^T y? Actually curl on edges corresponds to 2-cochains.
    # Simpler: Use networkx to compute cycle basis, project f onto cycle space.
    # Let's implement a practical approach: solve for harmonic as the part orthogonal to both gradient and curl.
    # We'll compute the projection onto the kernel of L1.
    L1 = B @ B.T + B.T @ B   # 1-Laplacian
    # Regularise and solve L1 * h = L1 * f? Actually harmonic satisfies L1 h = 0 and (f - h) is in range(L1).
    # So h = f - L1^+ L1 f.
    # Use LSQR to solve L1 * u = L1 * f, then h = f - u.
    L1 = L1 + eps * sp.eye(L1.shape[0], format='csr')
    rhs = L1 @ f
    u = lsqr(L1, rhs, atol=1e-6, btol=1e-6, iter_lim=max_iter)[0]
    harmonic = f - u
    # Then curl = f - grad - harmonic
    curl = f - grad - harmonic
    return grad, curl, harmonic, x

def get_node_scores(harmonic_flow, B):
    """Compute divergence of harmonic flow at nodes: B^T * harmonic."""
    div = B.T @ harmonic_flow
    return div
