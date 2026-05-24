import numpy as np
import networkx as nx

def build_graph_from_returns(returns_df, top_edge_fraction=0.2):
    corr = returns_df.corr().fillna(0)
    np.fill_diagonal(corr.values, 0)
    dist = 1 - np.abs(corr)
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
    n_nodes = len(node_list)
    n_edges = len(edge_list)
    B = np.zeros((n_edges, n_nodes), dtype=float)
    for e_idx, (i, j, _) in enumerate(edge_list):
        B[e_idx, i] = -1.0
        B[e_idx, j] = 1.0
    return B

def compute_edge_flow(returns_df, edge_list):
    last_ret = returns_df.iloc[-1].values
    f = np.zeros(len(edge_list), dtype=float)
    for idx, (i, j, _) in enumerate(edge_list):
        f[idx] = last_ret[i] - last_ret[j]
    return f

def cycle_basis_vectors(G, edge_list, nodes):
    edge_to_idx = {}
    for idx, (i, j, _) in enumerate(edge_list):
        edge_to_idx[(i, j)] = idx
        edge_to_idx[(j, i)] = idx
    cycles = nx.cycle_basis(G)
    basis = []
    for cycle_nodes in cycles:
        cycle_idx = [nodes.index(v) for v in cycle_nodes]
        flow = np.zeros(len(edge_list))
        for k in range(len(cycle_idx)):
            u = cycle_idx[k]
            v = cycle_idx[(k+1) % len(cycle_idx)]
            idx = edge_to_idx.get((u, v))
            if idx is not None:
                flow[idx] = 1.0
        basis.append(flow)
    if not basis:
        return np.zeros((len(edge_list), 0))
    basis_mat = np.array(basis).T
    Q, _ = np.linalg.qr(basis_mat, mode='reduced')
    return Q

def hodge_decomposition(B, f, G, edge_list, nodes, eps=1e-8):
    n_edges = B.shape[0]
    # Gradient component
    BtB = B.T @ B + eps * np.eye(B.shape[1])
    Btf = B.T @ f
    potential = np.linalg.lstsq(BtB, Btf, rcond=None)[0]
    grad = B @ potential
    # Residual after removing gradient
    residual = f - grad
    # Curl component (project onto cycle space)
    cycle_basis = cycle_basis_vectors(G, edge_list, nodes)
    if cycle_basis.shape[1] > 0:
        curl = cycle_basis @ (cycle_basis.T @ residual)
    else:
        curl = np.zeros_like(f)
    # Harmonic component = residual - curl
    harmonic = residual - curl
    return grad, curl, harmonic, potential

def get_node_scores(harmonic_flow, B):
    return B.T @ harmonic_flow
