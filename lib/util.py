import numpy as np
import tskit
import scipy.sparse


def mutation_freq_and_midpoint_age(ts, positions):
    position_map = {p: i for i, p in enumerate(positions)}
    mutation_age = np.full(positions.size, np.nan)
    mutation_frq = np.full(positions.size, np.nan)
    sites_mutations = np.bincount(ts.mutations_site, minlength=ts.num_sites)
    sites_mutations = {p: n for p, n in zip(ts.sites_position, sites_mutations)}
    for t in ts.trees():
        for m in t.mutations():
            if m.edge != tskit.NULL:
                p = ts.sites_position[m.site]
                if p in position_map and sites_mutations[p] == 1:
                    i = position_map[p]
                    mutation_age[i] = (t.time(t.parent(m.node)) + t.time(m.node)) / 2
                    mutation_frq[i] = t.num_samples(m.node)
    return mutation_frq, mutation_age


def edge_to_window_overlap(ts, time_windows):
    """
    Sparse matrix of overlapping branch length between edges and time windows.
    The first and last rows of the matrix represent intervals from (-inf, time_windows[0])
    and (time_windows[-1], inf) respectively
    """
    time_windows[0] = 0.0  # FIXME: why is this needed? to index the correct breakpoint?
    nodes_window = np.digitize(ts.nodes_time, time_windows)
    assert np.all(nodes_window > 0)
    assert np.all(nodes_window < time_windows.size)
    child_time = ts.nodes_time[ts.edges_child]
    parent_time = ts.nodes_time[ts.edges_parent]
    child_window = nodes_window[ts.edges_child]
    parent_window = nodes_window[ts.edges_parent]
    child_remainder = child_time - time_windows[child_window - 1]
    parent_remainder = time_windows[parent_window] - parent_time
    window_size = np.diff(time_windows)
    
    edge = []
    window = []
    overlap = []
    for i in range(ts.num_edges):
        for j in range(child_window[i] - 1, parent_window[i]):
            edge.append(i)
            window.append(j)
            overlap.append(window_size[j])
        # entries are summed so insert correction as:
        edge.append(i)
        window.append(child_window[i] - 1)
        overlap.append(-child_remainder[i])
        edge.append(i)
        window.append(parent_window[i] - 1)
        overlap.append(-parent_remainder[i])
    
    edge_to_window = \
        scipy.sparse.coo_matrix(
            (overlap, (window, edge)), 
            shape=(time_windows.size + 1, ts.num_edges),
        ).tocsc()

    return edge_to_window


def time_windowed_segregating_sites(ts, time_windows, mutation_rate):
    edge_to_window = edge_to_window_overlap(ts, time_windows)
    edge_length = ts.nodes_time[ts.edges_parent] - ts.nodes_time[ts.edges_child]
    edges_mutations = np.zeros(ts.num_edges)
    for m in ts.mutations():
        if m.edge != tskit.NULL:
            edges_mutations[m.edge] += 1
    edge_span = (ts.edges_right - ts.edges_left) * mutation_rate
    edge_muts = edges_mutations / edge_length
    expected = edge_to_window @ edge_span
    observed = edge_to_window @ edge_muts
    assert expected.size == time_windows.size + 1
    assert observed.size == time_windows.size + 1
    return observed[1:-1], expected[1:-1]


def edge_mutation_rate_adjustment(ts, time_windows, mutation_rate):
    """
    Find adjusted mutation rates such that expected segregating sites
    (approximately) matches observed segregating sites through time
    """
    edge_to_window = edge_to_window_overlap(ts, time_windows)
    edge_length = ts.nodes_time[ts.edges_parent] - ts.nodes_time[ts.edges_child]
    edges_mutations = np.zeros(ts.num_edges)
    for m in ts.mutations():
        if m.edge != tskit.NULL:
            edges_mutations[m.edge] += 1
    edge_span = (ts.edges_right - ts.edges_left) * mutation_rate
    edge_muts = edges_mutations / edge_length
    expected = edge_to_window @ edge_span
    observed = edge_to_window @ edge_muts

    # use geometric mean, so that if the time window corrections are
    # c_i and the overlap between time window i and branch j is o_ij, the
    # correction for edge j is \prod_i c_i^{o_ij / \sum_k o_kj}
    window_to_edge = scipy.sparse.diags(1 / edge_length) @ edge_to_window.T
    correction = np.append(np.append(0, np.log(observed[1:-1]/expected[1:-1])), 0)
    correction[~np.isfinite(correction)] = 0
    edge_correction = np.exp(window_to_edge @ correction)

    return edge_correction
