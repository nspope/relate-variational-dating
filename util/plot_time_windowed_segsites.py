import tskit
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tree-sequence", type=str)
parser.add_argument("--true-tree-sequence", type=str)
parser.add_argument("--output-path", type=str)
parser.add_argument("--time-grid-size", type=int, default=100)
parser.add_argument("--time-grid-max", type=float, default=1e6)
parser.add_argument("--time-grid-min", type=float, default=1e2)
parser.add_argument("--mutation-rate", type=float, default=1.29e-8)
parser.add_argument("--max-sequence-length", type=float, default=None)
parser.add_argument("--title", type=str, default=None)


def time_windowed_segsites(ts, time_windows, mutation_rate):
    """ slow but transparent """
    nodes_window = np.digitize(ts.nodes_time, time_windows) # bins[i-1] <= x < bins[i]
    obs = np.zeros(time_windows.size + 1)
    exp = np.zeros(time_windows.size + 1)
    window_length = np.diff(time_windows)
    for t in ts.trees():
        mutational_target = mutation_rate * t.span
        for n in t.nodes():
            p = t.parent(n)
            if p != tskit.NULL:
                time_n = t.time(n)
                time_p = t.time(p)
                window_n = nodes_window[n]
                window_p = nodes_window[p]
                remainder_n = time_n - time_windows[window_n - 1]
                remainder_p = time_windows[window_p] - time_p
                exp[window_n-1:window_p] += window_length[window_n-1:window_p] * mutational_target
                exp[window_n-1] -= remainder_n * mutational_target
                exp[window_p-1] -= remainder_p * mutational_target
        for m in t.mutations():
            if m.edge != tskit.NULL:
                n = m.node
                p = t.parent(n)
                time_n = t.time(n)
                time_p = t.time(p)
                window_n = nodes_window[n]
                window_p = nodes_window[p]
                remainder_n = time_n - time_windows[window_n - 1]
                remainder_p = time_windows[window_p] - time_p
                density = 1 / (time_p - time_n)
                obs[window_n-1:window_p] += window_length[window_n-1:window_p] * density
                obs[window_n-1] -= remainder_n * density
                obs[window_p-1] -= remainder_p * density
    return obs, exp


if __name__ == "__main__":

    args = parser.parse_args()

    ts = tskit.load(args.tree_sequence)
    true_ts = tskit.load(args.true_tree_sequence)

    if args.max_sequence_length is not None:
        tab = ts.dump_tables()
        tab.edges.drop_metadata()
        ts = tab.tree_sequence()
        ts = ts.keep_intervals(
            [[0, min(ts.sequence_length, args.max_sequence_length)]]
        ).trim()
        true_ts = true_ts.keep_intervals(
            [[0, min(true_ts.sequence_length, args.max_sequence_length)]]
        ).trim()

    ts_max = ts.nodes_time.max()
    time_grid = np.logspace(np.log10(args.time_grid_min), np.log10(args.time_grid_max), args.time_grid_size + 1)
    time_grid = time_grid[time_grid < ts_max]
    time_grid[0] = 0
    time_grid[-1] = ts_max + 1

    tru = np.bincount(np.digitize(true_ts.mutations_time, time_grid), minlength=time_grid.size + 1)
    obs, exp = time_windowed_segsites(ts, time_grid, mutation_rate=args.mutation_rate)

    ymax = max(exp[1:-1].max(), obs[1:-1].max()) * 1.1
    ymin = min(exp[1:-1].min(), obs[1:-1].min()) * 0.9
    plt.figure(figsize=(5, 3.5), constrained_layout=True)
    plt.plot(time_grid[:-1], tru[1:-1], "-o", color="black", markersize=2, label="true-site")
    plt.plot(time_grid[:-1], exp[1:-1], "-o", color="red", markersize=2, alpha=0.5, label="infr-branch")
    plt.plot(time_grid[:-1], obs[1:-1], "-o", color="blue", markersize=2, alpha=0.5, label="infr-site")
    plt.legend()
    plt.ylim(ymin, ymax)
    plt.xscale("log")
    plt.ylabel("Segregating sites")
    plt.xlabel("Time ago")
    if args.title is not None: plt.title(args.title)
    plt.savefig(args.output_path)
