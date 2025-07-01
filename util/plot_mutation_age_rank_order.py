import tskit
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tree-sequence-a", type=str)
parser.add_argument("--tree-sequence-b", type=str)
parser.add_argument("--output-path", type=str)
parser.add_argument("--min-age", type=float, default=None)
parser.add_argument("--title", type=str, default=None)


def mutation_parent_age(ts, positions):
    position_map = {p: i for i, p in enumerate(positions)}
    mutation_age = np.full(positions.size, np.nan)
    mutation_frq = np.full(positions.size, np.nan)
    sites_mutations = np.bincount(ts.mutations_site, minlength=ts.num_sites)
    sites_mutations = {p: n for p, n in zip(ts.sites_position, sites_mutations)}
    for t in ts.trees():
        for m in t.mutations():
            if m.edge != tskit.NULL:
                p = ts.sites_position[m.site]
                if p in position_map and sites_mutations[p] == 1 and t.parent(m.node) != tskit.NULL:
                    i = position_map[p]
                    mutation_age[i] = t.time(t.parent(m.node))
                    mutation_frq[i] = t.num_samples(m.node)
    return mutation_frq, mutation_age


if __name__ == "__main__":

    args = parser.parse_args()

    ts_a = tskit.load(args.tree_sequence_a)
    ts_b = tskit.load(args.tree_sequence_b)

    _, time_a = mutation_parent_age(ts_a, ts_b.sites_position)
    _, time_b = mutation_parent_age(ts_b, ts_b.sites_position)
    ok = np.logical_and(np.isfinite(time_a), np.isfinite(time_b))
    time_a = time_a[ok]
    time_b = time_b[ok]
    rank_a = scipy.stats.rankdata(time_a)
    rank_a /= np.max(rank_a)
    rank_b = scipy.stats.rankdata(time_b)
    rank_b /= np.max(rank_b)
    if args.min_age is not None:  # filter w/ ages from first ts
        rank_a = rank_a[time_a >= args.min_age]
        rank_b = rank_b[time_a >= args.min_age]
    r2 = np.corrcoef(rank_a, rank_b)[0, 1] ** 2
    plt.figure(figsize=(5, 5), constrained_layout=True)
    #img = plt.scatter(rank_a, rank_b, c=time, s=2, cmap="plasma")
    plt.hexbin(rank_a, rank_b, mincnt=1)
    plt.title(f"r2: {r2:.3f}")
    plt.axline((rank_a.mean(), rank_a.mean()), slope=1, linestyle="dashed", color="red")
    #plt.colorbar(img)
    plt.ylabel("Rank b")
    plt.xlabel("Rank a")
    if args.title is not None: plt.title(args.title)
    plt.savefig(args.output_path)
