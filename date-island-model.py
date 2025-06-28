docstring = \
"""
Prototype for variational dating in Relate
1. Multi-population island model with unequal sizes
2. Ancient samples scattered from 0-1000 generations ago
3. Run Relate mode 'All' without iterative pop size estimation
4. Use tsdate with Relate's propagated mutations/spans
5. Plot estimated vs true mutation ages, pair coalescence rates/pdf
"""

import os
import tskit
import glob
import numpy as np
import argparse
import subprocess
import msprime
import logging
import demesdraw
from sys import stdout, stderr
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(docstring)
parser.add_argument(
    "--output-dir", 
    type=str, required=True,
    help="Path to output directory (will be created)",
)
parser.add_argument(
    "--num-contemporary", 
    type=int, nargs="+", default=[250, 250],
    help="Number of contemporary diploid samples, one value per population",
)
parser.add_argument(
    "--num-ancient", 
    type=int, nargs="+", default=[0, 0],
    help="Number of ancient diploid samples, one value per population",
)
parser.add_argument(
    "--relate-lib-dir",
    type=str, default="../relate_lib",
    help="Path to relate_lib repo",
)
parser.add_argument(
    "--relate-dir", 
    type=str, default="../relate",
    help="Path to relate repo",
)
parser.add_argument(
    "--population-sizes", 
    type=float, nargs="+", default=[2e4, 1e3],
    help="Population sizes to use for island model; number of populations "
    "must match number of values given for --num-contemporary and --num-ancient",
)
parser.add_argument(
    "--migration-rate",
    type=float, default=5e-5,
    help="Migration rate between populations in island model",
)
parser.add_argument(
    "--ancients-max-time", 
    type=float, default=1000,
    help="Upper bound for age of ancient samples"
)
parser.add_argument(
    "--ancients-ages-unknown", 
    action="store_true",
    help="Treat ages of ancient samples as unknown",
)
parser.add_argument("--mutation-rate", type=float, default=1.29e-8)
parser.add_argument("--recombination-rate", type=float, default=1.0e-8)
parser.add_argument("--sequence-length", type=float, default=2.0e7)
parser.add_argument("--random-seed", type=int, default=1024)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--overwrite-from-ep", action="store_true")
parser.add_argument("--ep-iterations", type=int, default=5)
parser.add_argument("--regularise-roots", action="store_true")
parser.add_argument("--rescaling-iterations", type=int, default=5)
parser.add_argument("--rescaling-intervals", type=int, default=1000)
parser.add_argument("--rate-intervals", type=int, default=25)


def mutation_freq_and_midpoint_age(ts, positions):
    position_map = {p: i for i, p in enumerate(positions)}
    mutation_age = np.full(positions.size, np.nan)
    mutation_frq = np.full(positions.size, np.nan)
    sites_mutations = np.bincount(ts.mutations_site, minlength=ts.num_sites)
    sites_mutations = {p: n for p, n in zip(ts.sites_position, sites_mutations)}
    for t in ts.trees():
        for m in t.mutations():
            if m.edge != -1:
                p = ts.sites_position[m.site]
                if p in position_map and sites_mutations[p] == 1:
                    i = position_map[p]
                    mutation_age[i] = (t.time(t.parent(m.node)) + t.time(m.node)) / 2
                    mutation_frq[i] = t.num_samples(m.node)
    return mutation_frq, mutation_age


if __name__ == "__main__":

    args = parser.parse_args()
    rng = np.random.default_rng(args.random_seed)
    msprime_seed = rng.integers(2 ** 32 - 1, size=2)
    relate_seed = rng.integers(2 ** 32 - 1)
    relate_path = os.path.abspath(args.relate_dir)
    relate_lib_path = os.path.abspath(args.relate_lib_dir)
    output_path = os.path.abspath(args.output_dir)
    if not os.path.exists(output_path): os.makedirs(output_path)


    # --- simulate data ---

    # demographic model
    assert len(args.num_contemporary) == len(args.num_ancient)
    sample_ages = [
        np.append(np.zeros(n_c), rng.uniform(0, args.ancients_max_time, size=n_a))
        for n_c, n_a in zip(args.num_contemporary, args.num_ancient)
    ]
    assert len(args.population_sizes) == len(sample_ages)
    demogr = msprime.Demography.island_model(
        args.population_sizes, 
        migration_rate=args.migration_rate,
    )
    sample_config = []
    for i, ages in enumerate(sample_ages):
        sample_config.extend([msprime.SampleSet(1, time=t, population=i) for t in ages])

    # msprime simulation
    trees_path = f"{output_path}/true.trees"
    if not os.path.exists(trees_path) or args.overwrite:
        ts = msprime.sim_ancestry(
            sample_config,
            demography=demogr,
            recombination_rate=args.recombination_rate,
            sequence_length=args.sequence_length,
            random_seed=msprime_seed[0],
        )
        ts = msprime.sim_mutations(
            ts, 
            rate=args.mutation_rate, 
            random_seed=msprime_seed[1],
            model=msprime.MatrixMutationModel(
                alleles=["A", "G"], 
                root_distribution=[1., 0.], 
                transition_matrix=[[0., 1.], [1., 0.]],
            ),
        )
        ts.dump(trees_path)
    else:
        ts = tskit.load(trees_path)


    # --- run Relate --- 

    chrom_name = "chr1"
    relate_prefix = f"{output_path}/relate"
    input_prefix = f"{relate_prefix}/input/{chrom_name}"
    output_prefix = f"{relate_prefix}/output/{chrom_name}"
    if not os.path.exists(f"{relate_prefix}.trees") or args.overwrite:

        if not os.path.exists(os.path.dirname(input_prefix)):
            os.makedirs(os.path.dirname(input_prefix))

        if not os.path.exists(os.path.dirname(output_prefix)):
            os.makedirs(os.path.dirname(output_prefix))

        for f in glob.glob(f"{input_prefix}*"): os.remove(f)
        for f in glob.glob(f"{output_prefix}*"): os.remove(f)

        sample_names = []
        for p, ages in enumerate(sample_ages):
            sample_names.append([
                f"{p}{i}{'a' if t > 0 else 'c'}" 
                for i, t in enumerate(ages)
            ])

        # vcf
        vcf_path = f"{input_prefix}.vcf"
        ts.write_vcf(
            open(vcf_path, "w"), 
            contig_id=chrom_name, 
            individual_names=np.concatenate(sample_names),
        )

        # recombination map
        hapmap_path = f"{input_prefix}.hapmap"
        with open(hapmap_path, "w") as handle:
            r = args.recombination_rate * 1e8
            l = int(args.sequence_length)
            handle.write("Position(bp) Rate(cM/Mb) Map(cM)\n")
            handle.write(f"0 {r:.6f} 0.0\n")
            handle.write(f"{l} 0.0 {l * r / 1e6:.6f}\n")

        # ancestral fasta
        ancestral_path = f"{input_prefix}.anc.fa"
        with open(ancestral_path, "w") as handle:
            handle.write(f">{chrom_name}\n")
            handle.write("A" * int(args.sequence_length) + "\n")

        # missing data mask
        mask_path = f"{input_prefix}.mask.fa"
        with open(mask_path, "w") as handle:
            handle.write(f">{chrom_name}\n")
            handle.write("P" * int(args.sequence_length) + "\n")

        # population labels
        labels_path = f"{input_prefix}.labels"
        with open(labels_path, "w") as handle:
            handle.write(f"sample\tpopulation\tgroup\tsex\n")
            for p, names in enumerate(sample_names):
                for name in names:
                    handle.write(f"{name}\t{p}\t{p}\tNA\n")

        # sample ages
        ages_path = f"{input_prefix}.ages"
        with open(ages_path, "w") as handle:
            for age in np.concatenate(sample_ages):
                handle.write(f"{age:.4f}\n")

        # prepare inputs
        convert_from_vcf = [
            f"{relate_path}/bin/RelateFileFormats",
            "--mode", "ConvertFromVcf",
            "--haps", f"{input_prefix}.haps",
            "--sample", f"{input_prefix}.sample",
            "-i", input_prefix,
        ]
        job = subprocess.run(convert_from_vcf, capture_output=True)
        stdout.write(job.stdout.decode('utf-8'))
        stderr.write(job.stderr.decode('utf-8'))
        assert job.returncode == 0

        prepare_input_files = [
            f"{relate_path}/scripts/PrepareInputFiles/PrepareInputFiles.sh",
            "--haps", f"{input_prefix}.haps",
            "--sample", f"{input_prefix}.sample", 
            "--poplabels", f"{input_prefix}.labels",
            "--ancestor", f"{input_prefix}.anc.fa",
            "--mask", f"{input_prefix}.mask.fa",
            "-o", f"{output_prefix}"
        ]
        job = subprocess.run(prepare_input_files, capture_output=True)
        stdout.write(job.stdout.decode('utf-8'))
        stderr.write(job.stderr.decode('utf-8'))
        assert job.returncode == 0

        # run Relate
        samples = np.array(list(ts.samples()))
        contemporary = samples[ts.nodes_time[samples] == 0]
        Ne = ts.simplify(samples=contemporary).diversity(mode="site") / args.mutation_rate  # haploid
        relate = [
            f"{relate_path}/bin/Relate",
            "--mode", "All",
            "--haps", f"{output_prefix}.haps.gz",
            "--sample", f"{output_prefix}.sample.gz", 
            "--annot", f"{output_prefix}.annot",
            "--dist", f"{output_prefix}.dist.gz",
            "--map", f"{input_prefix}.hapmap",
            "--sample_ages", f"{input_prefix}.ages",
            "-m", f"{args.mutation_rate}", 
            "-N", f"{Ne}", 
            "-o", f"{chrom_name}",
        ]
        job = subprocess.run(relate, cwd=os.path.dirname(output_prefix), capture_output=True)
        stdout.write(job.stdout.decode('utf-8'))
        stderr.write(job.stderr.decode('utf-8'))
        assert job.returncode == 0

        # convert to tree sequence
        convert = [
            f"{relate_lib_path}/bin/Convert",
            "--mode", "ConvertToTreeSequence",
            "--anc", f"{output_prefix}.anc",
            "--mut", f"{output_prefix}.mut",
            "-o", f"{relate_prefix}",
        ]
        job = subprocess.run(convert, capture_output=True)
        stdout.write(job.stdout.decode('utf-8'))
        stderr.write(job.stderr.decode('utf-8'))
        assert job.returncode == 0


    # --- variational dating ---

    relate_ts = tskit.load(f"{relate_prefix}.trees")
    ep_trees_path = f"{output_path}/ep.trees"
    if not os.path.exists(ep_trees_path) or args.overwrite or args.overwrite_from_ep:

        import tsdate
        EP = tsdate.variational.ExpectationPropagation(relate_ts, mutation_rate=args.mutation_rate)
        
        # calculate "extended" SNP counts/edge spans from relate
        edge_meta = np.stack(
            [e.metadata.decode('utf-8').strip().split() for e in relate_ts.edges()]
        ).astype(int)
        propagated_edge_span = (edge_meta[:, 1] - edge_meta[:, 0]).astype(float)
        propagated_edge_muts = edge_meta[:, 2].astype(float)
        # NB: this is convoluted because the start position/end position/mutation count
        #     are integers converted to ASCII stored as binary in edge metadata
        
        # swap out per-tree counts/spans with propagated ones
        edge_likelihoods = EP.edge_likelihoods.copy()
        EP.edge_likelihoods[:, 0] = propagated_edge_muts
        EP.edge_likelihoods[:, 1] = propagated_edge_span * args.mutation_rate

        # run expectation propagation
        for _ in range(args.ep_iterations):
            EP.iterate(min_step=0.1, max_shape=1000, regularise=args.regularise_roots)
        
        # rescale using mutational clock, using the tree-by-tree counts/spans
        EP.edge_likelihoods[:] = edge_likelihoods[:]
        EP.infer(
            ep_iterations=0, 
            max_shape=1000, 
            rescale_iterations=args.rescaling_iterations, 
            rescale_segsites=True, 
            regularise=False, 
            rescale_intervals=args.rescaling_intervals,
        )
        node_ages_mean, node_ages_var = EP.node_moments()

        # put new dates into a tree sequence
        # TODO: this might need constraint with ancient samples
        tab = relate_ts.dump_tables()
        tab.edges.drop_metadata()
        tab.nodes.drop_metadata()
        tab.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        tab.nodes.time = node_ages_mean
        tab.nodes.packset_metadata([
            tab.nodes.metadata_schema.validate_and_encode_row({"mean": m, "var": v})
            for m, v in zip(node_ages_mean, node_ages_var)
        ])
        tab.mutations.time = node_ages_mean[relate_ts.mutations_node]
        tab.sort()
        tab.build_index()
        tab.compute_mutation_times()
        tab.tree_sequence().dump(ep_trees_path)


    # --- comparison figures

    ep_ts = tskit.load(ep_trees_path)
    plot_path = f"{output_path}/plots"
    if not os.path.exists(plot_path) or args.overwrite or args.overwrite_from_ep:

        if not os.path.exists(plot_path): os.makedirs(plot_path)

        simulation_info = (
            f"seqlen: {args.sequence_length/1e6:.1f}Mb; "
            f"samples: {','.join([str(x) for x in args.num_contemporary])}; "
            f"ancients: {','.join([str(x) for x in args.num_ancient])}; "
            f"popsize: {','.join([str(int(x)) for x in args.population_sizes])}; "
        )

        # demographic model
        demesdraw.tubes(demogr.to_demes())
        plt.savefig(f"{plot_path}/demography.png")
        plt.clf()

        # mutation age estimates versus true ages
        true_mutation_freq, true_mutation_ages = \
            mutation_freq_and_midpoint_age(ts, relate_ts.sites_position)
        relate_mutation_freq, relate_mutation_ages = \
            mutation_freq_and_midpoint_age(relate_ts, relate_ts.sites_position)
        ep_mutation_freq, ep_mutation_ages = \
            mutation_freq_and_midpoint_age(ep_ts, relate_ts.sites_position)

        rows, cols = 1, 2
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * 3, rows * 3),
            constrained_layout=True, sharex=True, sharey=True,
            squeeze=False,
        )
        mx = np.nanmean(true_mutation_ages)

        axs[0, 0].set_title("MCMC", size=10)
        axs[0, 0].hexbin(
            true_mutation_ages, relate_mutation_ages, 
            xscale="log", yscale="log", mincnt=1,
        )
        axs[0, 0].axline(
            (mx, mx), (mx + 1, mx + 1), 
            linestyle="dashed", color="red",
        )
        mse = np.nanmean((np.log10(true_mutation_ages) - np.log10(relate_mutation_ages)) ** 2)
        bias = np.nanmean(-(np.log10(true_mutation_ages) - np.log10(relate_mutation_ages)))
        axs[0, 0].text(
            0.01, 0.99, f"mse: {mse:.3f}\nbias: {bias:.3f}",
            transform=axs[0, 0].transAxes,
            size=10, ha="left", va="top",
        )

        axs[0, 1].set_title("EP", size=10)
        axs[0, 1].hexbin(
            true_mutation_ages, ep_mutation_ages, 
            xscale="log", yscale="log", mincnt=1,
        )
        axs[0, 1].axline(
            (mx, mx), (mx + 1, mx + 1), 
            linestyle="dashed", color="red",
        )
        mse = np.nanmean((np.log10(true_mutation_ages) - np.log10(ep_mutation_ages)) ** 2)
        bias = np.nanmean(-(np.log10(true_mutation_ages) - np.log10(ep_mutation_ages)))
        axs[0, 1].text(
            0.01, 0.99, f"mse: {mse:.3f}\nbias: {bias:.3f}",
            transform=axs[0, 1].transAxes,
            size=10, ha="left", va="top",
        )

        fig.suptitle(simulation_info, size=10)
        fig.supylabel("Estimated mutation age", size=10)
        fig.supxlabel("True mutation age", size=10)
        plt.savefig(f"{plot_path}/mutation-ages.png")
        plt.clf()

        # pair coalescence rates
        num_pops = len(args.population_sizes)
        samples = np.array(list(ts.samples()))
        sample_sets = [
            np.flatnonzero(ts.nodes_population[samples] == i) 
            for i in range(num_pops)
        ]
        indexes = [(i, j) for i in range(num_pops) for j in range(i, num_pops)]
        time_grid = np.logspace(2, 6, args.rate_intervals + 1)
        time_grid[0] = 0.0
        time_grid[-1] = np.inf
        true_pdf = ts.pair_coalescence_counts(
            sample_sets=sample_sets,
            indexes=indexes,
            time_windows=time_grid, 
            pair_normalise=True,
        )
        relate_pdf = relate_ts.pair_coalescence_counts(
            sample_sets=sample_sets,
            indexes=indexes,
            time_windows=time_grid, 
            pair_normalise=True,
        )
        ep_pdf = ep_ts.pair_coalescence_counts(
            sample_sets=sample_sets,
            indexes=indexes,
            time_windows=time_grid, 
            pair_normalise=True,
        )
        true_rates = ts.pair_coalescence_rates(
            time_windows=time_grid,
            sample_sets=sample_sets,
            indexes=indexes,
        ) 
        relate_rates = relate_ts.pair_coalescence_rates(
            time_windows=time_grid, 
            sample_sets=sample_sets,
            indexes=indexes,
        )
        ep_rates = ep_ts.pair_coalescence_rates(
            time_windows=time_grid, 
            sample_sets=sample_sets,
            indexes=indexes,
        )
        cutoff = np.argmax(np.cumsum(true_pdf) > 0.99) + 1

        rows, cols = num_pops, num_pops
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * 3.5, rows * 3),
            constrained_layout=True, sharex=True, sharey=False,
            squeeze=False,
        )
        k = 0
        for i in range(num_pops):
            for j in range(num_pops):
                if i > j:
                    axs[i, j].set_visible(False)
                else:
                    axs[i, j].step(
                        time_grid[:cutoff], true_rates[k][:cutoff], 
                        where="post", label="true", color="black",
                    )
                    axs[i, j].step(
                        time_grid[:cutoff], relate_rates[k][:cutoff], alpha=0.5,
                        where="post", label="mcmc", color="red",
                    )
                    axs[i, j].step(
                        time_grid[:cutoff], ep_rates[k][:cutoff], alpha=0.5,
                        where="post", label="ep", color="blue",
                    )
                    axs[i, j].set_title(f"pop{i} v. pop{j}", size=10)
                    axs[i, j].legend()
                    axs[i, j].set_xscale("log")
                    axs[i, j].set_yscale("log")
                    k += 1
        fig.supxlabel("Time ago", size=10)
        fig.supylabel("Pair coalescence rate", size=10)
        fig.suptitle(simulation_info, size=10)
        plt.savefig(f"{plot_path}/pair-rates.png")
        plt.clf()
                            
        # pair coalescence pdf
        rows, cols = num_pops, num_pops
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * 3.5, rows * 3),
            constrained_layout=True, sharex=True, sharey=False,
            squeeze=False,
        )
        k = 0
        for i in range(num_pops):
            for j in range(num_pops):
                if i > j:
                    axs[i, j].set_visible(False)
                else:
                    axs[i, j].step(
                        time_grid[:cutoff], true_pdf[k][:cutoff], 
                        where="post", label="true", color="black",
                    )
                    axs[i, j].step(
                        time_grid[:cutoff], relate_pdf[k][:cutoff], alpha=0.5,
                        where="post", label="mcmc", color="red",
                    )
                    axs[i, j].step(
                        time_grid[:cutoff], ep_pdf[k][:cutoff], alpha=0.5,
                        where="post", label="ep", color="blue",
                    )
                    axs[i, j].set_title(f"pop{i} v. pop{j}", size=10)
                    axs[i, j].legend()
                    axs[i, j].set_xscale("log")
                    k += 1
        fig.supxlabel("Time ago", size=10)
        fig.supylabel("Proportion coalescing pairs", size=10)
        fig.suptitle(simulation_info, size=10)
        plt.savefig(f"{plot_path}/pair-pdf.png")
        plt.clf()

