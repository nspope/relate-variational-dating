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
import time
import glob
import numpy as np
import argparse
import subprocess
import msprime
import logging
import demesdraw
from sys import stdout, stderr
import matplotlib.pyplot as plt

from lib.util import mutation_freq_and_midpoint_age, time_windowed_segregating_sites

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
    type=str, default="../relate_v1.2.2_x86_64_static",
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
parser.add_argument("--use-prior", action="store_true")
parser.add_argument("--use-relate-ages", action="store_true")
parser.add_argument("--rescaling-iterations", type=int, default=5)
parser.add_argument("--rescaling-intervals", type=int, default=1000)
parser.add_argument("--rate-intervals", type=int, default=25)


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

        # population labels
        sample_names = []
        labels_path = f"{input_prefix}.labels"
        with open(labels_path, "w") as handle:
            handle.write(f"sample\tpopulation\tgroup\tsex\n")
            for i in ts.individuals():
                p = i.population
                name = f"{p}{i.id}{'a' if i.time > 0 else 'c'}" 
                handle.write(f"{name}\t{p}\t{p}\tNA\n")
                sample_names.append(name)

        # vcf
        vcf_path = f"{input_prefix}.vcf"
        ts.write_vcf(
            open(vcf_path, "w"), 
            contig_id=chrom_name, 
            individual_names=sample_names,
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

        # sample ages
        # NB: these are per haplotype, not per diploid individual
        ages_path = f"{input_prefix}.ages"
        with open(ages_path, "w") as handle:
            for i in ts.individuals():
                for s in i.nodes:
                    age = ts.nodes_time[s]
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
            "--seed", "1024", #f"{relate_seed}",
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
    samples = np.array(list(ts.samples()))
    ancients = samples[ts.nodes_time[samples] > 0]
    contemporary = samples[ts.nodes_time[samples] == 0]
    assert np.allclose(relate_ts.nodes_time[ancients], ts.nodes_time[ancients])
    assert np.allclose(relate_ts.nodes_time[contemporary], 0.0)
    assert np.allclose(list(relate_ts.samples()), samples)

    ep_trees_path = f"{output_path}/ep.trees"
    if not os.path.exists(ep_trees_path) or args.overwrite or args.overwrite_from_ep:

        import tsdate

        EP = tsdate.variational.ExpectationPropagation(relate_ts, mutation_rate=args.mutation_rate)

        # initialize with coalescent prior
        # TODO: should reduce down to contemporary
        def conditional_coalescent(num_tips):
            coal_rates = np.array(
                [2 / (i * (i - 1)) if i > 1 else 0.0 for i in range(1, num_tips + 1)]
            )
            mean = coal_rates.copy()
            variance = coal_rates.copy() ** 2
            for i in range(coal_rates.size - 2, 0, -1):
                mean[i] += mean[i + 1]
                variance[i] += variance[i + 1]
            moments = tsdate.prior._marginalize_over_ancestors(np.stack((mean, variance + mean**2), 1))
            moments[:, 1] -= moments[:, 0] ** 2
            return moments

        if args.use_prior:
            Ne = relate_ts.diversity() / args.mutation_rate  # haploid Ne
            prior_moments = conditional_coalescent(relate_ts.num_samples)
            prior_moments[:, 0] *= Ne
            prior_moments[:, 1] *= Ne ** 2
            node_prior = np.zeros((relate_ts.num_nodes, 2))
            for t in relate_ts.trees():
                for n in t.nodes():
                    num_samples = t.num_samples(n)
                    #if num_samples == relate_ts.num_samples:  # roots only
                    if num_samples > 1:
                        # note this is natural not canonical parameterization of gamma
                        node_prior[n] = \
                            tsdate.approx.approximate_gamma_mom(*prior_moments[num_samples])
            EP.node_posterior[:] = node_prior[:]

        if args.use_relate_ages:
            # this should only ever be done if skipping EP entirely, so:
            args.ep_iterations = 0
            node_prior = np.zeros((relate_ts.num_nodes, 2))
            for n in range(relate_ts.num_samples, relate_ts.num_nodes):
                node_prior[n] = \
                    tsdate.approx.approximate_gamma_mom(relate_ts.nodes_time[n], 1)
                # variance is arbitrary
            EP.node_posterior[:] = node_prior[:]
        
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

        if args.ancients_ages_unknown:
            # FIXME: think more clearly about the right edge likelihoods to use.
            # if using the propagated counts, these will bias things because
            # the ancestral samples span multiple trees, and introduce a coupling.
            # one crude way to deal with this is to segment the ancestral samples
            # over trees. A better way would be use the propagated counts on upwards
            # traversal (without updating child), and unpropagated on downwards traversal
            # (without updating parent). This could be done with two separate calls
            # to EP.propagate_likelihood, with a "buffer" posterior for the ancients.
            EP.node_constraints[ancients, 0] = 0.0
            EP.node_constraints[ancients, 1] = np.inf
            assert False, "not implemented"

        ep_timing = time.time()
        for _ in range(args.ep_iterations):
            # TODO: use rescaling per iter here, to adjust mutation rate?
            # edge_adj = edge_mutation_rate_adjustment(node_times_from_ep, relate_ts)
            # EP.edge_likelihoods[:, 1] *= edge_adj
            EP.iterate(min_step=0.1, max_shape=1000, regularise=args.regularise_roots)
        ep_timing = time.time() - ep_timing
        logging.info(f"Node posteriors in {ep_timing:.2f} seconds")
        
        # rescale using mutational clock, using the tree-by-tree counts/spans
        EP.edge_likelihoods[:] = edge_likelihoods[:] #DEBUG
        EP.infer(
            ep_iterations=0, 
            max_shape=1000, 
            rescale_iterations=args.rescaling_iterations, 
            rescale_segsites=True, 
            regularise=False, 
            rescale_intervals=args.rescaling_intervals,
        )
        node_ages_mean, node_ages_var = EP.node_moments()
        constrained_ages = tsdate.util.constrain_ages(relate_ts, node_ages_mean)
        logging.info(
            f"Max constraint adjustment: "
            f"{np.max(constrained_ages - node_ages_mean)}"
        )

        # put new dates into a tree sequence
        tab = relate_ts.dump_tables()
        tab.edges.drop_metadata()
        tab.nodes.drop_metadata()
        tab.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        tab.nodes.time = constrained_ages  # node_ages_mean
        tab.nodes.packset_metadata([
            tab.nodes.metadata_schema.validate_and_encode_row({"mean": m, "var": v})
            for m, v in zip(node_ages_mean, node_ages_var)
        ])
        tab.mutations.time = tab.nodes.time[relate_ts.mutations_node]
        tab.sort()
        tab.build_index()
        tab.compute_mutation_times()
        tab.tree_sequence().dump(ep_trees_path)


    # --- comparison figures

    ep_ts = tskit.load(ep_trees_path)
    plot_path = f"{output_path}/plots"

    if not os.path.exists(plot_path): os.makedirs(plot_path)

    simulation_info = (
        f"seqlen: {args.sequence_length/1e6:.1f}Mb; "
        f"samples: {','.join([str(x) for x in args.num_contemporary])}; "
        f"ancients: {','.join([str(x) for x in args.num_ancient])}; "
        f"popsize: {','.join([str(int(x)) for x in args.population_sizes])}; "
    )

    # (1) demographic model
    demesdraw.tubes(demogr.to_demes())
    plt.savefig(f"{plot_path}/demography.png")
    plt.clf()

    # (2) mutation age estimates versus true ages
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

    # (3) pair coalescence rates (contemporary samples only)
    num_pops = len(args.population_sizes)
    samples = np.array(list(ts.samples()))
    sample_sets = [
        np.flatnonzero(
            np.logical_and(
                ts.nodes_population[samples] == i, 
                ts.nodes_time[samples] == 0.0,
            ) 
        )
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
        rows, cols, figsize=(cols * 4, rows * 3),
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
                        
    # (4) pair coalescence pdf (contemporary samples only)
    rows, cols = num_pops, num_pops
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 3),
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
                    time_grid[:-1], true_pdf[k], 
                    where="post", label="true", color="black",
                )
                axs[i, j].step(
                    time_grid[:-1], relate_pdf[k], alpha=0.5,
                    where="post", label="mcmc", color="red",
                )
                axs[i, j].step(
                    time_grid[:-1], ep_pdf[k], alpha=0.5,
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

    # (5) expected vs observed SFS (contemporary samples only)
    samples = np.array(list(relate_ts.samples()))
    subset = samples[relate_ts.nodes_time[samples] == 0.0]
    obs_sfs = relate_ts.allele_frequency_spectrum(
        sample_sets=[subset],
        mode='site',
        polarised=True,
    )
    relate_sfs = relate_ts.allele_frequency_spectrum(
        sample_sets=[subset],
        mode='branch',
        polarised=True,
    ) * args.mutation_rate
    ep_sfs = ep_ts.allele_frequency_spectrum(
        sample_sets=[subset],
        mode='branch',
        polarised=True,
    ) * args.mutation_rate

    rows, cols = 2, 2
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * 5, rows * 3),
        constrained_layout=True, sharex=True, sharey="row",
        squeeze=False,
    )
    
    axs[0, 0].set_title("MCMC")
    axs[0, 0].plot(
        np.arange(1, subset.size), obs_sfs[1:-1], "-o", 
        color="gray", label="observed", markersize=2,
    )
    axs[0, 0].plot(
        np.arange(1, subset.size), relate_sfs[1:-1], "-o", 
        color="red", label="expected", markersize=2, alpha=0.5,
    )
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_ylabel("# mutations", size=10)
    axs[0, 0].legend()

    axs[0, 1].set_title("EP")
    axs[0, 1].plot(
        np.arange(1, subset.size), obs_sfs[1:-1], "-o", 
        color="gray", label="observed", markersize=2,
    )
    axs[0, 1].plot(
        np.arange(1, subset.size), ep_sfs[1:-1], "-o", 
        color="blue", label="expected", markersize=2, alpha=0.5,
    )
    axs[0, 1].set_yscale("log")
    axs[0, 1].legend()

    resid = np.log10(relate_sfs[1:-1]) - np.log10(obs_sfs[1:-1])
    axs[1, 0].plot(
        np.arange(1, subset.size), resid, "-o", 
        color="red", label="mcmc", markersize=2,
    )
    axs[1, 0].axhline(y=0.0, linestyle="dashed", color="black")
    axs[1, 0].set_ylabel("residual (log10)", size=10)

    resid = np.log10(ep_sfs[1:-1]) - np.log10(obs_sfs[1:-1])
    axs[1, 1].plot(
        np.arange(1, subset.size), resid, "-o", 
        color="blue", label="ep", markersize=2,
    )
    axs[1, 1].axhline(y=0.0, linestyle="dashed", color="black")

    fig.supxlabel("Mutation frequency", size=10)
    fig.suptitle(simulation_info, size=10)
    plt.savefig(f"{plot_path}/site-vs-branch-sfs.png")
    plt.clf()

    # (6) segregating sites over time
    max_time = max(relate_ts.nodes_time.max(), ep_ts.nodes_time.max())
    time_windows = np.logspace(2, 6, 101)
    time_windows = time_windows[time_windows < max_time]
    time_windows[-1] = max_time + 1

    true_obs = np.bincount(
        np.digitize(ts.mutations_time, time_windows), 
        minlength=time_windows.size + 1,
    )[1:-1]
    relate_obs, relate_exp = time_windowed_segregating_sites(
        relate_ts, 
        time_windows, 
        mutation_rate=args.mutation_rate,
    )
    ep_obs, ep_exp = time_windowed_segregating_sites(
        ep_ts, 
        time_windows, 
        mutation_rate=args.mutation_rate,
    )

    rows, cols = 1, 2
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 3),
        constrained_layout=True, sharex=True, sharey=True,
        squeeze=False,
    )

    axs[0, 0].set_title("MCMC")
    axs[0, 0].plot(
        time_windows[:-1]/2 + time_windows[1:]/2, true_obs, 
        "-o", color="black", markersize=2, label="true",
    )
    axs[0, 0].plot(
        time_windows[:-1]/2 + time_windows[1:]/2, relate_exp, 
        "-o", color="red", markersize=2, alpha=0.5, label="branch",
    )
    axs[0, 0].plot(
        time_windows[:-1]/2 + time_windows[1:]/2, relate_obs, 
        "-o", color="blue", markersize=2, alpha=0.5, label="site",
    )
    axs[0, 0].set_xscale("log")

    axs[0, 1].set_title("EP")
    axs[0, 1].plot(
        time_windows[:-1]/2 + time_windows[1:]/2, true_obs, 
        "-o", color="black", markersize=2, label="true",
    )
    axs[0, 1].plot(
        time_windows[:-1]/2 + time_windows[1:]/2, ep_exp, 
        "-o", color="red", markersize=2, alpha=0.5, label="branch",
    )
    axs[0, 1].plot(
        time_windows[:-1]/2 + time_windows[1:]/2, ep_obs, 
        "-o", color="blue", markersize=2, alpha=0.5, label="site",
    )
    axs[0, 1].set_xscale("log")

    fig.supylabel("# segregating sites")
    fig.supxlabel("Time ago")
    fig.suptitle(simulation_info, size=10)
    plt.savefig(f"{plot_path}/segsites-over-time.png")
    plt.clf()
