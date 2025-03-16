import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



NAME_MAPPING = {
    'ds1000::case296':           'DS1000····#296',
    'ds1000::case309':           'DS1000····#309',
    'ds1000::case330':           'DS1000····#330',
    'ds1000::case360':           'DS1000····#360',
    'ds1000::case387':           'DS1000····#387',
    'ds1000::case418':           'DS1000····#418',
    'ds1000::case453':           'DS1000····#453',
    'ds1000::case459':           'DS1000····#459',
    'ds1000::case501':           'DS1000····#501',
    'ds1000::case510':           'DS1000····#510',
    'mlalgo::neuron':            'MLAlgo··Neuron',
    'mlalgo::kmeans':            'MLAlgo··KMeans',
    'mlalgo::linear_regression': 'MLAlgo··LinReg',
    'leetcode_array::p204':      'LC-Array··#204',
    'leetcode_array::p832':      'LC-Array··#832',
    'leetcode_dp::p740':         'LC-DP·····#740',
    'leetcode_dp::p1137':        'LC-DP····#1137',
    'leetcode_graph::p3112':     'LC-Graph·#3112',
    'leetcode_graph::p997':      'LC-Graph··#997',
    'leetcode_math::p492':       'LC-Math···#492',
    'leetcode_math::p2125':      'LC-Math··#2125',
    'leetcode_matrix::p73':      'LC-Matrix··#73',
    'leetcode_matrix::p2133':    'LC-Matrix#2133',
}


class AnyObject:
    def __init__(self, color: str):
        self.color = color


class AnyObjectHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle([x0, y0], width, height, facecolor=orig_handle.color,
                                   lw=3, transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch


def plot_evaluation_results():
    with open('results.json', 'r') as f:
        results_dict = json.load(f)

    names = []
    acc_rates = []
    prove_time_rates = []
    proving_time_baselines = []
    proving_time_optimizes = []
    verifying_time_baselines = []
    verifying_time_optimizes = []
    verify_time_rates = []
    snark_size_rates = []
    zinnia_compile_times = []
    for key, value in results_dict.items():
        names.append(NAME_MAPPING[key])
        zinnia_gates = value['zinnia']['advice_cells']
        halo2_gates = value['halo2']['advice_cells']
        zinnia_prove_time = value['zinnia']['proving_time']
        halo2_prove_time = value['halo2']['proving_time']
        zinnia_verify_time = value['zinnia']['verify_time']
        halo2_verify_time = value['halo2']['verify_time']
        zinnia_snark_size = value['zinnia']['snark_size']
        halo2_snark_size = value['halo2']['snark_size']
        proving_time_baselines.append(halo2_prove_time)
        proving_time_optimizes.append(zinnia_prove_time)
        verifying_time_baselines.append(halo2_verify_time)
        verifying_time_optimizes.append(zinnia_verify_time)
        acc_rates.append(-(zinnia_gates - halo2_gates) / halo2_gates * 100)
        prove_time_rates.append(-(zinnia_prove_time - halo2_prove_time) / halo2_prove_time * 100)
        verify_time_rates.append(-(zinnia_verify_time - halo2_verify_time) / halo2_verify_time * 100)
        snark_size_rates.append(-(zinnia_snark_size - halo2_snark_size) / halo2_snark_size * 100)
        zinnia_compile_times.append(value['zinnia_compile_time'])

    acc_rates = np.asarray(acc_rates)
    verifying_time_baselines = np.asarray(verifying_time_baselines)
    verifying_time_optimizes = np.asarray(verifying_time_optimizes)
    proving_time_baselines = np.asarray(proving_time_baselines)
    proving_time_optimizes = np.asarray(proving_time_optimizes)

    plt.rc('font', family='monospace', )
    # plt.rc('text', usetex=True)
    title_font = {'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 12}

    # Plot the comparison of gate reductions
    print(max(acc_rates), sum(acc_rates) / len(acc_rates))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(names, 100 - acc_rates, color='silver')
    ax.bar(names, acc_rates, color='lightgreen', bottom=100 - acc_rates, label='Optimized\nGates (%)')
    ax.tick_params(axis='x', labelrotation=90)
    ylabel = ax.set_ylabel('No. of Gates (%)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1]))
    ax.set_ylim(0, 117)
    ax.axhline(100, color='black', linewidth=1, linestyle='--')
    ax.text(len(names) - 1.5, 105, 'Baseline', fontsize=8, color='black', ha='center')
    fig.legend([AnyObject('silver'), AnyObject('lightgreen')],
               ['Zinnia Gates (%)', 'Optimized Gates (%)'],
               handler_map={
                   AnyObject: AnyObjectHandler()
               },
               loc='upper center', ncol=2,
               prop={'size': 8},
               frameon=False)
    fig.tight_layout(rect=[0, 0, 1.0, 0.97])
    plt.show()
    fig.savefig('gate-reductions.pdf', dpi=300)

    # Plot the comparison of proving time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    width = 0.4
    x = np.arange(len(names))
    ax1.bar(x - width / 2, proving_time_optimizes, width, color='mediumseagreen', label='Zinnia Proving Time')
    ax1.bar(x + width / 2, proving_time_baselines, width, color='cornflowerblue', label='Baseline Proving Time')
    ax1.set_xticks(x, names)
    ax1.tick_params(axis='x', labelrotation=90)
    ax1.set_yscale('log')
    ylabel = ax1.set_ylabel('Proving Time (s)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1]))
    ax2.bar(x - width / 2, verifying_time_optimizes * 1000, width, color='mediumseagreen', label='Zinnia Verifying Time')
    ax2.bar(x + width / 2, verifying_time_baselines * 1000, width, color='cornflowerblue', label='Baseline Verifying Time')
    ax2.set_xticks(x, names)
    ax2.tick_params(axis='x', labelrotation=90)
    ax2.set_yscale('log')
    ylabel = ax2.set_ylabel('Verifying Time (ms)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1]))
    fig.legend([AnyObject('mediumseagreen'), AnyObject('cornflowerblue')],
               ['Zinnia', 'Baseline'],
               handler_map={
                   AnyObject: AnyObjectHandler()
               },
               loc=(0.18, 0.92), ncol=2,
               prop={'size': 8},
               frameon=False)
    fig.legend([AnyObject('mediumseagreen'), AnyObject('cornflowerblue')],
               ['Zinnia', 'Baseline'],
               handler_map={
                   AnyObject: AnyObjectHandler()
               },
               loc=(0.68, 0.92), ncol=2,
               prop={'size': 8},
               frameon=False)
    fig.tight_layout(rect=[0, 0, 1.0, 0.97])
    plt.show()
    fig.savefig('proving-verifying-time.pdf', dpi=300)


def plot_performance_overviews():
    with open('results.json', 'r') as f:
        zinnia_results_dict = json.load(f)
    with open('results-sp1.json', 'r') as f:
        sp1_results_dict = json.load(f)
    with open('results-risc0.json', 'r') as f:
        risc0_results_dict = json.load(f)

    plt.rc('font', family='monospace', )
    # plt.rc('text', usetex=True)
    title_font = {'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 12}

    names = []
    zinnia_snark_proving_time = []
    baseline_snark_proving_time = []
    zinnia_verify_time = []
    baseline_verify_time = []
    zinnia_snark_size = []
    sp1_stark_proving_time = []
    sp1_snark_proving_time = []
    sp1_stark_verify_time = []
    sp1_snark_verify_time = []
    sp1_snark_size = []
    risc0_stark_proving_time = []
    risc0_stark_verify_time = []
    for key, value in sp1_results_dict.items():
        _zinnia_prove_time = zinnia_results_dict[key]['zinnia']['proving_time']
        _zinnia_verify_time = zinnia_results_dict[key]['zinnia']['verify_time']
        _zinnia_snark_size = zinnia_results_dict[key]['zinnia']['snark_size']
        _baseline_prove_time = zinnia_results_dict[key]['halo2']['proving_time']
        _baseline_verify_time = zinnia_results_dict[key]['halo2']['verify_time']

        names.append(NAME_MAPPING[key])
        zinnia_snark_proving_time.append(_zinnia_prove_time)
        baseline_snark_proving_time.append(_baseline_prove_time)
        zinnia_snark_size.append(_zinnia_snark_size)
        zinnia_verify_time.append(_zinnia_verify_time)
        baseline_verify_time.append(_baseline_verify_time)
        sp1_stark_proving_time.append(value['stark_proving_time'])
        sp1_snark_proving_time.append(value['snark_proving_time'])
        sp1_snark_size.append(value['snark_size'])
        sp1_stark_verify_time.append(value['stark_verify_time'])
        sp1_snark_verify_time.append(value['snark_verify_time'])

    for i, (key, value) in enumerate(risc0_results_dict.items()):
        assert NAME_MAPPING[key] == names[i]
        risc0_stark_proving_time.append(value['stark_proving_time'])
        risc0_stark_verify_time.append(value['stark_verify_time'])

    risc0_stark_verify_time = np.asarray(risc0_stark_verify_time) * 1000
    sp1_snark_verify_time = np.asarray(sp1_snark_verify_time) * 1000
    sp1_stark_verify_time = np.asarray(sp1_stark_verify_time) * 1000
    zinnia_verify_time = np.asarray(zinnia_verify_time) * 1000
    baseline_verify_time = np.asarray(baseline_verify_time) * 1000

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 3))
    width = 0.16
    x = np.arange(len(names))
    colors = ['mediumseagreen', 'cornflowerblue', 'wheat', 'orange', 'gold']
    ax1.bar(x + width * -2, zinnia_snark_proving_time, width, color=colors[0])
    ax1.bar(x + width * -1, baseline_snark_proving_time, width, color=colors[1])
    ax1.bar(x + width * 0, risc0_stark_proving_time, width, color=colors[2])
    ax1.bar(x + width * +1, sp1_stark_proving_time, width, color=colors[3])
    ax1.bar(x + width * +2, sp1_snark_proving_time, width, color=colors[4])
    ax1.tick_params(axis='x', labelrotation=90)
    ylabel = ax1.set_ylabel('Proving Time (s)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1]))
    ax1.set_xticks(x, names)
    ax1.set_yscale('log')
    ax2.bar(x + width * -2, zinnia_verify_time, width, color=colors[0])
    ax2.bar(x + width * -1, baseline_verify_time, width, color=colors[1])
    ax2.bar(x + width * 0, risc0_stark_verify_time, width, color=colors[2])
    ax2.bar(x + width * +1, sp1_stark_verify_time, width, color=colors[3])
    ax2.bar(x + width * +2, sp1_snark_verify_time, width, color=colors[4])
    ax2.tick_params(axis='x', labelrotation=90)
    ylabel = ax2.set_ylabel('Verifying Time (ms)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1]))
    ax2.set_xticks(x, names)
    ax2.set_yscale('log')
    fig.legend([AnyObject(c) for c in colors],
               ['Zinnia (zk-SNARK)', 'Baseline (zk-SNARK)', 'RISC0 (zk-STARK)', 'SP1 (zk-STARK)', 'SP1 (zk-SNARK)'],
               handler_map={
                   AnyObject: AnyObjectHandler()
               },
               loc='upper center', ncol=5,
               prop={'size': 8},
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    fig.savefig('results-zkvm-time.pdf', dpi=300)

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(10, 4.5))
    width = 0.16
    x = np.arange(len(names))
    colors = ['mediumseagreen', 'cornflowerblue', 'wheat', 'orange', 'gold']
    ax1.bar(x + width * -2, zinnia_snark_proving_time, width, color=colors[0])
    ax1.bar(x + width * -1, baseline_snark_proving_time, width, color=colors[1])
    ax1.bar(x + width * 0, risc0_stark_proving_time, width, color=colors[2])
    ax1.bar(x + width * +1, sp1_stark_proving_time, width, color=colors[3])
    ax1.bar(x + width * +2, sp1_snark_proving_time, width, color=colors[4])
    ax1.tick_params(labelbottom=False)
    ax1.set_xticks(x, names)
    ylabel = ax1.set_ylabel('Proving Time (s)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1]))
    ax1.set_yscale('log')
    ax2.bar(x + width * -2, zinnia_verify_time, width, color=colors[0])
    ax2.bar(x + width * -1, baseline_verify_time, width, color=colors[1])
    ax2.bar(x + width * 0, risc0_stark_verify_time, width, color=colors[2])
    ax2.bar(x + width * +1, sp1_stark_verify_time, width, color=colors[3])
    ax2.bar(x + width * +2, sp1_snark_verify_time, width, color=colors[4])
    ax2.tick_params(axis='x', labelrotation=90)
    ylabel = ax2.set_ylabel('Verifying Time (ms)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1]))
    ax2.set_xticks(x, names)
    ax2.set_yscale('log')
    fig.legend([AnyObject(c) for c in colors],
               ['Zinnia (zk-SNARK)', 'Baseline (zk-SNARK)', 'RISC0 (zk-STARK)', 'SP1 (zk-STARK)', 'SP1 (zk-SNARK)'],
               handler_map={
                   AnyObject: AnyObjectHandler()
               },
               loc='upper center', ncol=5,
               prop={'size': 8},
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    fig.savefig('results-zkvm-time-landscape.pdf', dpi=300)


def plot_ablation_study():
    with open('results.json', 'r') as f:
        zinnia_results_dict = json.load(f)
    with open('results-ablation.json', 'r') as f:
        ablation_results_dict = json.load(f)

    names = []
    acc_rates = []
    downgrade_rates = []
    prove_time_rates = []
    proving_time_baselines = []
    proving_time_optimizes = []
    proving_time_ablations = []
    verifying_time_baselines = []
    verifying_time_optimizes = []
    verifying_time_ablations = []
    verify_time_rates = []
    snark_size_rates = []
    zinnia_compile_times = []
    for key, value in zinnia_results_dict.items():
        names.append(NAME_MAPPING[key])
        zinnia_gates = value['zinnia']['advice_cells']
        halo2_gates = value['halo2']['advice_cells']
        zinnia_prove_time = value['zinnia']['proving_time']
        halo2_prove_time = value['halo2']['proving_time']
        zinnia_verify_time = value['zinnia']['verify_time']
        halo2_verify_time = value['halo2']['verify_time']
        zinnia_snark_size = value['zinnia']['snark_size']
        halo2_snark_size = value['halo2']['snark_size']
        proving_time_baselines.append(halo2_prove_time)
        proving_time_optimizes.append(zinnia_prove_time)
        proving_time_ablations.append(ablation_results_dict[key]['zinnia']['proving_time'])
        verifying_time_baselines.append(halo2_verify_time)
        verifying_time_optimizes.append(zinnia_verify_time)
        verifying_time_ablations.append(ablation_results_dict[key]['zinnia']['verify_time'])
        acc_rates.append(-(zinnia_gates - halo2_gates) / halo2_gates * 100)
        downgrade_rates.append((ablation_results_dict[key]['zinnia']['advice_cells'] - zinnia_gates) / halo2_gates * 100)
        prove_time_rates.append(-(zinnia_prove_time - halo2_prove_time) / halo2_prove_time * 100)
        verify_time_rates.append(-(zinnia_verify_time - halo2_verify_time) / halo2_verify_time * 100)
        snark_size_rates.append(-(zinnia_snark_size - halo2_snark_size) / halo2_snark_size * 100)
        zinnia_compile_times.append(value['zinnia_compile_time'])

    acc_rates = np.asarray(acc_rates)
    downgrade_rates = np.asarray(downgrade_rates)
    verifying_time_baselines = np.asarray(verifying_time_baselines) * 1000
    verifying_time_optimizes = np.asarray(verifying_time_optimizes) * 1000
    verifying_time_ablations = np.asarray(verifying_time_ablations) * 1000
    proving_time_baselines = np.asarray(proving_time_baselines)
    proving_time_optimizes = np.asarray(proving_time_optimizes)
    proving_time_ablations = np.asarray(proving_time_ablations)

    plt.rc('font', family='monospace', )
    # plt.rc('text', usetex=True)
    title_font = {'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 12}

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(4, 2, width_ratios=[1, 1], height_ratios=[1 for _ in range(4)])  # 2 rows, 2 columns
    ax1 = plt.subplot(gs[1:4, 0])  # Span all rows in the first column
    ax1_top = plt.subplot(gs[0, 0])  # Span all rows in the first column
    ax3 = plt.subplot(gs[2:4, 1])  # Bottom-right subplot
    ax2 = plt.subplot(gs[0:2, 1], sharex=ax3)  # Top-right subplot, sharing x-axis with bottom-right
    ax2.tick_params(labelbottom=False)

    # Plot the comparison of gate reductions
    ax1.bar(names, 100 - acc_rates, color='silver')
    ax1_top.bar(names, 100 - acc_rates, color='silver')
    ax1.bar(names, downgrade_rates, color='lightcoral', bottom=100 - acc_rates)
    ax1_top.bar(names, downgrade_rates, color='lightcoral', bottom=100 - acc_rates)
    ax1.tick_params(axis='x', labelrotation=90)
    ylabel = ax1.set_ylabel('No. of Gates (%)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1] + 0.2))
    ax1_top.spines.bottom.set_visible(False)
    ax1_top.set_xticks([])
    ax1.spines.top.set_visible(False)
    ax1_top.tick_params(labelbottom=False)
    ax1_top.xaxis.tick_top()
    ax1.xaxis.tick_bottom()
    ax1.set_ylim(0, 175)
    ax1_top.set_ylim(600, 625)
    ax1.axhline(y=100, color='black', linestyle='--', label='Baseline', linewidth=1)
    ax1.text(len(names) - 1.5, 105, 'Baseline', fontsize=8, color='black', ha='center')
    d = .15  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1_top.plot([0, 1], [0, 0], transform=ax1_top.transAxes, **kwargs)
    ax1.plot([0, 1], [1, 1], transform=ax1.transAxes, **kwargs)
    ax1.plot([(100 - acc_rates + downgrade_rates).argmax()], [175], **kwargs)
    ax1_top.plot([(100 - acc_rates + downgrade_rates).argmax()], [600], **kwargs)

    # Plot the comparison of proving & verifying time
    width = 0.25
    x = np.arange(len(names))
    colors = ['mediumseagreen', 'cornflowerblue', 'lightcoral']
    ax2.bar(x + width * -1, proving_time_optimizes, width, color=colors[0])
    ax2.bar(x + width * 0, proving_time_baselines, width, color=colors[1])
    ax2.bar(x + width * +1, proving_time_ablations, width, color=colors[2])
    ax2.set_xticks(x, names)
    ax2.set_yscale('log')
    ax2.set_xticks([])
    ax3.bar(x + width * -1, verifying_time_optimizes, width, color=colors[0])
    ax3.bar(x + width * 0, verifying_time_baselines, width, color=colors[1])
    ax3.bar(x + width * +1, verifying_time_ablations, width, color=colors[2])
    ylabel = ax3.set_ylabel('Verifying Time (ms)  Proving Time (s)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1]+0.6))
    ax3.set_xticks(x, names)
    ax3.set_yscale('log')
    ax3.tick_params(axis='x', labelrotation=90)
    fig.legend([AnyObject('silver'), AnyObject('lightcoral')],
               ['Zinnia (Optimized)', 'Zinnia (Unoptimized)'],
               handler_map={
                   AnyObject: AnyObjectHandler()
               },
               loc=(0.11, 0.92), ncol=2,
               prop={'size': 8},
               frameon=False)
    fig.legend([AnyObject('mediumseagreen'), AnyObject('cornflowerblue'), AnyObject('lightcoral')],
               ['Zinnia (Optimized)', 'Baseline', 'Zinnia (Unoptimized)'],
               handler_map={
                   AnyObject: AnyObjectHandler()
               },
               loc=(0.545, 0.92), ncol=3,
               prop={'size': 8},
               frameon=False)

    # Show the plot
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    fig.savefig('ablation-study.pdf', dpi=300)


def main():
    plot_evaluation_results()
    plot_performance_overviews()
    plot_ablation_study()


if __name__ == "__main__":
    main()
