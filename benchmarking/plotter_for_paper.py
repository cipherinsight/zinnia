import json

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
from scipy.stats import ks_2samp
from scipy.stats import binomtest
from scipy import stats
from scipy.stats import rankdata, norm

rcParams["axes.xmargin"] = 0.02

NAME_MAPPING = {
    'crypt::ecc':                'CRY······ECC',
    'crypt::poseidon':           'CRY·····Hash',
    'ds1000::case296':           'DS······#296',
    'ds1000::case309':           'DS······#309',
    'ds1000::case330':           'DS······#330',
    'ds1000::case360':           'DS······#360',
    'ds1000::case387':           'DS······#387',
    'ds1000::case418':           'DS······#418',
    'ds1000::case453':           'DS······#453',
    'ds1000::case459':           'DS······#459',
    'ds1000::case501':           'DS······#501',
    'ds1000::case510':           'DS······#510',
    'mlalgo::neuron':            'ML····Neuron',
    'mlalgo::kmeans':            'ML····KMeans',
    'mlalgo::linear_regression': 'ML····LinReg',
    'leetcode_array::p204':      'LC-Arr··#204',
    'leetcode_array::p832':      'LC-Arr··#832',
    'leetcode_dp::p740':         'LC-DP···#740',
    'leetcode_dp::p1137':        'LC-DP··#1137',
    'leetcode_graph::p3112':     'LC-Gra·#3112',
    'leetcode_graph::p997':      'LC-Gra··#997',
    'leetcode_math::p492':       'LC-Math·#492',
    'leetcode_math::p2125':      'LC-Math#2125',
    'leetcode_matrix::p73':      'LC-Mat···#73',
    'leetcode_matrix::p2133':    'LC-Mat·#2133',
}


def paired_t_test_one_sided(a, b, alternative='less'):
    """
    Perform a one-sided paired t-test.

    Parameters:
    - a, b: array-like, paired samples of the same length
    - alternative: 'less' for testing mean(a - b) < 0,
                   'greater' for testing mean(a - b) > 0

    Returns:
    - t_stat: the computed t statistic
    - p_value: the one-sided p-value
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("Samples a and b must have the same shape")

    diff = a - b
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se = std_diff / np.sqrt(n)
    t_stat = mean_diff / se
    df = n - 1

    if alternative == 'less':
        p_value = stats.t.cdf(t_stat, df)
    elif alternative == 'greater':
        p_value = 1 - stats.t.cdf(t_stat, df)
    else:
        raise ValueError("alternative must be 'less' or 'greater'")

    return t_stat, p_value


def wilcoxon_signed_rank(x, y):
    """
    Compute the Wilcoxon signed-rank test for paired samples x and y.

    Parameters
    ----------
    x, y : array-like, same shape
        Paired observations.

    Returns
    -------
    W : float
        The smaller of the sum of ranks for positive and negative differences.
    p_value : float
        Two-sided p-value (normal approximation).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("Both inputs must have the same shape.")

    # 1. Compute differences and drop zeros
    d = x - y
    non_zero = d != 0
    d = d[non_zero]
    n = len(d)
    if n == 0:
        raise ValueError("All differences are zero.")

    # 2. Get signed ranks
    signs = np.sign(d)
    abs_d = np.abs(d)
    ranks = rankdata(abs_d)  # average ranks for ties

    # 3. Sum ranks for positive and negative differences
    W_pos = np.sum(ranks[signs > 0])
    W_neg = np.sum(ranks[signs < 0])

    # Use the smaller sum as the test statistic
    W = min(W_pos, W_neg)

    # 4. Approximate p-value via normal approximation
    mean_W = n * (n + 1) / 4
    var_W = n * (n + 1) * (2 * n + 1) / 24
    sigma_W = np.sqrt(var_W)

    # continuity correction can be added by +/-0.5; omitted here
    z = (W - mean_W) / sigma_W
    p_value = 2 * norm.cdf(z)  # two-sided

    return W, p_value

def ks_test_stochastic_dominance(A, B):
    A = np.array(A)
    B = np.array(B)
    # alternative='less' tests F_A(x) >= F_B(x) for some x  (A stochastically less than B)
    result = ks_2samp(A, B, alternative='less', mode='asymp')
    return result.statistic, result.pvalue


def sign_test_binomial(A, B):
    """
    Perform a one-sided sign test (binomial) for H1: median(A) < median(B).

    Parameters:
    - A, B: array-like of paired measurements.

    Returns:
    - k: number of pairs where A < B
    - n: total number of non-tied pairs
    - p_value: one-sided p-value (H1: A < B)
    """
    A = np.array(A)
    B = np.array(B)
    diffs = A - B
    # Exclude ties (diff == 0)
    non_ties = diffs[diffs != 0]
    n = len(non_ties)
    # Count how many differences are negative (A < B)
    k = np.sum(non_ties < 0)

    # Perform binomial test for H1: probability of "success" (A < B) > 0.5
    test_result = binomtest(k, n, p=0.5, alternative='greater')
    return k, n, test_result.pvalue


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
    zinnia_gates = []
    halo2_gates = []
    for key, value in results_dict.items():
        names.append(NAME_MAPPING[key])
        _zinnia_gates = value['zinnia']['advice_cells']
        _halo2_gates = value['halo2']['advice_cells']
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
        zinnia_gates.append(_zinnia_gates)
        halo2_gates.append(_halo2_gates)
        acc_rates.append(-(_zinnia_gates - _halo2_gates) / _halo2_gates * 100)
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
    ax.bar(names, acc_rates, color='lightgreen', bottom=100 - acc_rates, label='Optimized\nConstraints (%)')
    ax.tick_params(axis='x', labelrotation=90)
    ylabel = ax.set_ylabel('No. of Constraints (%)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1]))
    ax.set_ylim(0, 117)
    ax.axhline(100, color='black', linewidth=1, linestyle='--')
    ax.text(len(names) - 1.5, 105, 'Baseline', fontsize=8, color='black', ha='center')
    fig.legend([AnyObject('silver'), AnyObject('lightgreen')],
               ['Zinnia Constraints (%)', 'Optimized Constraints (%)'],
               handler_map={
                   AnyObject: AnyObjectHandler()
               },
               loc='upper center', ncol=2,
               prop={'size': 8},
               frameon=False)
    fig.tight_layout(rect=[0, 0, 1.0, 0.97])
    plt.show()
    fig.savefig('constraint-reductions.pdf', dpi=300)

    fig = plt.figure(figsize=(10, 3.2))
    gs = gridspec.GridSpec(4, 2, width_ratios=[1, 1], height_ratios=[1 for _ in range(4)])  # 2 rows, 2 columns
    ax1 = plt.subplot(gs[0:4, 0])  # Span all rows in the first column
    ax3 = plt.subplot(gs[2:4, 1])  # Bottom-right subplot
    ax2 = plt.subplot(gs[0:2, 1], sharex=ax3)  # Top-right subplot, sharing x-axis with bottom-right
    ax2.tick_params(labelbottom=False)

    # Plot the comparison of gate reductions
    ax1.bar(names, 100 - acc_rates, color='silver')
    ax1.bar(names, acc_rates, color='lightgreen', bottom=100 - acc_rates)
    ax1.tick_params(axis='x', labelrotation=90)
    ax1.set_ylabel('No. of Constraints (%)', fontdict=title_font)
    ax1.set_ylim(0, 117)
    ax1.axhline(y=100, color='black', linestyle='--', label='Baseline', linewidth=1)
    ax1.text(len(names) - 3.5, 105, 'Halo2 Baseline', fontsize=8, color='black', ha='center')
    d = .15  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)

    # Plot the comparison of proving & verifying time
    width = 0.4
    x = np.arange(len(names))
    colors = ['mediumseagreen', 'cornflowerblue', 'lightcoral']
    ax2.bar(x + width * -1, proving_time_optimizes, width, color=colors[0])
    ax2.bar(x + width * 0, proving_time_baselines, width, color=colors[1])
    ax2.set_xticks(x, names)
    ax2.set_yscale('log')
    ax2.set_xticks([])
    ax3.bar(x + width * -1, verifying_time_optimizes, width, color=colors[0])
    ax3.bar(x + width * 0, verifying_time_baselines, width, color=colors[1])
    ylabel = ax3.set_ylabel('Verifying Time (ms)  Proving Time (s)', fontdict=title_font)
    # ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1] + 0.6))
    ax3.set_xticks(x, names)
    ax3.set_yscale('log')
    ax3.tick_params(axis='x', labelrotation=90)
    fig.legend([AnyObject('silver'), AnyObject('lightgreen')],
               ['Zinnia Constraints (%)', 'Optimized Constraints (%)'],
               handler_map={
                   AnyObject: AnyObjectHandler()
               },
               loc=(0.07, 0.92), ncol=2,
               prop={'size': 8},
               frameon=False)
    fig.legend([AnyObject('mediumseagreen'), AnyObject('cornflowerblue')],
               ['Zinnia', 'Halo2'],
               handler_map={
                   AnyObject: AnyObjectHandler()
               },
               loc=(0.7, 0.92), ncol=3,
               prop={'size': 8},
               frameon=False)

    # Show the plot
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    fig.savefig('constraint-reductions-compact.pdf', dpi=300)

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

    stat, p_value = wilcoxon_signed_rank(
        np.asarray(zinnia_snark_proving_time), np.asarray(sp1_stark_proving_time))
    # print(f"statistic: {t_stat:.4f}")
    print(f"p-value (H1: A < B): {p_value:.4f}")

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 3))
    width = 0.16
    x = np.arange(len(names))
    colors = ['mediumseagreen', 'lightcoral', 'cornflowerblue', 'orange']
    ax1.bar(x + width * -1.5, zinnia_snark_proving_time, width, color=colors[0])
    ax1.bar(x + width * -0.5, risc0_stark_proving_time, width, color=colors[1])
    ax1.bar(x + width * +0.5, sp1_stark_proving_time, width, color=colors[2])
    ax1.bar(x + width * +1.5, sp1_snark_proving_time, width, color=colors[3])
    ax1.tick_params(axis='x', labelrotation=90)
    ylabel = ax1.set_ylabel('Proving Time (s)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1]))
    ax1.set_xticks(x, names)
    ax1.set_yscale('log')
    ax2.bar(x + width * -1.5, zinnia_verify_time, width, color=colors[0])
    ax2.bar(x + width * -0.5, risc0_stark_verify_time, width, color=colors[1])
    ax2.bar(x + width * +0.5, sp1_stark_verify_time, width, color=colors[2])
    ax2.bar(x + width * +1.5, sp1_snark_verify_time, width, color=colors[3])
    ax2.tick_params(axis='x', labelrotation=90)
    ylabel = ax2.set_ylabel('Verifying Time (ms)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1]))
    ax2.set_xticks(x, names)
    ax2.set_yscale('log')
    fig.legend([AnyObject(c) for c in colors],
               ['Zinnia (zk-SNARK)', 'RISC0 (zk-STARK)', 'SP1 (zk-STARK)', 'SP1 (zk-SNARK)'],
               handler_map={
                   AnyObject: AnyObjectHandler()
               },
               loc='upper center', ncol=4,
               prop={'size': 8},
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    fig.savefig('results-zkvm-time.pdf', dpi=300)

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(5, 3.5))
    width = 0.2
    x = np.arange(len(names))
    colors = ['mediumseagreen', 'lightcoral', 'cornflowerblue', 'orange']
    ax1.bar(x + width * -1.5, zinnia_snark_proving_time, width, color=colors[0])
    ax1.bar(x + width * -0.5, risc0_stark_proving_time, width, color=colors[1])
    ax1.bar(x + width * +0.5, sp1_stark_proving_time, width, color=colors[2])
    ax1.bar(x + width * +1.5, sp1_snark_proving_time, width, color=colors[3])
    ax1.tick_params(labelbottom=False)
    ax1.set_xticks(x, names)
    ylabel = ax1.set_ylabel('  Proving Time (s)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1]))
    ax1.set_yscale('log')
    ax2.bar(x + width * -1.5, zinnia_verify_time, width, color=colors[0])
    ax2.bar(x + width * -0.5, risc0_stark_verify_time, width, color=colors[1])
    ax2.bar(x + width * +0.5, sp1_stark_verify_time, width, color=colors[2])
    ax2.bar(x + width * +1.5, sp1_snark_verify_time, width, color=colors[3])
    ax2.tick_params(axis='x')
    ylabel = ax2.set_ylabel('Verifying Time (ms)                  ', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1]))
    ax2.set_xticks(x, names)
    ax2.set_xticklabels(names, rotation=90)
    ax2.set_yscale('log')
    # print(np.mean(np.asarray(zinnia_verify_time) / risc0_stark_verify_time))
    fig.legend([AnyObject(c) for c in colors],
               ['Zinnia (zk-SNARK)', 'RISC0 (zk-STARK)', 'SP1 (zk-STARK)', 'SP1 (zk-SNARK)'],
               handler_map={
                   AnyObject: AnyObjectHandler()
               },
               loc=(0.24, 0.90), ncol=2,
               prop={'size': 8},
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    fig.savefig('results-zkvm-time-landscape.pdf', dpi=300)


def plot_ablation_study_old():
    with open('results.json', 'r') as f:
        zinnia_results_dict = json.load(f)
    with open('results-ablation-1.json', 'r') as f:
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
    zinnia_gates_list = []
    halo2_gates_list = []
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
        zinnia_gates_list.append(zinnia_gates)
        halo2_gates_list.append(halo2_gates)
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
    ylabel = ax1.set_ylabel('No. of Constraints (%)', fontdict=title_font)
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
    ax1.text(len(names) - 3.5, 105, 'Halo2 Baseline', fontsize=8, color='black', ha='center')
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
               ['Zinnia (Optimized)', 'Halo2', 'Zinnia (Unoptimized)'],
               handler_map={
                   AnyObject: AnyObjectHandler()
               },
               loc=(0.565, 0.92), ncol=3,
               prop={'size': 8},
               frameon=False)

    # Show the plot
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    fig.savefig('ablation-study.pdf', dpi=300)


def plot_performance_heatmap():
    with open('results.json', 'r') as f:
        results_dict = json.load(f)
    with open('results-noir.json', 'r') as f:
        noir_results_dict = json.load(f)

    names = []
    zinnia_plonk_gates = []
    zinnia_ultrahonk_gates = []
    halo2_gates = []
    noir_gates = []
    zinnia_plonk_proving_times = []
    zinnia_ultrahonk_proving_times = []
    zinnia_ultrahonk_verifying_times = []
    halo2_proving_times = []
    noir_proving_times = []
    zinnia_plonk_verifying_times = []
    noir_verifying_times = []
    halo2_verifying_times = []
    noir_excluded = []
    for i, (key, value) in enumerate(results_dict.items()):
        names.append(NAME_MAPPING[key])
        _zinnia_gates = value['zinnia']['advice_cells']
        _halo2_gates = value['halo2']['advice_cells']
        _zinnia_prove_time = value['zinnia']['proving_time']
        _halo2_prove_time = value['halo2']['proving_time']
        _zinnia_verify_time = value['zinnia']['verify_time']
        _halo2_verify_time = value['halo2']['verify_time']
        if key in noir_results_dict:
            _zinnia_ultrahonk_gates = noir_results_dict[key]['ours_on_noir']['total_gates']
            _noir_gates = noir_results_dict[key]['baseline_on_noir']['total_gates']
            _zinnia_ultrahonk_proving_time = noir_results_dict[key]['ours_on_noir']['proving_time']
            _zinnia_ultrahonk_verifying_time = noir_results_dict[key]['ours_on_noir']['verifying_time']
            _noir_proving_time = noir_results_dict[key]['baseline_on_noir']['proving_time']
            _noir_verifying_time = noir_results_dict[key]['baseline_on_noir']['verifying_time']
        else:
            noir_excluded.append(i)
            _zinnia_ultrahonk_gates = 0
            _noir_gates = 0
            _zinnia_ultrahonk_proving_time = 0
            _zinnia_ultrahonk_verifying_time = 0
            _noir_proving_time = 0
            _noir_verifying_time = 0
        zinnia_plonk_gates.append(_zinnia_gates)
        zinnia_ultrahonk_gates.append(_zinnia_ultrahonk_gates)
        halo2_gates.append(_halo2_gates)
        noir_gates.append(_noir_gates)
        zinnia_plonk_proving_times.append(_zinnia_prove_time)
        zinnia_plonk_verifying_times.append(_zinnia_verify_time)
        halo2_proving_times.append(_halo2_prove_time)
        halo2_verifying_times.append(_halo2_verify_time)
        zinnia_ultrahonk_proving_times.append(_zinnia_ultrahonk_proving_time)
        noir_proving_times.append(_noir_proving_time)
        noir_verifying_times.append(_noir_verifying_time)
        zinnia_ultrahonk_verifying_times.append(_zinnia_ultrahonk_verifying_time)

    plt.rc('font', family='monospace', )
    # plt.rc('text', usetex=True)
    title_font = {'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 12}

    # Generate random data for 25 tasks, 3 metrics (size, p-time, v-time), 2 conditions (I and II)
    tasks = names

    # A and B values for condition I and II
    A_I = np.asarray([zinnia_plonk_gates, zinnia_plonk_proving_times, zinnia_plonk_verifying_times]).transpose()
    B_I = np.asarray([halo2_gates, halo2_proving_times, halo2_verifying_times]).transpose()

    A_II = np.asarray([zinnia_ultrahonk_gates, zinnia_ultrahonk_proving_times, zinnia_ultrahonk_verifying_times]).transpose()
    B_II = np.asarray([noir_gates, noir_proving_times, noir_verifying_times]).transpose()


    # Compute improvement percentage: (B - A) / A
    imp_I = (A_I / B_I)
    imp_II = (A_II / B_II)

    mask = np.isnan(imp_II).any(axis=1)
    clean = imp_II[~mask]

    # Setup diverging colormap: red (negative) to white (zero) to green (positive)
    cmap = mcolors.LinearSegmentedColormap.from_list('mycolormap',
                                                     [(0, 'green'), (0.5, 'white'), (1, 'lightcoral')])

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(7, 8), sharey=True)
    fig.subplots_adjust(wspace=0.05)
    metrics = ['No. Constraints', 'Proving Time (s)', 'Verifying Time (ms)']

    for ax, imp, A_vals, title in zip(axes, [imp_I, imp_II], [A_I, A_II], ['Halo2 Baseline (PLONK)', 'Noir Baseline (UNTRAHONK)']):
        im = ax.imshow(imp, cmap=cmap, norm=mcolors.LogNorm(vmin=0.75, vmax=1.333333), aspect='auto')
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(metrics, fontdict=title_font, rotation=10)
        ax.set_title(title, fontdict=title_font, fontsize=14)
        # Annotate each cell with A's value
        for i in range(25):
            if i in noir_excluded and 'Noir' in title:
                for j in range(3):
                    ax.text(j, i, 'N/A', ha='center', va='center', fontsize=13)
            else:
                for j in range(3):
                    if j == 0:
                        ax.text(j, i, f"{int(A_vals[i, j])}", ha='center', va='center', fontsize=13)
                    elif j == 1:
                        ax.text(j, i, f"{A_vals[i, j]:.1f}", ha='center', va='center', fontsize=13)
                    elif j == 2:
                        tmp = A_vals[i, j] * 1000
                        ax.text(j, i, f"{tmp:.1f}", ha='center', va='center', fontsize=13)

    axes[0].set_yticks(np.arange(25))
    axes[0].set_yticklabels(tasks, fontsize=13)
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()

    legend_ax = inset_axes(
        ax,
        width="30%",  # or a fraction, or "3in", etc.
        height="20%",
        bbox_to_anchor=(0.05, 0.04, 0.3, 0.05),  # (left, bottom, w, h)
        bbox_transform=fig.transFigure,
        loc='lower left'
    )
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    legend_ax.imshow(gradient, aspect='auto', cmap=cmap)
    legend_ax.set_axis_off()  # hide default axes
    legend_ax.set_axis_on()
    legend_ax.xaxis.set_ticks_position('bottom')
    legend_ax.set_yticks([])
    N = gradient.shape[1]
    legend_ax.set_xticks([0, N // 2, N - 1])
    legend_ax.set_xticklabels(['ACC', '1$\\times$', 'DEC'], fontsize=8)
    for spine in legend_ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)

    plt.tight_layout(w_pad=0.05)
    plt.show()
    fig.savefig('circuit-performance.pdf', dpi=300)


def plot_ablation_study():
    with open('results.json', 'r') as f:
        zinnia_results_dict = json.load(f)
    with open('results-ablation-1.json', 'r') as f:
        ablation_results_1 = json.load(f)
    with open('results-ablation-2.json', 'r') as f:
        ablation_results_2 = json.load(f)
    with open('results-ablation-3.json', 'r') as f:
        ablation_results_3 = json.load(f)
    with open('results-ablation-4.json', 'r') as f:
        ablation_results_4 = json.load(f)

    names = []
    baseline_gates = []
    no_ablation_gates = []
    ablation_1_increased_gates = []
    ablation_2_increased_gates = []
    ablation_3_increased_gates = []
    ablation_4_increased_gates = []
    for key, value in zinnia_results_dict.items():
        names.append(NAME_MAPPING[key])
        zinnia_gates = value['zinnia']['advice_cells']
        halo2_gates = value['halo2']['advice_cells']
        ablation_1_gates = ablation_results_1[key]['zinnia']['advice_cells']
        ablation_2_gates = ablation_results_2[key]['zinnia']['advice_cells']
        ablation_3_gates = ablation_results_3[key]['zinnia']['advice_cells']
        ablation_4_gates = ablation_results_4[key]['zinnia']['advice_cells']
        baseline_gates.append(halo2_gates)
        no_ablation_gates.append(zinnia_gates)
        ablation_1_increased_gates.append(ablation_1_gates - zinnia_gates)
        ablation_2_increased_gates.append(ablation_2_gates - zinnia_gates)
        ablation_3_increased_gates.append(ablation_3_gates - zinnia_gates)
        ablation_4_increased_gates.append(ablation_4_gates - zinnia_gates)
    ablation_1_increased_gates = np.asarray(ablation_1_increased_gates)
    ablation_2_increased_gates = np.asarray(ablation_2_increased_gates)
    ablation_3_increased_gates = np.asarray(ablation_3_increased_gates)
    ablation_4_increased_gates = np.asarray(ablation_4_increased_gates)
    no_ablation_gates = np.asarray(no_ablation_gates)
    baseline_gates = np.asarray(baseline_gates)
    the_base_bar = (no_ablation_gates / baseline_gates) * 100
    ablation_dce_bar = (ablation_2_increased_gates / baseline_gates) * 100
    ablation_cse_bar = (ablation_3_increased_gates / baseline_gates) * 100
    ablation_pm_bar = (ablation_4_increased_gates / baseline_gates) * 100
    ablation_symex_bar = (ablation_1_increased_gates / baseline_gates) * 100
    for i in range(len(names)):
        sum = ablation_symex_bar[i] + the_base_bar[i] + ablation_dce_bar[i] + ablation_cse_bar[i] + ablation_pm_bar[i]
        # some symbolic execution pruned branches cannot be detected by the disabling optimizations
        # so we need to ensure the sum is at least 100%
        if sum < 100:
            ablation_symex_bar[i] += 100 - sum

    print('mean::', ((ablation_symex_bar + the_base_bar + ablation_dce_bar + ablation_cse_bar + ablation_pm_bar)).mean())

    plt.rc('font', family='monospace', )
    # plt.rc('text', usetex=True)
    title_font = {'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 12}
    # Plot the comparison of gate reductions
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(names, the_base_bar, color='silver')
    ax.bar(names, ablation_dce_bar, color='wheat', bottom=the_base_bar, label='DCE')
    ax.bar(names, ablation_cse_bar, color='lightskyblue', bottom=the_base_bar + ablation_dce_bar, label='CSE')
    ax.bar(names, ablation_pm_bar, color='mediumpurple', bottom=the_base_bar + ablation_dce_bar + ablation_cse_bar, label='PM')
    ax.bar(names, ablation_symex_bar, color='mediumseagreen', bottom=ablation_pm_bar + the_base_bar + ablation_dce_bar + ablation_cse_bar, label='SymEx')
    ax.tick_params(axis='x', labelrotation=90)
    ylabel = ax.set_ylabel('No. of Constraints (%)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1]))
    ax.set_ylim(0, 400)
    ax.axhline(100, color='black', linewidth=1, linestyle='--')
    ax.text(len(names) - 2, 70, 'Baseline', fontsize=8, color='black', ha='center')
    fig.legend([AnyObject('silver'), AnyObject('wheat'), AnyObject('lightskyblue'), AnyObject('mediumpurple'), AnyObject('mediumseagreen')],
               ['No Ablation', 'w/o Dead Code Elimination', 'w/o Common Sub-expression Elimination', 'w/o Pattern Matching Rewrites', 'w/o Symbolic Execution Pruning'],
               handler_map={
                   AnyObject: AnyObjectHandler()
               },
               loc=(0.16, 0.64), ncol=1,
               prop={'size': 8},
               frameon=False)
    fig.tight_layout()
    plt.show()
    fig.savefig('ablation-study.pdf', dpi=300)


def main():
    plot_evaluation_results()
    plot_performance_overviews()
    plot_ablation_study()
    plot_performance_heatmap()


if __name__ == "__main__":
    main()
