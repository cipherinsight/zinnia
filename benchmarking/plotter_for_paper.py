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
    'crypt::ecc':                'CR··ECC',
    'crypt::poseidon':           'CR·PSDN',
    'crypt::elgamal':            'CR·ELGM',
    'crypt::merkle':             'CR·MRKL',
    'crypt::mimc':               'CR·MIMC',
    'mlalgo::neuron':            'ML·NEUR',
    'mlalgo::kmeans':            'ML·KMNS',
    'mlalgo::linear_regression': 'ML·LNRG',
    'mlalgo::svm':               'ML··SVM',
    'mlalgo::logistic':          'ML·LREG',
    'mlalgo::decision_tree':     'ML··DCT',
    'mlalgo::bayes':             'ML·BAYE',
    'mlalgo::mlp':               'ML··MLP',
    'leetcode_array::p204':      'LC#0204',
    'leetcode_array::p832':      'LC#0832',
    'leetcode_dp::p740':         'LC#0740',
    'leetcode_dp::p1137':        'LC#1137',
    'leetcode_graph::p3112':     'LC#3112',
    'leetcode_graph::p997':      'LC#0997',
    'leetcode_math::p492':       'LC#0492',
    'leetcode_math::p2125':      'LC#2125',
    'leetcode_matrix::p73':      'LC#0073',
    'leetcode_matrix::p2133':    'LC#2133',
}
DS1000 = [
    'case295', 'case296', 'case297', 'case298', 'case299', 'case300', 'case301', 'case302', 'case303', 'case304', 'case309', 'case310', 'case313', 'case318', 'case319', 'case322', 'case323', 'case324', 'case329', 'case330', 'case334', 'case335', 'case336', 'case337', 'case338', 'case339', 'case353', 'case354', 'case360', 'case368', 'case369', 'case370', 'case373', 'case374', 'case375', 'case385', 'case387', 'case388', 'case389', 'case390', 'case391', 'case392', 'case393', 'case406', 'case407', 'case408', 'case409', 'case414', 'case415', 'case416', 'case417', 'case418', 'case419', 'case420', 'case428', 'case429', 'case430', 'case431', 'case433', 'case434', 'case435', 'case436', 'case437', 'case438', 'case440', 'case441', 'case452', 'case453', 'case459', 'case480', 'case501', 'case507', 'case510'
]
for key in DS1000:
    NAME_MAPPING[f'ds1000::{key}'] = f'DS#0{key[4:]}'


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
    with open('results-cairo.json', 'r') as f:
        cairo_results_dict = json.load(f)

    plt.rc('font', family='monospace', )
    title_font = {'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 12}

    # Sort keys by display name (alphabetical)
    sorted_keys = sorted(sp1_results_dict.keys(), key=lambda k: NAME_MAPPING.get(k, k))
    names = [NAME_MAPPING.get(k, k) for k in sorted_keys]

    # Collect data in sorted order
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
    cairo_stark_proving_time = []
    cairo_stark_verify_time = []

    for key in sorted_keys:
        sp1_val = sp1_results_dict[key]
        zinnia_val = zinnia_results_dict[key]
        # base halo2 info stored inside zinnia_results_dict
        baseline_snark_proving_time.append(zinnia_val['halo2']['proving_time'])
        baseline_verify_time.append(zinnia_val['halo2']['verify_time'])

        zinnia_snark_proving_time.append(zinnia_val['zinnia']['proving_time'])
        zinnia_verify_time.append(zinnia_val['zinnia']['verify_time'])
        zinnia_snark_size.append(zinnia_val['zinnia']['snark_size'])

        sp1_stark_proving_time.append(sp1_val.get('stark_proving_time', 0))
        sp1_snark_proving_time.append(sp1_val.get('snark_proving_time', 0))
        sp1_stark_verify_time.append(sp1_val.get('stark_verify_time', 0))
        sp1_snark_verify_time.append(sp1_val.get('snark_verify_time', 0))
        sp1_snark_size.append(sp1_val.get('snark_size', 0))

        # risc0 and cairo may miss some keys
        risc0_val = risc0_results_dict.get(key, {})
        risc0_stark_proving_time.append(risc0_val.get('stark_proving_time', 0))
        risc0_stark_verify_time.append(risc0_val.get('stark_verify_time', 0))

        cairo_val = cairo_results_dict.get(key, {})
        cairo_stark_proving_time.append(cairo_val.get('stark_proving_time', 0))
        cairo_stark_verify_time.append(cairo_val.get('stark_verify_time', 0))

    # Convert to numpy arrays and convert verify times to ms
    zinnia_snark_proving_time = np.asarray(zinnia_snark_proving_time)
    baseline_snark_proving_time = np.asarray(baseline_snark_proving_time)
    sp1_stark_proving_time = np.asarray(sp1_stark_proving_time)
    sp1_snark_proving_time = np.asarray(sp1_snark_proving_time)
    risc0_stark_proving_time = np.asarray(risc0_stark_proving_time)
    cairo_stark_proving_time = np.asarray(cairo_stark_proving_time)

    zinnia_verify_time = np.asarray(zinnia_verify_time) * 1000
    baseline_verify_time = np.asarray(baseline_verify_time) * 1000
    sp1_stark_verify_time = np.asarray(sp1_stark_verify_time) * 1000
    sp1_snark_verify_time = np.asarray(sp1_snark_verify_time) * 1000
    risc0_stark_verify_time = np.asarray(risc0_stark_verify_time) * 1000
    cairo_stark_verify_time = np.asarray(cairo_stark_verify_time) * 1000

    # Helper: mask zeros so they don't appear on log scale
    def mask_zero(arr):
        a = np.asarray(arr, dtype=float)
        a[a <= 0] = np.nan
        return a

    # Prepare scatter plot with offsets to avoid overlap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 3.5))
    x = np.arange(len(names))
    colors = ['mediumseagreen', 'lightcoral', 'cornflowerblue', 'orange', 'gray']
    labels = ['Zinnia (zk-SNARK)', 'RISC0 (zk-STARK)', 'SP1 (zk-STARK)', 'SP1 (zk-SNARK)', 'Cairo (zk-STARK)']
    # series order for proving (seconds) and verifying (ms)
    proving_series = [
        zinnia_snark_proving_time,
        risc0_stark_proving_time,
        sp1_stark_proving_time,
        sp1_snark_proving_time,
        cairo_stark_proving_time,
    ]
    verifying_series = [
        zinnia_verify_time,
        risc0_stark_verify_time,
        sp1_stark_verify_time,
        sp1_snark_verify_time,
        cairo_stark_verify_time,
    ]

    marker_size = 40
    for ser, col, lbl in zip(proving_series, colors, labels):
        y = mask_zero(ser)
        ax1.scatter(x, y, marker='o', s=marker_size, color=col, edgecolors='k', linewidths=0.6, alpha=0.95)

    ax1.set_yscale('log')
    ax1.tick_params(labelbottom=False)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel('  Proving Time (s)', fontdict=title_font)
    ax1.set_ylim(bottom=np.nanmin([np.nanmin(mask_zero(s)) for s in proving_series if np.nanmax(s) > 0]) * 0.5 if len(names) else None)

    # verifying scatter
    for ser, col, lbl in zip(verifying_series, colors, labels):
        y = mask_zero(ser)
        ax2.scatter(x, y, marker='o', s=marker_size, color=col, edgecolors='k', linewidths=0.6, alpha=0.95)

    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=90)
    ax2.set_ylabel('Verifying Time (ms)                  ', fontdict=title_font)

    # Legend (using color patches as before)
    fig.legend([AnyObject(c) for c in colors],
               labels,
               handler_map={AnyObject: AnyObjectHandler()},
               loc=(0.19, 0.88), ncol=5,
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

    # Sort keys by display name (alphabetical)
    sorted_keys = sorted(results_dict.keys(), key=lambda k: NAME_MAPPING.get(k, k))
    names = [NAME_MAPPING.get(k, k) for k in sorted_keys]

    zinnia_plonk_gates = []
    zinnia_plonk_proving_times = []
    zinnia_plonk_verifying_times = []
    halo2_gates = []
    halo2_proving_times = []
    halo2_verifying_times = []
    zinnia_ultrahonk_gates = []
    zinnia_ultrahonk_proving_times = []
    zinnia_ultrahonk_verifying_times = []
    noir_gates = []
    noir_proving_times = []
    noir_verifying_times = []
    noir_excluded = []

    # Build arrays in sorted order and record noir-missing indices
    for i, key in enumerate(sorted_keys):
        value = results_dict[key]
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
        zinnia_plonk_proving_times.append(_zinnia_prove_time)
        zinnia_plonk_verifying_times.append(_zinnia_verify_time)
        halo2_gates.append(_halo2_gates)
        halo2_proving_times.append(_halo2_prove_time)
        halo2_verifying_times.append(_halo2_verify_time)

        zinnia_ultrahonk_gates.append(_zinnia_ultrahonk_gates)
        zinnia_ultrahonk_proving_times.append(_zinnia_ultrahonk_proving_time)
        zinnia_ultrahonk_verifying_times.append(_zinnia_ultrahonk_verifying_time)
        noir_gates.append(_noir_gates)
        noir_proving_times.append(_noir_proving_time)
        noir_verifying_times.append(_noir_verifying_time)

    plt.rc('font', family='monospace', )
    title_font = {'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 11}

    # Build data matrices (n_tasks x 3 metrics)
    A_I = np.asarray([zinnia_plonk_gates, zinnia_plonk_proving_times, zinnia_plonk_verifying_times]).transpose()
    B_I = np.asarray([halo2_gates, halo2_proving_times, halo2_verifying_times]).transpose()
    A_II = np.asarray([zinnia_ultrahonk_gates, zinnia_ultrahonk_proving_times, zinnia_ultrahonk_verifying_times]).transpose()
    B_II = np.asarray([noir_gates, noir_proving_times, noir_verifying_times]).transpose()

    # Compute improvement ratios
    imp_I = (A_I / B_I)
    imp_II = (A_II / B_II)

    # Ensure we use the dynamic number of tasks
    n_tasks = len(names)
    metrics = ['    No. of\nConstraints', 'Proving\nTime (s)', ' Verifying\nTime (ms)']

    # Setup colormap and plotting
    cmap = mcolors.LinearSegmentedColormap.from_list('mycolormap',
                                                     [(0, 'green'), (0.5, 'white'), (1, 'lightcoral')])

    # Create stacked subplots: top = Circom (PLONK), bottom = Noir (UNTRAHONK); share x-axis (task names)
    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    # make room on the left for the left-mounted titles
    fig.subplots_adjust(hspace=0.00, left=0.14)

    for ax, imp, A_vals, title in zip(axes, [imp_I, imp_II], [A_I, A_II], ['Halo2 (PLONK)', 'Noir (UNTRAHONK)']):
        # Force third metric to neutral factor 1 for visualization (as previous behavior)
        imp_copy = imp.copy()
        if imp_copy.shape[0] > 0:
            imp_copy[:, 2] = 1.0
        # Transpose so metrics are on y-axis and tasks on x-axis => shape (3, n_tasks)
        im = ax.imshow(imp_copy.T, cmap=cmap, norm=mcolors.LogNorm(vmin=0.75, vmax=1.333333), aspect='auto')

        # x axis: tasks (names) - tasks labels shown only on bottom subplot (shared x-axis)
        ax.set_xticks(np.arange(n_tasks))
        if ax is axes[-1]:
            ax.set_xticklabels(names, rotation=90)
            ax.xaxis.tick_bottom()
        else:
            ax.set_xticklabels([''] * n_tasks)
            ax.xaxis.set_tick_params(labelbottom=False)

        # y axis: metrics on the right side now
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_yticklabels(metrics, fontdict={'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 7}, rotation=90, va='center', )
        ax.yaxis.tick_right()

        # Place title on the left side as vertical text
        ax.text(-0.02, 0.5, title, transform=ax.transAxes, fontdict=title_font, va='center', ha='left', rotation=90)

        # Annotate each cell with A's value. For imshow(imp.T), x index = task, y index = metric
        for i in range(n_tasks):
            for j in range(len(metrics)):
                if (i in noir_excluded) and ('Noir' in title):
                    ax.text(i, j, 'N/A', ha='center', va='center', fontsize=7, rotation=90)
                else:
                    if j == 0:
                        ax.text(i, j, f"{int(A_vals[i, j])}", ha='center', va='center', fontsize=7, rotation=90)
                    elif j == 1:
                        ax.text(i, j, f"{A_vals[i, j]:.1f}", ha='center', va='center', fontsize=7, rotation=90)
                    elif j == 2:
                        tmp = A_vals[i, j] * 1000
                        ax.text(i, j, f"{tmp:.1f}", ha='center', va='center', fontsize=7, rotation=90)

    legend_ax = inset_axes(
        ax,
        width="8%",  # narrow vertical bar
        height="20%",  # tall
        bbox_to_anchor=(0.962, 0.035, 0.03, 0.6),  # (left, bottom, w, h) in figure coords
        bbox_transform=fig.transFigure,
        loc='lower left'
    )
    # vertical gradient (N x 1)
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    legend_ax.imshow(gradient, aspect='auto', cmap=cmap, origin='lower')
    # hide x-axis, show y-axis ticks on the right
    legend_ax.set_xticks([])
    legend_ax.set_yticks([0, gradient.shape[0] // 2, gradient.shape[0] - 1])
    legend_ax.set_yticklabels(['0.75×', '1×', '1.33×'], fontsize=6, rotation=90, va='center')
    legend_ax.yaxis.set_ticks_position('right')
    legend_ax.xaxis.set_ticks_position('none')
    # draw thin border
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

    # Sort keys by their display names (alphabetical)
    sorted_keys = sorted(zinnia_results_dict.keys(), key=lambda k: NAME_MAPPING.get(k, k))

    names = []
    baseline_gates = []
    no_ablation_gates = []
    ablation_1_increased_gates = []
    ablation_2_increased_gates = []
    ablation_3_increased_gates = []
    ablation_4_increased_gates = []
    for key in sorted_keys:
        value = zinnia_results_dict[key]
        names.append(NAME_MAPPING.get(key, key))
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

    print('Mean increase in ablation:', ((ablation_symex_bar + the_base_bar + ablation_dce_bar + ablation_cse_bar + ablation_pm_bar)).mean())
    plt.rc('font', family='monospace', )
    # plt.rc('text', usetex=True)
    title_font = {'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 12}
    # Plot the comparison of gate reductions
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(names, the_base_bar, color='silver')
    ax.bar(names, ablation_dce_bar, color='wheat', bottom=the_base_bar, label='DCE')
    ax.bar(names, ablation_cse_bar, color='lightskyblue', bottom=the_base_bar + ablation_dce_bar, label='CSE')
    ax.bar(names, ablation_pm_bar, color='mediumpurple', bottom=the_base_bar + ablation_dce_bar + ablation_cse_bar, label='PM')
    ax.bar(names, ablation_symex_bar, color='mediumseagreen', bottom=ablation_pm_bar + the_base_bar + ablation_dce_bar + ablation_cse_bar, label='SymEx')
    ax.tick_params(axis='x', labelrotation=90)
    ylabel = ax.set_ylabel('Arithmetic Circuit Size (%)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1]-0.15))
    ax.set_ylim(0, 620)
    ax.axhline(100, color='black', linewidth=1, linestyle='--')
    ax.text(len(names) - 7, 125, 'Baseline (100%)', fontsize=8, color='black', ha='center')
    fig.legend([AnyObject('grey'), AnyObject('wheat'), AnyObject('lightskyblue'), AnyObject('mediumpurple'), AnyObject('mediumseagreen')],
               ['No Ablation', 'w/o Dead Code Elimination', 'w/o Common Sub-expression Elimination', 'w/o Pattern Matching Rewrites', 'w/o Symbolic Execution Path Pruning'],
               handler_map={
                   AnyObject: AnyObjectHandler()
               },
               loc=(0.07, 0.56), ncol=1,
               frameon=False)
    fig.tight_layout()
    plt.show()
    fig.savefig('ablation-study.pdf', dpi=300)


def plot_compile_time_scalability():
    with open('results.json', 'r') as f:
        zinnia_results_dict = json.load(f)

    # Sort keys by display names (alphabetical)
    sorted_keys = sorted(zinnia_results_dict.keys(), key=lambda k: NAME_MAPPING.get(k, k))

    names = []
    rust_compile_times = []
    ast_ir_transform_times = []
    smt_reasoning_times = []
    exec_ir_pass_times = []
    code_gen_times = []

    for key in sorted_keys:
        value = zinnia_results_dict[key]
        names.append(NAME_MAPPING.get(key, key))
        rust_compile_times.append(value['halo2']['cargo_compile_time'])
        ast_ir_transform_times.append(value['zinnia_compile_time']['time_transform'])
        smt_reasoning_times.append(value['zinnia_compile_time']['time_smt'])
        exec_ir_pass_times.append(value['zinnia_compile_time']['time_ir_pass'])
        code_gen_times.append(value['zinnia_compile_time']['time_code_gen'])

    # Convert to numpy arrays
    rust_compile_times = np.asarray(rust_compile_times)
    ast_ir_transform_times = np.asarray(ast_ir_transform_times)
    smt_reasoning_times = np.asarray(smt_reasoning_times)
    exec_ir_pass_times = np.asarray(exec_ir_pass_times)
    code_gen_times = np.asarray(code_gen_times)

    total_zinnia_compile_times = (
        ast_ir_transform_times + smt_reasoning_times + exec_ir_pass_times + code_gen_times
    )

    # Normalize for bottom chart (100%)
    total_nonzero = np.where(total_zinnia_compile_times == 0, 1, total_zinnia_compile_times)
    ast_norm = ast_ir_transform_times / total_nonzero * 100
    smt_norm = smt_reasoning_times / total_nonzero * 100
    exec_norm = exec_ir_pass_times / total_nonzero * 100
    code_norm = code_gen_times / total_nonzero * 100

    # Plot setup
    plt.rc('font', family='monospace')
    title_font = {'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 12}

    # Two stacked bar charts, shared x-axis
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 4), sharex=True,
        gridspec_kw={'height_ratios': [1.5, 1]}
    )

    ax1.bar(names, ast_ir_transform_times, color='mediumseagreen')
    ax1.bar(names, smt_reasoning_times, color='mediumpurple',
            bottom=ast_ir_transform_times)
    ax1.bar(names, exec_ir_pass_times, color='wheat',
            bottom=ast_ir_transform_times + smt_reasoning_times)
    ax1.bar(names, code_gen_times, color='lightskyblue',
            bottom=ast_ir_transform_times + smt_reasoning_times + exec_ir_pass_times)

    ax1.set_ylabel('Compilation Time (s)', fontdict=title_font)
    # ax1.set_yscale('log')
    ax1.tick_params(axis='x', labelbottom=False)

    ax2.bar(names, ast_norm, color='mediumseagreen')
    ax2.bar(names, smt_norm, color='mediumpurple', bottom=ast_norm)
    ax2.bar(names, exec_norm, color='wheat', bottom=ast_norm + smt_norm)
    ax2.bar(names, code_norm, color='lightskyblue',
            bottom=ast_norm + smt_norm + exec_norm)

    ax2.set_ylabel('Percentage (%)', fontdict=title_font)
    ax2.tick_params(axis='x', labelrotation=90)
    ax2.set_ylim(0, 100)

    fig.legend(
        [AnyObject('mediumseagreen'), AnyObject('mediumpurple'), AnyObject('wheat'), AnyObject('lightskyblue')],
        ['AST Traversal & IR Generation', 'SMT Reasoning', 'Executing IR Passes', 'Generating ZK Circuit'],
        handler_map={AnyObject: AnyObjectHandler()}, loc=(0.07, 0.84), ncol=2, frameon=False
    )
    fig.tight_layout(h_pad=0.1)
    plt.show()
    fig.savefig('compilation-scalability.pdf', dpi=300, bbox_inches='tight')


def compute_pct_advantage(baseline_arr, optimized_arr):
    """
    Return (mean_percent_advantage, n) where percent advantage = (baseline - optimized) / baseline * 100.
    Only entries with baseline>0 and optimized>0 are considered.
    """
    b = np.asarray(baseline_arr, dtype=float)
    o = np.asarray(optimized_arr, dtype=float)
    mask = (b > 0) & (o > 0) & (~np.isnan(b)) & (~np.isnan(o))
    if np.sum(mask) == 0:
        return float('nan'), 0
    pct = (b[mask] - o[mask]) / b[mask] * 100.0
    return float(np.nanmean(pct)), int(np.sum(mask))


# NEW helper: compute mean multiplicative ratio baseline/optimized
def compute_mean_ratio(baseline_arr, optimized_arr):
    """
    Return (mean_ratio, n) where ratio = baseline / optimized, averaged across valid entries.
    Only entries with baseline>0 and optimized>0 are considered.
    """
    b = np.asarray(baseline_arr, dtype=float)
    o = np.asarray(optimized_arr, dtype=float)
    mask = (b > 0) & (o > 0) & (~np.isnan(b)) & (~np.isnan(o))
    if np.sum(mask) == 0:
        return float('nan'), 0
    ratios = b[mask] / o[mask]
    return float(np.nanmean(ratios)), int(np.sum(mask))


def print_average_advantages():
    """
    Compute and print average proving/verifying time advantages of Zinnia over:
      - Halo2, Noir, Cairo, SP1 (STARK), SP1 (SNARK), Risc0
    And average constraint-count advantages over Halo2 and Noir.
    """
    with open('results.json', 'r') as f:
        zinnia_results = json.load(f)
    with open('results-noir.json', 'r') as f:
        noir_results = json.load(f)
    with open('results-sp1.json', 'r') as f:
        sp1_results = json.load(f)
    with open('results-risc0.json', 'r') as f:
        risc0_results = json.load(f)
    with open('results-cairo.json', 'r') as f:
        cairo_results = json.load(f)

    # Use keys present in zinnia results for Halo2 comparisons
    keys = sorted(zinnia_results.keys(), key=lambda k: NAME_MAPPING.get(k, k))

    # Prepare arrays for Halo2 comparisons (halo2 stored inside zinnia results)
    halo2_prove = []
    halo2_verify = []
    zinnia_prove = []
    zinnia_verify = []
    halo2_gates = []
    zinnia_gates = []

    for k in keys:
        z = zinnia_results[k]
        halo2_prove.append(z['halo2'].get('proving_time', 0))
        halo2_verify.append(z['halo2'].get('verify_time', 0) * 1000.0)  # ms
        zinnia_prove.append(z['zinnia'].get('proving_time', 0))
        zinnia_verify.append(z['zinnia'].get('verify_time', 0) * 1000.0)  # ms
        halo2_gates.append(z['halo2'].get('advice_cells', 0))
        zinnia_gates.append(z['zinnia'].get('advice_cells', 0))

    # Halo2 averages
    mean_p_adv_halo2, n_p_halo2 = compute_pct_advantage(halo2_prove, zinnia_prove)
    mean_v_adv_halo2, n_v_halo2 = compute_pct_advantage(halo2_verify, zinnia_verify)
    mean_c_adv_halo2, n_c_halo2 = compute_pct_advantage(halo2_gates, zinnia_gates)
    mean_p_ratio_halo2, _ = compute_mean_ratio(halo2_prove, zinnia_prove)
    mean_v_ratio_halo2, _ = compute_mean_ratio(halo2_verify, zinnia_verify)
    mean_c_ratio_halo2, _ = compute_mean_ratio(halo2_gates, zinnia_gates)

    # Noir comparisons: only keys present in noir_results
    noir_keys = sorted([k for k in keys if k in noir_results], key=lambda k: NAME_MAPPING.get(k, k))
    noir_baseline_prove = []
    noir_baseline_verify = []
    noir_ours_prove = []
    noir_ours_verify = []
    noir_baseline_gates = []
    noir_ours_gates = []

    for k in noir_keys:
        nrec = noir_results[k]
        baseline = nrec.get('baseline_on_noir', {})
        ours = nrec.get('ours_on_noir', {})
        noir_baseline_prove.append(baseline.get('proving_time', 0))
        noir_baseline_verify.append(baseline.get('verifying_time', 0) * 1000.0)
        noir_ours_prove.append(ours.get('proving_time', 0))
        noir_ours_verify.append(ours.get('verifying_time', 0) * 1000.0)
        noir_baseline_gates.append(baseline.get('total_gates', 0))
        noir_ours_gates.append(ours.get('total_gates', 0))

    mean_p_adv_noir, n_p_noir = compute_pct_advantage(noir_baseline_prove, noir_ours_prove)
    mean_v_adv_noir, n_v_noir = compute_pct_advantage(noir_baseline_verify, noir_ours_verify)
    mean_c_adv_noir, n_c_noir = compute_pct_advantage(noir_baseline_gates, noir_ours_gates)
    mean_p_ratio_noir, _ = compute_mean_ratio(noir_baseline_prove, noir_ours_prove)
    mean_v_ratio_noir, _ = compute_mean_ratio(noir_baseline_verify, noir_ours_verify)
    mean_c_ratio_noir, _ = compute_mean_ratio(noir_baseline_gates, noir_ours_gates)

    # SP1 comparisons (may miss some keys)
    sp1_keys = sorted([k for k in keys if k in sp1_results], key=lambda k: NAME_MAPPING.get(k, k))
    sp1_stark_baseline_prove = []
    sp1_stark_baseline_verify = []
    sp1_snark_baseline_prove = []
    sp1_snark_baseline_verify = []
    z_prove_for_sp1 = []
    z_verify_for_sp1 = []

    for k in sp1_keys:
        s = sp1_results[k]
        # SP1 may have stark_* and snark_* fields; use get with 0 default
        sp1_stark_baseline_prove.append(s.get('stark_proving_time', 0))
        sp1_stark_baseline_verify.append(s.get('stark_verify_time', 0) * 1000.0)
        sp1_snark_baseline_prove.append(s.get('snark_proving_time', 0))
        sp1_snark_baseline_verify.append(s.get('snark_verify_time', 0) * 1000.0)
        # zinnia values for this key
        z = zinnia_results[k]
        z_prove_for_sp1.append(z['zinnia'].get('proving_time', 0))
        z_verify_for_sp1.append(z['zinnia'].get('verify_time', 0) * 1000.0)

    mean_p_adv_sp1_stark, n_p_sp1_stark = compute_pct_advantage(sp1_stark_baseline_prove, z_prove_for_sp1)
    mean_v_adv_sp1_stark, n_v_sp1_stark = compute_pct_advantage(sp1_stark_baseline_verify, z_verify_for_sp1)
    mean_p_adv_sp1_snark, n_p_sp1_snark = compute_pct_advantage(sp1_snark_baseline_prove, z_prove_for_sp1)
    mean_v_adv_sp1_snark, n_v_sp1_snark = compute_pct_advantage(sp1_snark_baseline_verify, z_verify_for_sp1)
    mean_p_ratio_sp1_stark, _ = compute_mean_ratio(sp1_stark_baseline_prove, z_prove_for_sp1)
    mean_v_ratio_sp1_stark, _ = compute_mean_ratio(sp1_stark_baseline_verify, z_verify_for_sp1)
    mean_p_ratio_sp1_snark, _ = compute_mean_ratio(sp1_snark_baseline_prove, z_prove_for_sp1)
    mean_v_ratio_sp1_snark, _ = compute_mean_ratio(sp1_snark_baseline_verify, z_verify_for_sp1)

    # Risc0 comparisons
    risc0_keys = sorted([k for k in keys if k in risc0_results], key=lambda k: NAME_MAPPING.get(k, k))
    risc0_prove = []
    risc0_verify = []
    z_prove_risc0 = []
    z_verify_risc0 = []
    for k in risc0_keys:
        r = risc0_results[k]
        risc0_prove.append(r.get('stark_proving_time', 0))
        risc0_verify.append(r.get('stark_verify_time', 0) * 1000.0)
        z = zinnia_results[k]
        z_prove_risc0.append(z['zinnia'].get('proving_time', 0))
        z_verify_risc0.append(z['zinnia'].get('verify_time', 0) * 1000.0)
    mean_p_adv_risc0, n_p_risc0 = compute_pct_advantage(risc0_prove, z_prove_risc0)
    mean_v_adv_risc0, n_v_risc0 = compute_pct_advantage(risc0_verify, z_verify_risc0)
    mean_p_ratio_risc0, _ = compute_mean_ratio(risc0_prove, z_prove_risc0)
    mean_v_ratio_risc0, _ = compute_mean_ratio(risc0_verify, z_verify_risc0)

    # Cairo comparisons
    cairo_keys = sorted([k for k in keys if k in cairo_results], key=lambda k: NAME_MAPPING.get(k, k))
    cairo_prove = []
    cairo_verify = []
    z_prove_cairo = []
    z_verify_cairo = []
    for k in cairo_keys:
        c = cairo_results[k]
        cairo_prove.append(c.get('stark_proving_time', 0))
        cairo_verify.append(c.get('stark_verify_time', 0) * 1000.0)
        z = zinnia_results[k]
        z_prove_cairo.append(z['zinnia'].get('proving_time', 0))
        z_verify_cairo.append(z['zinnia'].get('verify_time', 0) * 1000.0)
    mean_p_adv_cairo, n_p_cairo = compute_pct_advantage(cairo_prove, z_prove_cairo)
    mean_v_adv_cairo, n_v_cairo = compute_pct_advantage(cairo_verify, z_verify_cairo)
    mean_p_ratio_cairo, _ = compute_mean_ratio(cairo_prove, z_prove_cairo)
    mean_v_ratio_cairo, _ = compute_mean_ratio(cairo_verify, z_verify_cairo)

    # Print concise summary with multiplicative factors
    def ratio_str(r, n):
        return f"{r:.2f}x" if n > 0 and not np.isnan(r) else "n/a"

    print("=== Average advantages (positive => Zinnia is faster / smaller) ===")
    print(f"Halo2: Proving {mean_p_adv_halo2:+.1f}% (n={n_p_halo2}) ~{ratio_str(mean_p_ratio_halo2, n_p_halo2)} faster, "
          f"Verifying {mean_v_adv_halo2:+.1f}% (n={n_v_halo2}) ~{ratio_str(mean_v_ratio_halo2, n_v_halo2)} faster, "
          f"Constraints {mean_c_adv_halo2:+.1f}% (n={n_c_halo2}) ~{ratio_str(mean_c_ratio_halo2, n_c_halo2)}× smaller")
    print(f"Noir:  Proving {mean_p_adv_noir:+.1f}% (n={n_p_noir}) ~{ratio_str(mean_p_ratio_noir, n_p_noir)} faster, "
          f"Verifying {mean_v_adv_noir:+.1f}% (n={n_v_noir}) ~{ratio_str(mean_v_ratio_noir, n_v_noir)} faster, "
          f"Constraints {mean_c_adv_noir:+.1f}% (n={n_c_noir}) ~{ratio_str(mean_c_ratio_noir, n_c_noir)}× smaller")
    print(f"SP1 (STARK): Proving {mean_p_adv_sp1_stark:+.1f}% (n={n_p_sp1_stark}) ~{ratio_str(mean_p_ratio_sp1_stark, n_p_sp1_stark)} faster, "
          f"Verifying {mean_v_adv_sp1_stark:+.1f}% (n={n_v_sp1_stark}) ~{ratio_str(mean_v_ratio_sp1_stark, n_v_sp1_stark)} faster")
    print(f"SP1 (SNARK): Proving {mean_p_adv_sp1_snark:+.1f}% (n={n_p_sp1_snark}) ~{ratio_str(mean_p_ratio_sp1_snark, n_p_sp1_snark)} faster, "
          f"Verifying {mean_v_adv_sp1_snark:+.1f}% (n={n_v_sp1_snark}) ~{ratio_str(mean_v_ratio_sp1_snark, n_v_sp1_snark)} faster")
    print(f"Risc0: Proving {mean_p_adv_risc0:+.1f}% (n={n_p_risc0}) ~{ratio_str(mean_p_ratio_risc0, n_p_risc0)} faster, "
          f"Verifying {mean_v_adv_risc0:+.1f}% (n={n_v_risc0}) ~{ratio_str(mean_v_ratio_risc0, n_v_risc0)} faster")
    print(f"Cairo: Proving {mean_p_adv_cairo:+.1f}% (n={n_p_cairo}) ~{ratio_str(mean_p_ratio_cairo, n_p_cairo)} faster, "
          f"Verifying {mean_v_adv_cairo:+.1f}% (n={n_v_cairo}) ~{ratio_str(mean_v_ratio_cairo, n_v_cairo)} faster")
    print("===============================================================")


def main():
    # plot_evaluation_results()
    plot_performance_overviews()
    plot_ablation_study()
    plot_performance_heatmap()
    plot_compile_time_scalability()
    # Print summary advantages
    print_average_advantages()


if __name__ == "__main__":
    main()
