import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import rcParams


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


MLALGO = [
    "neuron",
    "kmeans",
    "linear_regression",
    "mlp",
    "svm",
    "logistic",
    "decision_tree",
    "bayes"
]
LEETCODE_ARRAY = [
    "p204",
    "p832"
]
LEETCODE_DP = [
    "p740",
    "p1137"
]
LEETCODE_GRAPH = [
    "p3112",
    "p997"
]
LEETCODE_MATH = [
    "p492",
    "p2125"
]
LEETCODE_MATRIX = [
    "p73",
    "p2133"
]
DS1000 = [
    'case295', 'case296', 'case297', 'case298', 'case299', 'case300', 'case301', 'case302', 'case303',
    'case304', 'case309', 'case310', 'case313', 'case318', 'case319', 'case322', 'case323', 'case324',
    'case329', 'case330', 'case334', 'case335', 'case336', 'case337', 'case338', 'case339', 'case353',
    'case354', 'case360', 'case368', 'case369', 'case370', 'case373', 'case374', 'case375', 'case385',
    'case387', 'case388', 'case389', 'case390', 'case391', 'case392', 'case393', 'case406', 'case407',
    'case408', 'case409', 'case414', 'case415', 'case416', 'case417', 'case418', 'case419', 'case420',
    'case428', 'case429', 'case430', 'case431', 'case433', 'case434', 'case435', 'case436', 'case437',
    'case438', 'case440', 'case441', 'case452', 'case453', 'case459', 'case480', 'case501', 'case507',
    'case510'
]
CRYPT = [
    "poseidon",
    "ecc",
    "elgamal",
    "mimc",
    "merkle"
]

DATASETS = {
    "crypt": CRYPT,
    "ds1000": DS1000,
    "leetcode_array": LEETCODE_ARRAY,
    "leetcode_dp": LEETCODE_DP,
    "leetcode_graph": LEETCODE_GRAPH,
    "leetcode_math": LEETCODE_MATH,
    "leetcode_matrix": LEETCODE_MATRIX,
    "mlalgo": MLALGO,
}


def compute_cyclomatic_complexity(lines) -> int:
    complexity = 0
    decision_keywords = ['if', 'for', 'while', '?', 'match', 'case', 'elif', 'else if', 'switch', 'else']
    for line in lines:
        stripped_line = line.strip()
        while len(stripped_line) > 0:
            has_match = False
            for keyword in decision_keywords:
                try:
                    idx = stripped_line.index(keyword)
                    complexity += 1
                    stripped_line = stripped_line[idx + len(keyword):].strip()
                    has_match = True
                    break
                except ValueError:
                    continue
            if not has_match:
                break
    return complexity


# -----------------------------
# Count functions for each DSL
# -----------------------------
def count_zinnia(dataset: str, problem: str):
    path = os.path.join('../benchmarking', dataset, problem, 'sol.py')
    if not os.path.exists(path):
        return np.nan, np.nan

    with open(path, 'r') as f:
        lines = f.read().split('\n')

    # Exclude everything before the first `@zk_circuit` or `@zk_chip`
    try:
        start_idx = next(i for i, l in enumerate(lines) if '@zk_c' in l)
    except StopIteration:
        start_idx = 0

    # Exclude everything after `if __name__ ==`
    try:
        end_idx = next(i for i, l in enumerate(lines) if 'if __name__' in l)
    except StopIteration:
        end_idx = len(lines)

    # Slice relevant lines
    lines = lines[start_idx:end_idx]

    # Filter out comments, imports, and empty lines
    lines = [l for l in lines if l.strip() != '' and not l.startswith(('#', 'from', 'import'))]

    return len(lines), compute_cyclomatic_complexity(lines)


def count_halo2(dataset: str, problem: str):
    path = os.path.join('../benchmarking', dataset, problem, 'sol.rs')
    if not os.path.exists(path):
        return np.nan, np.nan
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    try:
        idx1 = lines.index("    const PRECISION: u32 = 63;")
        idx2 = lines.index("fn main() {")
        lines = lines[idx1:idx2]
    except ValueError:
        pass
    lines = [l for l in lines if l.strip() != '' and not l.startswith('//')]
    return len(lines), compute_cyclomatic_complexity(lines)


def count_sp1(dataset: str, problem: str):
    path = os.path.join('../benchmarking', dataset, problem, 'sp1.prog.rs')
    if not os.path.exists(path):
        return np.nan, np.nan
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    try:
        idx = lines.index("// source start", 0)
        lines = lines[idx:]
    except ValueError:
        try:
            idx = lines.index("pub fn main() {")
            lines = lines[idx:]
        except ValueError:
            pass
    lines = [l for l in lines if l.strip() != '' and not l.startswith('//')]
    return len(lines), compute_cyclomatic_complexity(lines)


def count_risc0(dataset: str, problem: str):
    path = os.path.join('../benchmarking', dataset, problem, 'risc0.prog.rs')
    if not os.path.exists(path):
        return np.nan, np.nan
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    try:
        idx = lines.index("// source start", 0)
        lines = lines[idx:]
    except ValueError:
        try:
            idx = lines.index("fn main() {")
            lines = lines[idx:]
        except ValueError:
            pass
    lines = [l for l in lines if l.strip() != '' and not l.startswith('//')]
    return len(lines), compute_cyclomatic_complexity(lines)


def count_cairo(dataset: str, problem: str):
    path = os.path.join('../benchmarking', dataset, problem, 'main.cairo')
    if not os.path.exists(path):
        return np.nan, np.nan
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [l for l in lines if l.strip() != '' and not l.startswith('//')]
    return len(lines), compute_cyclomatic_complexity(lines)


def count_noir(dataset: str, problem: str):
    path = os.path.join('../benchmarking', dataset, problem, 'main.nr')
    if not os.path.exists(path):
        return np.nan, np.nan
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [l for l in lines if l.strip() != '' and not l.startswith('//')]
    return len(lines), compute_cyclomatic_complexity(lines)


# -----------------------------
# Plotting LoC landscape
# -----------------------------
def plot_loc_landscape():
    rcParams["axes.xmargin"] = 0.02
    plt.rc('font', family='monospace')

    # Collect dataset/problem pairs
    all_problems = []
    for dataset in DATASETS.keys():
        dpath = os.path.join('../benchmarking', dataset)
        if not os.path.exists(dpath):
            continue
        for problem in sorted(DATASETS[dataset]):
            all_problems.append((dataset, problem))
    names = [NAME_MAPPING[f'{d}::{p}'] for d, p in all_problems]

    # Gather metrics
    zinnia_loc, zinnia_complexity = [], []
    halo2_loc, halo2_complexity = [], []
    noir_loc, noir_complexity = [], []
    risc0_loc, risc0_complexity = [], []
    sp1_loc, sp1_complexity = [], []
    cairo_loc, cairo_complexity = [], []

    for dataset, problem in all_problems:
        a, b = count_zinnia(dataset, problem)
        zinnia_loc.append(a)
        zinnia_complexity.append(b)
        a, b = count_halo2(dataset, problem)
        halo2_loc.append(a)
        halo2_complexity.append(b)
        a, b = count_noir(dataset, problem)
        noir_loc.append(a)
        noir_complexity.append(b)
        a, b = count_risc0(dataset, problem)
        risc0_loc.append(a)
        risc0_complexity.append(b)
        a, b = count_sp1(dataset, problem)
        sp1_loc.append(a)
        sp1_complexity.append(b)
        a, b = count_cairo(dataset, problem)
        cairo_loc.append(a)
        cairo_complexity.append(b)

    # Convert to numpy arrays
    zinnia_loc = np.asarray(zinnia_loc, dtype=float)
    halo2_loc = np.asarray(halo2_loc, dtype=float)
    noir_loc = np.asarray(noir_loc, dtype=float)
    risc0_loc = np.asarray(risc0_loc, dtype=float)
    sp1_loc = np.asarray(sp1_loc, dtype=float)
    cairo_loc = np.asarray(cairo_loc, dtype=float)
    zinnia_complexity = np.asarray(zinnia_complexity, dtype=float)
    halo2_complexity = np.asarray(halo2_complexity, dtype=float)
    noir_complexity = np.asarray(noir_complexity, dtype=float)
    risc0_complexity = np.asarray(risc0_complexity, dtype=float)
    sp1_complexity = np.asarray(sp1_complexity, dtype=float)
    cairo_complexity = np.asarray(cairo_complexity, dtype=float)

    # =============================
    # NEW: Compute average advantages
    # =============================
    baselines = {
        'Halo2': (halo2_loc, halo2_complexity),
        'Noir': (noir_loc, noir_complexity),
        'Risc0': (risc0_loc, risc0_complexity),
        'SP1': (sp1_loc, sp1_complexity),
        'Cairo': (cairo_loc, cairo_complexity)
    }

    print("\n=== Average Advantage of Zinnia over Other Baselines ===")
    print(f"{'Baseline':<10} | {'ΔLoC':>10} | {'ΔLoC %':>10} | {'ΔComplex':>12} | {'ΔComplex %':>12}")
    print("-" * 62)

    for name, (loc_arr, comp_arr) in baselines.items():
        # Filter NaNs
        mask = (~np.isnan(zinnia_loc)) & (~np.isnan(loc_arr))
        loc_diff = np.nanmean(loc_arr[mask] / zinnia_loc[mask])
        loc_pct = np.nanmean((loc_arr[mask] - zinnia_loc[mask]) / loc_arr[mask] * 100)

        mask = (~np.isnan(zinnia_complexity)) & (~np.isnan(comp_arr))
        comp_diff = np.nanmean(comp_arr[mask] / zinnia_complexity[mask])
        comp_pct = np.nanmean((comp_arr[mask] - zinnia_complexity[mask]) / comp_arr[mask] * 100)

        print(f"{name:<10} | {loc_diff:>10.2f} | {loc_pct:>9.2f}% | {comp_diff:>12.2f} | {comp_pct:>11.2f}%")

    print("-" * 62)
    print("Positive values → Zinnia uses fewer LoC / lower complexity on average\n")

    # =============================
    # Existing plotting code follows
    # =============================

    colors = list(reversed(['mediumseagreen', 'purple', 'azure', 'lightcoral', 'orange', 'gray']))
    labels = list(reversed(['Zinnia', 'Halo2', 'Noir', 'Risc0', 'SP1', 'Cairo']))
    loc_series = list(reversed([zinnia_loc, halo2_loc, noir_loc, risc0_loc, sp1_loc, cairo_loc]))
    complexity_series = list(reversed([zinnia_complexity, halo2_complexity, noir_complexity, risc0_complexity, sp1_complexity, cairo_complexity]))

    x = np.arange(len(names))
    marker_size = 40

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True, height_ratios=[1, 1])
    for ser, col, lbl in zip(loc_series, colors, labels):
        ax1.scatter(x, ser, marker='o', s=marker_size, color=col, edgecolors='k', linewidths=0.5, alpha=0.9, label=lbl)
    ax1.set_ylabel('LoC', fontdict={'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 12})
    ax1.set_xlim(-0.5, len(names) - 0.5)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

    for ser, col, lbl in zip(complexity_series, colors, labels):
        ax2.scatter(x, ser, marker='s', s=marker_size, color=col, edgecolors='k', linewidths=0.5, alpha=0.9)
    ax2.set_ylabel('Cyclomatic Complexity', fontdict={'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 10})
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=90)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

    fig.legend(loc='upper center', ncol=6, prop={'size': 8}, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    fig.savefig('loc-complexity-landscape.pdf', dpi=300)

    # ---- Aggregate distribution (box or violin) ----
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(5, 3.4), sharey=False)

    data_loc = [ser[~np.isnan(ser)] for ser in loc_series]
    data_complexity = [ser[~np.isnan(ser)] for ser in complexity_series]

    # Boxplot version (swap with violinplot if desired)
    data_loc = list(reversed(data_loc))
    data_complexity = list(reversed(data_complexity))
    labels = list(reversed(labels))
    ax3.boxplot(data_loc, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightgray', color='k', alpha=0.6))
    ax3.tick_params(axis='x', labelrotation=30, labelfontfamily='Times New Roman')
    ax3.set_ylabel("LoC Distribution", fontdict={'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 12})
    ax3.grid(True, axis='y', linestyle='--', alpha=0.3)

    ax4.boxplot(data_complexity, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightgray', color='k', alpha=0.6))
    ax4.tick_params(axis='x', labelrotation=30, labelfontfamily='Times New Roman')
    ax4.set_ylabel("Cyclomatic Complexity Distribution", fontdict={'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 12})
    ax4.grid(True, axis='y', linestyle='--', alpha=0.3)

    fig2.tight_layout()
    plt.show()
    fig2.savefig('loc-complexity-distribution.pdf', dpi=300)



def main():
    plot_loc_landscape()


if __name__ == "__main__":
    main()
