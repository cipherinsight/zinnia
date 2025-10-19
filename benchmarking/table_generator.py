import json
import numpy as np

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
    'leetcode_array::p204':      'LC\\#0204',
    'leetcode_array::p832':      'LC\\#0832',
    'leetcode_dp::p740':         'LC\\#0740',
    'leetcode_dp::p1137':        'LC\\#1137',
    'leetcode_graph::p3112':     'LC\\#3112',
    'leetcode_graph::p997':      'LC\\#0997',
    'leetcode_math::p492':       'LC\\#0492',
    'leetcode_math::p2125':      'LC\\#2125',
    'leetcode_matrix::p73':      'LC\\#0073',
    'leetcode_matrix::p2133':    'LC\\#2133',
}
DS1000 = [
    'case295', 'case296', 'case297', 'case298', 'case299', 'case300', 'case301', 'case302', 'case303', 'case304', 'case309', 'case310', 'case313', 'case318', 'case319', 'case322', 'case323', 'case324', 'case329', 'case330', 'case334', 'case335', 'case336', 'case337', 'case338', 'case339', 'case353', 'case354', 'case360', 'case368', 'case369', 'case370', 'case373', 'case374', 'case375', 'case385', 'case387', 'case388', 'case389', 'case390', 'case391', 'case392', 'case393', 'case406', 'case407', 'case408', 'case409', 'case414', 'case415', 'case416', 'case417', 'case418', 'case419', 'case420', 'case428', 'case429', 'case430', 'case431', 'case433', 'case434', 'case435', 'case436', 'case437', 'case438', 'case440', 'case441', 'case452', 'case453', 'case459', 'case480', 'case501', 'case507', 'case510'
]
for key in DS1000:
    NAME_MAPPING[f'ds1000::{key}'] = f'DS\\#0{key[4:]}'


def export_zkvm_time_tables(metric='both'):
    # === Load JSONs ===
    def _load(path):
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[warn] missing {path}")
            return {}
    zinnia = _load('results.json')
    sp1 = _load('results-sp1.json')
    risc0 = _load('results-risc0.json')
    cairo = _load('results-cairo.json')

    # === Utility ===
    fmt = lambda v: 'N/A' if v in [None,0] or np.isnan(v) else f'{v:.2f}'
    chunk = lambda lst,n: [lst[i:i+n] for i in range(0,len(lst),n)]

    # === Table generator ===
    def make_table(kind):
        sorted_keys = sorted(zinnia.keys(), key=lambda k: NAME_MAPPING.get(k,k))
        names = [NAME_MAPPING.get(k,k) for k in sorted_keys]
        rows = {lab: [] for lab in ['\\tool','RISC0','SP1','SP1$^{*}$','Cairo']}
        for key in sorted_keys:
            z = zinnia[key]; s=sp1.get(key,{}); r=risc0.get(key,{}); c=cairo.get(key,{})
            if kind=='proving':
                rows['\\tool'].append(fmt(z['zinnia']['proving_time']))
                rows['RISC0'].append(fmt(r.get('stark_proving_time')))
                rows['SP1'].append(fmt(s.get('stark_proving_time')))
                rows['SP1$^{*}$'].append(fmt(s.get('snark_proving_time')))
                rows['Cairo'].append(fmt(c.get('stark_proving_time')))
            else:
                rows['\\tool'].append(fmt(z['zinnia']['verify_time']*1000))
                rows['RISC0'].append(fmt(r.get('stark_verify_time',0)*1000))
                rows['SP1'].append(fmt(s.get('stark_verify_time',0)*1000))
                rows['SP1$^{*}$'].append(fmt(s.get('snark_verify_time',0)*1000))
                rows['Cairo'].append(fmt(c.get('stark_verify_time',0)*1000))
        # break into ≤10-col blocks
        BLOCK_WIDTH = 12
        blocks=[]
        for i in range(0,len(names),BLOCK_WIDTH):
            subnames = names[i:i+BLOCK_WIDTH]
            part=[" & ".join(['']+subnames)+r" \\ \midrule"]
            for lab in rows:
                part.append(lab+" & "+ " & ".join(rows[lab][i:i+BLOCK_WIDTH])+r" \\ ")
            blocks.append("\n        ".join(part))
        cap = ("Detailed zkVMs' Proving Time (s) in RQ2~(SP1$^*$ denotes SP1's SNARK prover)."
               if kind=='proving' else
               "Detailed zkVMs' Verifying Time (ms) in RQ2.")
        label = ("tab:zkvm-proving-time-eval" if kind=='proving' else
                 "tab:zkvm-verifying-time-eval")
        latex = [r"\begin{table*}[ht]",
                 r"    \centering",
                 f"    \\caption{{{cap}}}",
                 f"    \\label{{{label}}}",
                 r"    \resizebox{\textwidth}{!}{",
                 r"    \begin{tabular}{c|" + "c"*BLOCK_WIDTH + "}",
                 r"        \toprule",
                 "        \\midrule\n        ".join(blocks),
                 r"        \bottomrule",
                 r"    \end{tabular}}",
                 r"\end{table*}"]
        print("\n".join(latex))
        print("\n\n")

    # === Output ===
    if metric in ['proving','both']: make_table('proving')
    if metric in ['verifying','both']: make_table('verifying')


def export_circuit_size_tables():
    """
    Generate two LaTeX tables:
      (1) Circuit size comparison on PLONK protocol (Halo2 vs Zinnia)
      (2) Circuit size comparison on ULTRAHONK protocol (Noir vs Zinnia)
    Assumes global NAME_MAPPING is defined.
    """

    # --- Load JSON data ---
    with open('results.json', 'r') as f:
        results_dict = json.load(f)
    with open('results-noir.json', 'r') as f:
        noir_results_dict = json.load(f)

    # --- Utility helpers ---
    fmt = lambda v: 'N/A' if v in [None, 0] or np.isnan(v) else f'{int(v)}'
    chunk = lambda lst, n: [lst[i:i + n] for i in range(0, len(lst), n)]

    # --- Sorted keys ---
    sorted_keys = sorted(results_dict.keys(), key=lambda k: NAME_MAPPING.get(k, k))
    names = [NAME_MAPPING.get(k, k) for k in sorted_keys]

    # --- Gather data for PLONK (Halo2 baseline) ---
    zinnia_vals, halo2_vals = [], []
    for key in sorted_keys:
        val = results_dict[key]
        zinnia_vals.append(val['zinnia']['advice_cells'])
        halo2_vals.append(val['halo2']['advice_cells'])

    # --- Gather data for ULTRAHONK (Noir baseline) ---
    zinnia_noir_vals, noir_vals = [], []
    for key in sorted_keys:
        if key in noir_results_dict:
            nv = noir_results_dict[key]
            zinnia_noir_vals.append(nv['ours_on_noir']['total_gates'])
            noir_vals.append(nv['baseline_on_noir']['total_gates'])
        else:
            zinnia_noir_vals.append(np.nan)
            noir_vals.append(np.nan)

    # --- Helper to build one table ---
    BLOCK_WIDTH = 12
    def build_table(title, label, tool_name, base_name, tool_data, base_data):
        blocks = []
        for i in range(0, len(names), BLOCK_WIDTH):
            subnames = names[i:i + BLOCK_WIDTH]
            part = [" & ".join([''] + subnames) + r" \\ \midrule"]
            part.append(tool_name + " & " + " & ".join(fmt(v) for v in tool_data[i:i + BLOCK_WIDTH]) + r" \\ ")
            part.append(base_name + " & " + " & ".join(fmt(v) for v in base_data[i:i + BLOCK_WIDTH]) + r" \\ ")
            blocks.append("\n        ".join(part))

        latex = [
            r"\begin{table*}[ht]",
            r"    \centering",
            f"    \\caption{{{title}}}",
            f"    \\label{{{label}}}",
            r"    \resizebox{\textwidth}{!}{",
            r"    \begin{tabular}{c|" + "c"*BLOCK_WIDTH + "}",
            r"        \toprule",
            "        \\midrule\n        ".join(blocks),
            r"        \bottomrule",
            r"    \end{tabular}}",
            r"\end{table*}",
        ]
        print("\n".join(latex))
        print("\n\n")

    # --- Emit both tables ---
    build_table(
        "Detailed Circuit Size (No. of Polynomial Constraints) in RQ2 on PLONK Protocol.",
        "tab:plonk-circuit-size-eval",
        r"\tool",
        "Halo2",
        zinnia_vals,
        halo2_vals,
    )

    build_table(
        "Detailed Circuit Size (No. of Polynomial Constraints) in RQ2 on ULTRAHONK Protocol.",
        "tab:ultrahonk-circuit-size-eval",
        r"\tool",
        "Noir",
        zinnia_noir_vals,
        noir_vals,
    )


def export_ablation_circuit_increase_table():
    """
    Generate LaTeX table summarizing the circuit size increase
    (no. of polynomial constraints) in ablation variants V1–V4.
    Uses NAME_MAPPING defined globally.
    """

    # --- Load JSONs ---
    with open('results.json', 'r') as f:
        zinnia_results = json.load(f)
    with open('results-ablation-1.json', 'r') as f:
        ab1 = json.load(f)
    with open('results-ablation-2.json', 'r') as f:
        ab2 = json.load(f)
    with open('results-ablation-3.json', 'r') as f:
        ab3 = json.load(f)
    with open('results-ablation-4.json', 'r') as f:
        ab4 = json.load(f)

    # --- Helper ---
    def fmt(v):
        return '0' if v == 0 else ('N/A' if v is None or np.isnan(v) else f'{int(v)}')

    # --- Sort keys ---
    sorted_keys = sorted(zinnia_results.keys(), key=lambda k: NAME_MAPPING.get(k, k))
    names = [NAME_MAPPING.get(k, k) for k in sorted_keys]

    # --- Compute increases ---
    inc_v1, inc_v2, inc_v3, inc_v4 = [], [], [], []
    for key in sorted_keys:
        base = zinnia_results[key]['zinnia']['advice_cells']
        def get_increase(abset, key):
            try:
                return abset[key]['zinnia']['advice_cells'] - base
            except Exception:
                return np.nan
        inc_v1.append(get_increase(ab1, key))
        inc_v2.append(get_increase(ab2, key))
        inc_v3.append(get_increase(ab3, key))
        inc_v4.append(get_increase(ab4, key))

    # --- Build LaTeX blocks (≤BLOCK_WIDTH columns) ---
    BLOCK_WIDTH = 12
    def build_table_block(start, end):
        subnames = names[start:end]
        part = [" & ".join([''] + subnames) + r" \\ \midrule"]
        for label, data in zip(
            ["V1", "V2", "V3", "V4"],
            [inc_v1, inc_v2, inc_v3, inc_v4]
        ):
            vals = " & ".join(fmt(v) for v in data[start:end])
            part.append(f"{label} & {vals} \\\\")
        return "\n        ".join(part)

    blocks = [build_table_block(i, i + BLOCK_WIDTH) for i in range(0, len(names), BLOCK_WIDTH)]

    # --- Assemble LaTeX ---
    latex = [
        r"\begin{table*}[ht]",
        r"    \centering",
        r"    \caption{Detailed Circuit Size Increase (No. of Polynomial Constraints) in the Ablation Study in RQ3.}",
        r"    \label{tab:ablation-study-eval}",
        r"    \resizebox{\textwidth}{!}{",
        r"    \begin{tabular}{c|" + "c"*BLOCK_WIDTH + "}",
        r"        \toprule",
        "        \\midrule\n        ".join(blocks),
        r"        \bottomrule",
        r"    \end{tabular}}",
        r"\end{table*}"
    ]

    print("\n".join(latex))


import json
import numpy as np

def export_proof_size_tables():
    """
    Generate LaTeX tables for SNARK proof size comparison:
      (1) PLONK protocol (Halo2 vs Zinnia)
      (2) ULTRAHONK protocol (Noir vs Zinnia)
    Assumes global NAME_MAPPING is already defined.
    """

    # --- Load data ---
    with open('results.json', 'r') as f:
        results_dict = json.load(f)
    with open('results-noir.json', 'r') as f:
        noir_results_dict = json.load(f)

    # --- Utilities ---
    fmt = lambda v: 'N/A' if v in [None, 0] or np.isnan(v) else f'{int(v)}'
    sorted_keys = sorted(results_dict.keys(), key=lambda k: NAME_MAPPING.get(k, k))
    names = [NAME_MAPPING.get(k, k) for k in sorted_keys]

    # --- Extract proof sizes (PLONK) ---
    zinnia_sizes, halo2_sizes = [], []
    for key in sorted_keys:
        val = results_dict[key]
        zinnia_sizes.append(val['zinnia'].get('snark_size', np.nan))
        halo2_sizes.append(val['halo2'].get('snark_size', np.nan))

    # --- Extract proof sizes (ULTRAHONK) ---
    zinnia_noir_sizes, noir_sizes = [], []
    for key in sorted_keys:
        if key in noir_results_dict:
            nv = noir_results_dict[key]
            zinnia_noir_sizes.append(nv['ours_on_noir'].get('circuit_size', np.nan))
            noir_sizes.append(nv['baseline_on_noir'].get('circuit_size', np.nan))
        else:
            zinnia_noir_sizes.append(np.nan)
            noir_sizes.append(np.nan)

    # --- Helper to build each LaTeX table ---
    BLOCK_WIDTH = 12
    def build_table(title, label, tool_name, base_name, tool_data, base_data):
        blocks = []
        for i in range(0, len(names), BLOCK_WIDTH):
            subnames = names[i:i + BLOCK_WIDTH]
            part = [" & ".join([''] + subnames) + r" \\ \midrule"]
            part.append(tool_name + " & " + " & ".join(fmt(v) for v in tool_data[i:i + BLOCK_WIDTH]) + r" \\ ")
            part.append(base_name + " & " + " & ".join(fmt(v) for v in base_data[i:i + BLOCK_WIDTH]) + r" \\ ")
            blocks.append("\n        ".join(part))

        latex = [
            r"\begin{table*}[ht]",
            r"    \centering",
            f"    \\caption{{{title}}}",
            f"    \\label{{{label}}}",
            r"    \resizebox{\textwidth}{!}{",
            r"    \begin{tabular}{c|" + "c"*BLOCK_WIDTH + "}",
            r"        \toprule",
            "        \\midrule\n        ".join(blocks),
            r"        \bottomrule",
            r"    \end{tabular}}",
            r"\end{table*}"
        ]
        print("\n".join(latex))
        print("\n\n")

    # --- Emit both tables ---
    build_table(
        "SNARK Proof Size (in Bytes) Comparison Results on PLONK.",
        "tab:proof-size-eval-plonk",
        r"\tool",
        "Halo2",
        zinnia_sizes,
        halo2_sizes
    )

    build_table(
        "SNARK Proof Size (in Bytes) Comparison Results on ULTRAHONK.",
        "tab:proof-size-eval-ultrahonk",
        r"\tool",
        "Noir",
        zinnia_noir_sizes,
        noir_sizes
    )


def export_benchmark_overview():
    """
    Print a LaTeX table with four major groups:
    Cryptography | id | description
    DS1000       | id id id ...
    Leetcode     | id | description
    Machine Learning | id | description
    """
    # Assuming NAME_MAPPING exists globally, and you have a list of all keys (e.g. from results.json).
    import json
    with open('results.json') as f:
        res = json.load(f)
    # Also possibly results-noir for noir-only ones, but we focus on keys in results.json

    crypt = sorted([k for k in res if k.startswith("crypt::")], key=lambda k: NAME_MAPPING.get(k, k))
    ds = sorted([k for k in res if k.startswith("ds1000::")], key=lambda k: NAME_MAPPING.get(k, k))
    ml = sorted([k for k in res if k.startswith("mlalgo::")], key=lambda k: NAME_MAPPING.get(k, k))
    leet = sorted([k for k in res if k.startswith("leetcode")], key=lambda k: NAME_MAPPING.get(k, k))

    # small mapping for crypt descriptions
    crypt_desc = {
        'crypt::ecc': 'Elliptic-curve based operations',
        'crypt::poseidon': 'Poseidon hash function (SNARK-friendly) ' + '(efficient circuit hash)',
        'crypt::elgamal': 'ElGamal public-key encryption',
        'crypt::merkle': 'Merkle tree hashing / authentication',
        'crypt::mimc': 'MiMC cipher / hash (low multiplicative depth)',
    }
    # small mapping for leetcode descriptions (filling known ones)
    leet_desc = {
        'leetcode_array::p204': 'Count primes less than n',  # from LeetCode #204 :contentReference[oaicite:0]{index=0}
        # placeholders or guesses:
        'leetcode_array::p832': 'Flip and invert image (placeholder)',
        'leetcode_dp::p740': 'Delete and Earn (DP-style) (placeholder)',
        'leetcode_dp::p1137': 'Nth Tribonacci (DP) (placeholder)',
        'leetcode_graph::p3112': 'Graph traversal problem (placeholder)',
        'leetcode_graph::p997': 'Find the Town Judge (placeholder)',
        'leetcode_math::p492': 'Construct rectangle from area (placeholder)',
        'leetcode_math::p2125': 'Ways to split array (placeholder)',
        'leetcode_matrix::p73': 'Set matrix zeroes (placeholder)',
        'leetcode_matrix::p2133': 'Longest palindrome by concatenation (placeholder)',
    }
    # mapping for ml descriptions
    ml_desc = {
        'mlalgo::neuron': 'Single neuron forward / activation test',
        'mlalgo::kmeans': 'K-means clustering',
        'mlalgo::linear_regression': 'Linear regression model training/inference',
        'mlalgo::svm': 'Support vector machine classification',
        'mlalgo::logistic': 'Logistic regression',
        'mlalgo::decision_tree': 'Decision tree classifier',
        'mlalgo::bayes': 'Naive Bayes classification',
        'mlalgo::mlp': 'Multi-layer perceptron (neural network)',
    }

    # LaTeX output
    print(r"\begin{table*}[ht]")
    print(r"  \centering")
    print(r"  \caption{Benchmark Task Overview}")
    print(r"  \label{tab:benchmark-overview}")
    # We'll make columns: first major group label spanning multiple rows, then id, then description (for crypt / leet / ml) or list of ids (for ds)
    # Use e.g. \multirow for spanning
    print(r"  \begin{tabular}{l l l}")
    print(r"    \toprule")
    print(r"    \textbf{Category} & \textbf{ID(s)} & \textbf{Description} \\")
    print(r"    \midrule")

    # Cryptography
    n_crypt = len(crypt)
    print(r"    \multirow{" + f"{n_crypt}" + r"}{*}{Cryptography} & " + NAME_MAPPING.get(crypt[0], crypt[
        0]) + " & " + crypt_desc.get(crypt[0], "") + r" \\")
    for k in crypt[1:]:
        print(r"    & " + NAME_MAPPING.get(k, k) + " & " + crypt_desc.get(k, "") + r" \\")
    print(r"    \midrule")

    # DS1000 (just list ids in one row)
    ds_ids = [NAME_MAPPING.get(k, k) for k in ds]
    # combine as space-separated (or comma) list
    ds_ids_str = " \\quad ".join(ds_ids)
    print(r"    DS1000 & " + ds_ids_str + r" & --- \\")
    print(r"    \midrule")

    # LeetCode
    n_leet = len(leet)
    print(r"    \multirow{" + f"{n_leet}" + r"}{*}{LeetCode} & " + NAME_MAPPING.get(leet[0],
                                                                                    leet[0]) + " & " + leet_desc.get(
        leet[0], "") + r" \\")
    for k in leet[1:]:
        print(r"    & " + NAME_MAPPING.get(k, k) + " & " + leet_desc.get(k, "") + r" \\")
    print(r"    \midrule")

    # Machine Learning
    n_ml = len(ml)
    print(r"    \multirow{" + f"{n_ml}" + r"}{*}{Machine Learning} & " + NAME_MAPPING.get(ml[0],
                                                                                          ml[0]) + " & " + ml_desc.get(
        ml[0], "") + r" \\")
    for k in ml[1:]:
        print(r"    & " + NAME_MAPPING.get(k, k) + " & " + ml_desc.get(k, "") + r" \\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table*}")

if __name__ == '__main__':
    # export_zkvm_time_tables('both')
    # export_circuit_size_tables()
    # export_ablation_circuit_increase_table()
    # export_proof_size_tables()
    export_benchmark_overview()
