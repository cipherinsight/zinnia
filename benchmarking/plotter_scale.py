# Build the requested column-style scatter: x = task, y = advice_cells (log scale).
# Each column shows 2 (backends) × 5 (scales) points. We restrict to tasks present
# in both 1× data and scale data so all 5 scales exist. Zinnia vs Halo2 share color
# and use different markers; scale is encoded by marker size. A small horizontal
# jitter separates the two backends within a column.

import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

with open('./results-scale.json', 'r') as f:
    scale_data = json.loads(f.read())

with open('./results.json', 'r') as f:
    one_x = json.loads(f.read())


NAME_MAPPING = {
    'crypt::ecc':                'CRY···ECC',
    'crypt::poseidon':           'CRY··Hash',
    'ds1000::case296':           'DS···#296',
    'ds1000::case309':           'DS···#309',
    'ds1000::case330':           'DS···#330',
    'ds1000::case360':           'DS···#360',
    'ds1000::case387':           'DS···#387',
    'ds1000::case418':           'DS···#418',
    'ds1000::case453':           'DS···#453',
    'ds1000::case459':           'DS···#459',
    'ds1000::case501':           'DS···#501',
    'ds1000::case510':           'DS···#510',
    'mlalgo::neuron':            'ML·Neuron',
    'mlalgo::kmeans':            'ML·KMeans',
    'mlalgo::linear_regression': 'ML·LinReg',
}

plt.rc('font', family='monospace', )
# plt.rc('text', usetex=True)
title_font = {'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 12}

# Restrict to tasks that have scale data so we can show 5 scales (1,2,4,8,16)
tasks = [t for t in one_x.keys() if t in scale_data]

# Sort tasks for consistent order
tasks = sorted(tasks)

# Prepare plotting
fig, ax = plt.subplots(figsize=(6, 4))

colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [f"C{i}" for i in range(10)])
backend_markers = {"zinnia": "o", "halo2": "s"}
backend_offsets = {"zinnia": -0.17, "halo2": 0.17}
scale_sizes = {1: 25, 2: 45, 4: 65, 8: 85, 16: 105}

x_positions = np.arange(len(tasks))

for i, task in enumerate(tasks):
    color = colors[i % len(colors)]
    # gather values for scales
    for backend in ["zinnia", "halo2"]:
        xs = []
        ys = []
        ss = []
        for scale in [1, 2, 4, 8, 16]:
            if scale == 1:
                if backend in one_x[task]:
                    y = one_x[task][backend]["advice_cells"]
                else:
                    continue
            else:
                key = f"{backend}_{scale}"
                if key in scale_data[task]:
                    y = scale_data[task][key]["advice_cells"]
                else:
                    continue
            xs.append(x_positions[i] + backend_offsets[backend])
            ys.append(y)
            ss.append(scale_sizes[scale])
        if xs:
            ax.scatter(xs, ys, s=ss, marker=backend_markers[backend], color=color, edgecolor="black", linewidths=0.5, alpha=0.9)

# Aesthetics
ax.set_yscale("log")
ax.set_xlim(-0.6, len(tasks)-0.4)
ax.set_xticks(x_positions)
ax.set_xticklabels([NAME_MAPPING[t] for t in tasks], rotation=90, ha="center")
ax.set_ylabel("Arithmetic Circuit Size (Log Scale)", fontdict=title_font)

ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.6, alpha=0.7)

# Legends: one for backends (shape), one for scale (size)
backend_handles = [
    Line2D([0],[0], marker=backend_markers["zinnia"], color="gray", linestyle="None", markerfacecolor="gray", markeredgecolor="black", label="Zinnia", markersize=8),
    Line2D([0],[0], marker=backend_markers["halo2"], color="gray", linestyle="None", markerfacecolor="gray", markeredgecolor="black", label="Halo2", markersize=8)
]
scale_handles = [
    Line2D([0],[0], marker="o", linestyle="None", color="gray", markerfacecolor="gray", markeredgecolor="black", label="×1", markersize=np.sqrt(scale_sizes[1])),
    Line2D([0],[0], marker="o", linestyle="None", color="gray", markerfacecolor="gray", markeredgecolor="black", label="×2", markersize=np.sqrt(scale_sizes[2])),
    Line2D([0],[0], marker="o", linestyle="None", color="gray", markerfacecolor="gray", markeredgecolor="black", label="×4", markersize=np.sqrt(scale_sizes[4])),
    Line2D([0],[0], marker="o", linestyle="None", color="gray", markerfacecolor="gray", markeredgecolor="black", label="×8", markersize=np.sqrt(scale_sizes[8])),
    Line2D([0],[0], marker="o", linestyle="None", color="gray", markerfacecolor="gray", markeredgecolor="black", label="×16", markersize=np.sqrt(scale_sizes[16])),
]

legend1 = ax.legend(handles=backend_handles, loc="upper left", frameon=True)
legend2 = ax.legend(handles=scale_handles, loc="lower right", frameon=True, title="Scale")
ax.add_artist(legend1)

plt.tight_layout()
plt.show()
fig.savefig("scaling-experiment.pdf")
