import json

import matplotlib.pyplot as plt
import numpy as np



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
    'leetcode_array::p204':      'LC-Ary····#204',
    'leetcode_array::p832':      'LC-Ary····#832',
    'leetcode_dp::p740':         'LC-DP·····#740',
    'leetcode_dp::p1137':        'LC-DP····#1137',
    'leetcode_graph::p3112':     'LC-Graph·#3112',
    'leetcode_graph::p997':      'LC-Graph··#997',
    'leetcode_math::p492':       'LC-Math···#492',
    'leetcode_math::p2125':      'LC-Math··#2125',
    'leetcode_matrix::p73':      'LC-Mat·····#73',
    'leetcode_matrix::p2133':    'LC-Mat···#2133',
}


def plot_evaluation_results():
    with open('results.json', 'r') as f:
        results_dict = json.load(f)

    names = []
    acc_rates = []
    prove_time_rates = []
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
        acc_rates.append(-(zinnia_gates - halo2_gates) / halo2_gates * 100)
        prove_time_rates.append(-(zinnia_prove_time - halo2_prove_time) / halo2_prove_time * 100)
        verify_time_rates.append(-(zinnia_verify_time - halo2_verify_time) / halo2_verify_time * 100)
        snark_size_rates.append(-(zinnia_snark_size - halo2_snark_size) / halo2_snark_size * 100)
        zinnia_compile_times.append(value['zinnia_compile_time'])

    plt.rc('font', family='monospace', )
    # plt.rc('text', usetex=True)
    title_font = {'fontweight': 'bold', 'fontname': 'Times New Roman', 'fontsize': 12}

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(names, acc_rates, color=['firebrick' if x < 0 else 'forestgreen' for x in acc_rates])
    ax.tick_params(axis='x', labelrotation=90)
    ylabel = ax.set_ylabel('No. of Gates Reductions (%)', fontdict=title_font)
    ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1] - 0.3))
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    fig.tight_layout(pad=0)
    plt.show()
    fig.savefig('results-gate.pdf', dpi=300)


def main():
    plot_evaluation_results()


if __name__ == "__main__":
    main()
