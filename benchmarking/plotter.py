import json

import matplotlib.pyplot as plt
import numpy as np


def plot_halo2_benchmarking_results():
    with open('results.json', 'r') as f:
        results_dict = json.load(f)

    names = []
    acc_rates = []
    prove_time_rates = []
    verify_time_rates = []
    snark_size_rates = []
    zinnia_compile_times = []
    for key, value in results_dict.items():
        names.append(key)
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

    fig, ax = plt.subplots()
    ax.bar(names, acc_rates, color=['firebrick' if x < 0 else 'forestgreen' for x in acc_rates])
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('No. of Gates Reduction Rate (%)')
    ax.set_title('Zinnia Compiler Performance Overviews')
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    fig.tight_layout()
    plt.show()
    fig.savefig('results-gate.png', dpi=300)

    fig, ax = plt.subplots()
    ax.bar(names, prove_time_rates, color=['firebrick' if x < 0 else 'forestgreen' for x in prove_time_rates])
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('Prove Time Reduction Rate (%)')
    ax.set_title('Zinnia Compiler Performance Overviews')
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    fig.tight_layout()
    plt.show()
    fig.savefig('results-prove-time.png', dpi=300)

    fig, ax = plt.subplots()
    ax.bar(names, verify_time_rates, color=['firebrick' if x < 0 else 'forestgreen' for x in verify_time_rates])
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('Verify Time Reduction Rate (%)')
    ax.set_title('Zinnia Compiler Performance Overviews')
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    fig.tight_layout()
    plt.show()
    fig.savefig('results-verify-time.png', dpi=300)

    fig, ax = plt.subplots()
    ax.bar(names, snark_size_rates, color=['firebrick' if x < 0 else 'forestgreen' for x in snark_size_rates])
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('Snark Size Reduction Rate (%)')
    ax.set_title('Zinnia Compiler Performance Overviews')
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    fig.tight_layout()
    plt.show()
    fig.savefig('results-snark-size.png', dpi=300)

    fig, ax = plt.subplots()
    ax.bar(names, zinnia_compile_times, color='skyblue')
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('Zinnia Compile Time (s)')
    ax.set_title('Zinnia Compiler Performance Overviews')
    fig.tight_layout()
    plt.show()
    fig.savefig('results-zinnia-compile-time.png', dpi=300)


def plot_sp1_benchmarking_results():
    with open('results.json', 'r') as f:
        zinnia_results_dict = json.load(f)
    with open('results-sp1.json', 'r') as f:
        sp1_results_dict = json.load(f)
    
    names = []
    zinnia_snark_proving_time = []
    zinnia_verify_time = []
    zinnia_snark_size = []
    sp1_stark_proving_time = []
    sp1_snark_proving_time = []
    sp1_verify_time = []
    sp1_snark_size = []
    for key, value in sp1_results_dict.items():
        _zinnia_prove_time = zinnia_results_dict[key]['zinnia']['proving_time']
        _zinnia_verify_time = zinnia_results_dict[key]['zinnia']['verify_time']
        _zinnia_snark_size = zinnia_results_dict[key]['zinnia']['snark_size']

        names.append(key)
        zinnia_snark_proving_time.append(_zinnia_prove_time)
        zinnia_snark_size.append(_zinnia_snark_size)
        zinnia_verify_time.append(_zinnia_verify_time)
        sp1_stark_proving_time.append(value['stark_proving_time'])
        sp1_snark_proving_time.append(value['snark_proving_time'])
        sp1_snark_size.append(value['snark_size'])
        sp1_verify_time.append(value['verify_time'])
    
    fig, ax = plt.subplots()
    width = 0.25
    x = np.arange(len(names))
    ax.bar(x + width * 0, zinnia_snark_proving_time, width, label='Zinnia zk-SNARK')
    ax.bar(x + width * 1, sp1_stark_proving_time, width, label='SP1 zk-STARK')
    ax.bar(x + width * 2, sp1_snark_proving_time, width, label='SP1 zk-SNARK')
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('Prove Time (s)')
    ax.set_xticks(x + width, names)
    ax.set_title('Zinnia Compiler Performance Overviews')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    plt.show()
    fig.savefig('results-sp1-proving-time.png', dpi=300)

    fig, ax = plt.subplots()
    width = 0.25
    x = np.arange(len(names))
    ax.bar(x + width * 0, zinnia_verify_time, width, label='Zinnia')
    ax.bar(x + width * 1, sp1_verify_time, width, label='SP1')
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('Verify Time (s)')
    ax.set_xticks(x + width / 2, names)
    ax.set_title('Zinnia Compiler Performance Overviews')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    plt.show()
    fig.savefig('results-sp1-verifying-time.png', dpi=300)

    fig, ax = plt.subplots()
    width = 0.25
    x = np.arange(len(names))
    ax.bar(x + width * 0, zinnia_snark_size, width, label='Zinnia')
    ax.bar(x + width * 1, sp1_snark_size, width, label='SP1')
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('Snark Size (bytes)')
    ax.set_xticks(x + width / 2, names)
    ax.set_title('Zinnia Compiler Performance Overviews')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    plt.show()
    fig.savefig('results-sp1-snark-size.png', dpi=300)



def main():
    plot_halo2_benchmarking_results()
    plot_sp1_benchmarking_results()


if __name__ == "__main__":
    main()
