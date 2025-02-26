import json

import matplotlib.pyplot as plt


def plot_halo2_benchmarking_results():
    with open('results.json', 'r') as f:
        results_dict = json.load(f)

    names = []
    acc_rates = []
    prove_time_rates = []
    verify_time_rates = []
    snark_size_rates = []
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

    fig, ax = plt.subplots()
    ax.bar(names, acc_rates, color=['firebrick' if x < 0 else 'forestgreen' for x in acc_rates])
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('No. of Gates Impro. Rate (%)')
    ax.set_title('Zinnia Compiler Performance Overviews')
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    fig.tight_layout()
    plt.show()
    fig.savefig('results-gate.png', dpi=300)

    fig, ax = plt.subplots()
    ax.bar(names, prove_time_rates, color=['firebrick' if x < 0 else 'forestgreen' for x in prove_time_rates])
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('Prove Time Impro. Rate (%)')
    ax.set_title('Zinnia Compiler Performance Overviews')
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    fig.tight_layout()
    plt.show()
    fig.savefig('results-prove-time.png', dpi=300)

    fig, ax = plt.subplots()
    ax.bar(names, verify_time_rates, color=['firebrick' if x < 0 else 'forestgreen' for x in verify_time_rates])
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('Verify Time Impro. Rate (%)')
    ax.set_title('Zinnia Compiler Performance Overviews')
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    fig.tight_layout()
    plt.show()
    fig.savefig('results-verify-time.png', dpi=300)

    fig, ax = plt.subplots()
    ax.bar(names, snark_size_rates, color=['firebrick' if x < 0 else 'forestgreen' for x in snark_size_rates])
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('Snark Size Impro. Rate (%)')
    ax.set_title('Zinnia Compiler Performance Overviews')
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    fig.tight_layout()
    plt.show()
    fig.savefig('results-snark-size.png', dpi=300)


def main():
    plot_halo2_benchmarking_results()


if __name__ == "__main__":
    main()
