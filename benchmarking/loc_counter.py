import importlib
import os

MLALGO = [
    "neuron",
    "kmeans",
    "linear_regression"
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
    "case296",
    "case309",
    "case330",
    "case360",
    "case387",
    "case418",
    "case453",
    "case459",
    "case501",
    "case510",
]
CRYPT = [
    "ecc",
    "poseidon",
]

DATASETS = {
    "crypt": CRYPT,
    "mlalgo": MLALGO,
    "leetcode_array": LEETCODE_ARRAY,
    "leetcode_dp": LEETCODE_DP,
    "leetcode_graph": LEETCODE_GRAPH,
    "leetcode_math": LEETCODE_MATH,
    "leetcode_matrix": LEETCODE_MATRIX,
    "ds1000": DS1000
}


def count_zinnia(dataset: str, problem: str):
    with open(os.path.join('../benchmarking', dataset, problem, 'sol.py'), 'r') as f:
        zinnia_source = f.read()

    lines = zinnia_source.split('\n')
    lines = [line for line in lines if line.strip() != '']
    lines = [line for line in lines if not line.startswith('#')]
    lines = [line for line in lines if not line.startswith('from')]
    lines = [line for line in lines if not line.startswith('import')]
    return len(lines)


def count_halo2(dataset: str, problem: str):
    with open(os.path.join('../benchmarking', dataset, problem, 'sol.rs'), 'r') as f:
        source = f.read()

    lines = source.split('\n')
    idx1 = lines.index("    const PRECISION: u32 = 63;")
    idx2 = lines.index("fn main() {")
    lines = lines[idx1:idx2]
    lines = [line for line in lines if line.strip() != '']
    lines = [line for line in lines if not line.startswith('//')]
    return len(lines)


def count_sp1(dataset: str, problem: str):
    with open(os.path.join('../benchmarking', dataset, problem, 'sp1.prog.rs'), 'r') as f:
        source = f.read()

    lines = source.split('\n')
    try:
        idx = lines.index("// source start", 0)
        lines = lines[idx:]
    except ValueError:
        idx = lines.index("pub fn main() {")
        lines = lines[idx:]
    lines = [line for line in lines if line.strip() != '']
    lines = [line for line in lines if not line.startswith('//')]
    return len(lines)


def count_risc0(dataset: str, problem: str):
    with open(os.path.join('../benchmarking', dataset, problem, 'risc0.prog.rs'), 'r') as f:
        source = f.read()

    lines = source.split('\n')
    try:
        idx = lines.index("// source start", 0)
        lines = lines[idx:]
    except ValueError:
        idx = lines.index("fn main() {")
        lines = lines[idx:]
    lines = [line for line in lines if line.strip() != '']
    lines = [line for line in lines if not line.startswith('//')]
    return len(lines)


def count_cairo(dataset: str, problem: str):
    with open(os.path.join('../benchmarking', dataset, problem, 'main.cairo'), 'r') as f:
        source = f.read()

    lines = source.split('\n')
    lines = [line for line in lines if line.strip() != '']
    lines = [line for line in lines if not line.startswith('//')]
    return len(lines)


def count_noir(dataset: str, problem: str):
    with open(os.path.join('../benchmarking', dataset, problem, 'main.nr'), 'r') as f:
        source = f.read()

    lines = source.split('\n')
    lines = [line for line in lines if line.strip() != '']
    lines = [line for line in lines if not line.startswith('//')]
    return len(lines)



def main():
    zinnia_counts = []
    halo2_counts = []
    sp1_counts = []
    risc0_counts = []
    noir_counts = []
    for dataset, problems in DATASETS.items():
        tmp = []
        for problem in problems:
            cnt = count_zinnia(dataset, problem)
            tmp.append(cnt)
            zinnia_counts.append(cnt)
            print(f'{dataset}:{problem} (zinnia) = {cnt}')
        print(f'{dataset} avg (zinnia):', sum(tmp) / len(tmp))
    for dataset, problems in DATASETS.items():
        tmp = []
        for problem in problems:
            cnt = count_halo2(dataset, problem)
            halo2_counts.append(cnt)
            tmp.append(cnt)
            print(f'{dataset}:{problem} (halo2) = {cnt}')
        print(f'{dataset} avg (halo2):', sum(tmp) / len(tmp))
    for dataset, problems in DATASETS.items():
        tmp = []
        for problem in problems:
            cnt = count_sp1(dataset, problem)
            sp1_counts.append(cnt)
            tmp.append(cnt)
            print(f'{dataset}:{problem} (sp1) = {cnt}')
        print(f'{dataset} avg (sp1):', sum(tmp) / len(tmp))
    for dataset, problems in DATASETS.items():
        tmp = []
        for problem in problems:
            cnt = count_risc0(dataset, problem)
            risc0_counts.append(cnt)
            tmp.append(cnt)
            print(f'{dataset}:{problem} (risc0) = {cnt}')
        print(f'{dataset} avg (risc0):', sum(tmp) / len(tmp))
    for dataset, problems in DATASETS.items():
        tmp = []
        for problem in problems:
            try:
                cnt = count_noir(dataset, problem)
                noir_counts.append(cnt)
                tmp.append(cnt)
                print(f'{dataset}:{problem} (noir) = {cnt}')
            except:
                continue
        if tmp:
            print(f'{dataset} avg (noir):', sum(tmp) / len(tmp))
    for dataset, problems in DATASETS.items():
        tmp = []
        for problem in problems:
            try:
                cnt = count_cairo(dataset, problem)
                noir_counts.append(cnt)
                tmp.append(cnt)
                print(f'{dataset}:{problem} (cairo) = {cnt}')
            except:
                continue
        if tmp:
            print(f'{dataset} avg (cairo):', sum(tmp) / len(tmp))
    print('Zinnia avg:', sum(zinnia_counts) / len(zinnia_counts))
    print('Halo2 avg:', sum(halo2_counts) / len(halo2_counts))
    print('SP1 avg:', sum(sp1_counts) / len(sp1_counts))
    print('RISC0 avg:', sum(risc0_counts) / len(risc0_counts))
    print('Noir avg:', sum(noir_counts) / len(noir_counts))
    print('Cairo avg:', sum(noir_counts) / len(noir_counts))

if __name__ == '__main__':
    main()
