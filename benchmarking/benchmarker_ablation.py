import importlib
import json
import os.path
import re
import subprocess
import time
from multiprocessing import Pool

from zinnia import ZKCircuit, ZinniaConfig
from zinnia.config.optimization_config import OptimizationConfig

HALO2_FOLDER = "/home/zhantong/halo2-graph"
MULTIPROCESSING_POOL_SIZE = 1
ABLATION_SETTING = 1  # supported 1, 2, 3, 4
RESULT_PATH = f'results-ablation-{ABLATION_SETTING}.json'


def prove_executor(_):
    my_env = os.environ.copy()
    my_env['RUST_MIN_STACK'] = '536870912'  # 512 MiB
    my_env['LOOKUP_BITS'] = '6'
    prove_process = subprocess.run(
        ['cargo', 'run', '--example', 'target', '--', '--name', 'target', '-k', '16', '--input', 'target.in', 'prove'],
        capture_output=True, text=True, env=my_env)
    prove_feedback = prove_process.stdout + prove_process.stderr
    assert prove_process.returncode == 0, prove_feedback
    match = re.search(r"Proving time: \s*([\d\.]+)(ms|s)", prove_feedback)
    assert match
    proving_time = float(match.group(1))
    proving_unit = match.group(2)
    return proving_time / 1000 if proving_unit == "ms" else proving_time


def verify_executor(_):
    my_env = os.environ.copy()
    my_env['RUST_MIN_STACK'] = '536870912'  # 512 MiB
    my_env['LOOKUP_BITS'] = '6'
    verify_process = subprocess.run(
        ['cargo', 'run', '--example', 'target', '--', '--name', 'target', '-k', '16', '--input', 'target.in', 'verify'],
        capture_output=True, text=True, env=my_env)
    verify_feedback = verify_process.stdout + verify_process.stderr
    assert verify_process.returncode == 0, verify_feedback
    match = re.search(r"Gate Chip \| Phase 0: \s*([\d\.]+) advice cells", verify_feedback)
    assert match
    advice_cells = int(match.group(1))
    match = re.search(r"Total \s*([\d\.]+) fixed cells", verify_feedback)
    assert match
    fixed_cells = int(match.group(1))
    match = re.search(r"Total range check advice cells to lookup per phase: \[\s*([\d\.]+), 0, 0]", verify_feedback)
    assert match
    range_check_advice_cells = int(match.group(1))
    match = re.search(r"Snark verified successfully in \s*([\d\.]+)(ms|s)", verify_feedback)
    assert match
    verify_time = float(match.group(1))
    verify_unit = match.group(2)
    return advice_cells, fixed_cells, range_check_advice_cells, verify_time / 1000 if verify_unit == "ms" else verify_time


def run_prove(name: str, source: str, data: str):
    original_directory = os.getcwd()
    try:
        os.chdir(HALO2_FOLDER)
        with open(os.path.join(HALO2_FOLDER, "examples/target.rs"), "w") as f:
            f.write(source)
        with open(os.path.join(HALO2_FOLDER, "data/target.in"), "w") as f:
            f.write(data)
        # set pwd to halo2 folder
        # run the command
        my_env = os.environ.copy()
        my_env['RUST_MIN_STACK'] = '536870912'  # 512 MiB
        my_env['LOOKUP_BITS'] = '6'
        keygen_process = subprocess.run(['cargo', 'run', '--example', 'target', '--', '--name', 'target', '-k', '16', '--input', 'target.in', 'keygen'], capture_output=True, text=True, env=my_env)
        keygen_feedback = keygen_process.stdout + keygen_process.stderr
        assert keygen_process.returncode == 0, keygen_feedback
        with Pool(MULTIPROCESSING_POOL_SIZE) as p:
            results = p.map(prove_executor, [_ for _ in range(1)])
            proving_time_in_seconds = sum([result for result in results]) / len(results)
        snark_size = os.path.getsize(os.path.join(HALO2_FOLDER, "data/target.snark"))
        with Pool(MULTIPROCESSING_POOL_SIZE) as p:
            results = p.map(verify_executor, [_ for _ in range(1)])
            advice_cells, fixed_cells, range_check_advice_cells, _ = results[0]
            verify_time_in_seconds = sum([result[3] for result in results]) / len(results)
    except Exception as e:
        os.chdir(original_directory)
        raise e
    os.chdir(original_directory)
    return {
        "name": name,
        "proving_time": proving_time_in_seconds,
        "snark_size": snark_size,
        "advice_cells": advice_cells,
        "fixed_cells": fixed_cells,
        "range_check_advice_cells": range_check_advice_cells,
        "verify_time": verify_time_in_seconds
    }


def compile_executor(circuit):
    start_time = time.time()
    source = circuit.compile().source
    end_time = time.time()
    return source, end_time - start_time


def run_evaluate(dataset: str, problem: str):
    module = importlib.import_module('.' + dataset + '.' + problem + '.sol', 'benchmarking')
    # Get the method from the module
    method = getattr(module, 'verify_solution')
    chips = getattr(module, 'chips', [])
    # Get the circuit
    config = ZinniaConfig(optimization_config=OptimizationConfig(
        always_satisfied_elimination=False if ABLATION_SETTING == 1 else True,
        constant_fold=False if ABLATION_SETTING == 1 else True,
        dead_code_elimination=False if ABLATION_SETTING == 2 else True,  # the program gets toooo big without those optimizations. So in abalation study we always enable it
        duplicate_code_elimination=False if ABLATION_SETTING == 3 else True,
        shortcut_optimization=False if ABLATION_SETTING == 4 else True
    ))
    circuit = ZKCircuit.from_method(method, chips=chips, config=config)
    # Compile the circuit
    with Pool(MULTIPROCESSING_POOL_SIZE) as p:
        results = p.map(compile_executor, [circuit for _ in range(1)])
        source = results[0][0]
        avg_time = sum([result[1] for result in results]) / len(results)
    # Get the input data
    with open(os.path.join('../benchmarking', dataset, problem, 'sol.py.in'), 'r') as f:
        data = f.read()
    # Run
    result1 = run_prove(f"{dataset}::{problem}.py", source, data)
    return {
        "zinnia": result1,
        "zinnia_compile_time": avg_time
    }



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
    'case295', 'case296', 'case297', 'case298', 'case299', 'case300', 'case301', 'case302', 'case303', 'case304', 'case309', 'case310', 'case313', 'case318', 'case319', 'case322', 'case323', 'case324', 'case329', 'case330', 'case334', 'case335', 'case336', 'case337', 'case338', 'case339', 'case353', 'case354', 'case360', 'case368', 'case369', 'case370', 'case373', 'case374', 'case375', 'case385', 'case387', 'case388', 'case389', 'case390', 'case391', 'case392', 'case393', 'case406', 'case407', 'case408', 'case409', 'case414', 'case415', 'case416', 'case417', 'case418', 'case419', 'case420', 'case428', 'case429', 'case430', 'case431', 'case433', 'case434', 'case435', 'case436', 'case437', 'case438', 'case440', 'case441', 'case452', 'case453', 'case459', 'case480', 'case501', 'case507', 'case510'
]
CRYPT = [
    "ecc",
    "poseidon",
    "elgamal",
    "mimc",
    "merkle"
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


def main():
    if not os.path.exists(RESULT_PATH):
        results_dict = {}
    else:
        with open(RESULT_PATH, 'r') as f:
            results_dict = json.load(f)

    for dataset, problems in DATASETS.items():
        for problem in problems:
            if f"{dataset}::{problem}" in results_dict.keys():
                continue
            try:
                print('Evaluating', f"{dataset}::{problem}")
                result = run_evaluate(dataset, problem)
                results_dict[f"{dataset}::{problem}"] = result
            except AssertionError as e:
                print(f"Failed to evaluate {dataset}::{problem}. Skipping...")
                with open(RESULT_PATH, 'w') as f:
                    f.write(json.dumps(results_dict, indent=2))
                raise e
        with open(RESULT_PATH, 'w') as f:
            f.write(json.dumps(results_dict, indent=2))


    with open(RESULT_PATH, 'w') as f:
        f.write(json.dumps(results_dict, indent=2))


if __name__ == '__main__':
    main()
