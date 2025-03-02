import importlib
import json
import os.path
import re
import subprocess
import time

from zinnia import ZKCircuit

HALO2_FOLDER = "/home/zhantong/halo2-graph"
TIME_MEASURE_REPETITIONS = 10


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
        my_env['RUST_MIN_STACK'] = '16777216'
        my_env['LOOKUP_BITS'] = '6'
        keygen_process = subprocess.run(['cargo', 'run', '--example', 'target', '--', '--name', 'target', '-k', '16', '--input', 'target.in', 'keygen'], capture_output=True, text=True, env=my_env)
        keygen_feedback = keygen_process.stdout + keygen_process.stderr
        assert keygen_process.returncode == 0, keygen_feedback
        proving_time_in_seconds = 0
        snark_size = 0
        for i in range(TIME_MEASURE_REPETITIONS):
            prove_process = subprocess.run(['cargo', 'run', '--example', 'target', '--', '--name', 'target', '-k', '16', '--input', 'target.in', 'prove'], capture_output=True, text=True, env=my_env)
            prove_feedback = prove_process.stdout + prove_process.stderr
            assert prove_process.returncode == 0, prove_feedback
            match = re.search(r"Proving time: \s*([\d\.]+)(ms|s)", prove_feedback)
            assert match
            proving_time = float(match.group(1))
            proving_unit = match.group(2)
            proving_time_in_seconds += proving_time / 1000 if proving_unit == "ms" else proving_time
            snark_size = os.path.getsize(os.path.join(HALO2_FOLDER, "data/target.snark"))
        verify_time_in_seconds = 0
        advice_cells, fixed_cells, range_check_advice_cells = 0, 0, 0
        for i in range(TIME_MEASURE_REPETITIONS):
            verify_process = subprocess.run(['cargo', 'run', '--example', 'target', '--', '--name', 'target', '-k', '16', '--input', 'target.in', 'verify'], capture_output=True, text=True, env=my_env)
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
            verify_time_in_seconds += verify_time / 1000 if verify_unit == "ms" else verify_time
    except Exception as e:
        os.chdir(original_directory)
        raise e
    os.chdir(original_directory)
    return {
        "name": name,
        "proving_time": proving_time_in_seconds / TIME_MEASURE_REPETITIONS,
        "snark_size": snark_size,
        "advice_cells": advice_cells,
        "fixed_cells": fixed_cells,
        "range_check_advice_cells": range_check_advice_cells,
        "verify_time": verify_time_in_seconds / TIME_MEASURE_REPETITIONS
    }


def run_evaluate(dataset: str, problem: str):
    module = importlib.import_module('.' + dataset + '.' + problem + '.sol', 'benchmarking')
    # Get the method from the module
    method = getattr(module, 'verify_solution')
    # Get the circuit
    circuit = ZKCircuit.from_method(method)
    # Compile the circuit
    source = ""
    avg_time = 0
    for i in range(10):
        start_time = time.time()
        source = circuit.compile().source
        end_time = time.time()
        avg_time += end_time - start_time
    avg_time /= 10
    # Get the input data
    with open(os.path.join('../benchmarking', dataset, problem, 'sol.py.in'), 'r') as f:
        data = f.read()
    # Run
    result1 = run_prove(f"{dataset}::{problem}.py", source, data)
    # source 2
    with open(os.path.join('../benchmarking', dataset, problem, 'sol.rs'), 'r') as f:
        source = f.read()
    # Get the input data
    with open(os.path.join('../benchmarking', dataset, problem, 'sol.rs.in'), 'r') as f:
        data = f.read()
    result2 = run_prove(f"{dataset}::{problem}.rs", source, data)
    return {
        "zinnia": result1,
        "halo2": result2,
        "zinnia_compile_time": avg_time
    }


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

DATASETS = {
    "mlalgo": MLALGO,
    "leetcode_array": LEETCODE_ARRAY,
    "leetcode_dp": LEETCODE_DP,
    "leetcode_graph": LEETCODE_GRAPH,
    "leetcode_math": LEETCODE_MATH,
    "leetcode_matrix": LEETCODE_MATRIX
}


def main():
    if not os.path.exists('results.json'):
        results_dict = {}
    else:
        with open('results.json', 'r') as f:
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
                with open('results.json', 'w') as f:
                    f.write(json.dumps(results_dict, indent=2))
                raise e


    with open('results.json', 'w') as f:
        f.write(json.dumps(results_dict, indent=2))


if __name__ == '__main__':
    main()
