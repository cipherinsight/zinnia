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
RESULT_PATH = 'results-scale.json'
ENABLE_OPTIMIZATIONS = True


def run_keygen(name: str, source: str, data: str):
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
        match = re.search(r': (\d+) advice cells', keygen_feedback)
        assert match
        advice_cells = int(match.group(1))
    except Exception as e:
        os.chdir(original_directory)
        raise e
    os.chdir(original_directory)
    return {
        "name": name,
        "advice_cells": advice_cells
    }


def compile_executor(circuit):
    start_time = time.time()
    source = circuit.compile().source
    end_time = time.time()
    return source, end_time - start_time


def run_evaluate(dataset: str, problem: str):
    # Get the input data
    all_results = []
    for scale in [2, 4, 8, 16]:
        module = importlib.import_module('.' + dataset + '.' + problem + '.sol-' + str(scale), 'benchmarking')
        # Get the method from the module
        method = getattr(module, 'verify_solution')
        chips = getattr(module, 'chips', [])
        # Get the circuit
        config = ZinniaConfig(optimization_config=OptimizationConfig(
            always_satisfied_elimination=ENABLE_OPTIMIZATIONS,
            constant_fold=ENABLE_OPTIMIZATIONS,
            dead_code_elimination=ENABLE_OPTIMIZATIONS,  # the program gets toooo big without those optimizations. So in abalation study we always enable it
            duplicate_code_elimination=ENABLE_OPTIMIZATIONS,
            shortcut_optimization=ENABLE_OPTIMIZATIONS
        ))
        circuit = ZKCircuit.from_method(method, chips=chips, config=config)
        # Compile the circuit
        results = compile_executor(circuit)
        source = results[0]
        print(source)
        with open(os.path.join('../benchmarking', dataset, problem, f'sol.{scale}.py.in'), 'r') as f:
            data = f.read()
        # Run
        result1 = run_keygen(f"{dataset}::{problem}.py", source, data)
        # source 2
        with open(os.path.join('../benchmarking', dataset, problem, f'sol.{scale}.rs'), 'r') as f:
            source = f.read()
        # Get the input data
        with open(os.path.join('../benchmarking', dataset, problem, f'sol.{scale}.rs.in'), 'r') as f:
            data = f.read()
        result2 = run_keygen(f"{dataset}::{problem}.rs", source, data)
        all_results.append([result1, result2])
    return {
        "zinnia_10": all_results[0][0],
        "halo2_10": all_results[0][1],
        "zinnia_100": all_results[1][0],
        "halo2_100": all_results[1][1],
        "zinnia_1000": all_results[2][0],
        "halo2_1000": all_results[2][1],
    }


MLALGO = [
    "neuron",
    "kmeans",
    "linear_regression"
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


DATASETS = {
    "mlalgo": MLALGO,
    # "ds1000": DS1000
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
