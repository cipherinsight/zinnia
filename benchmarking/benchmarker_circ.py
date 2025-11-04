import importlib
import json
import os.path
import re
import subprocess
import time
from multiprocessing import Pool

from zinnia import ZKCircuit, ZinniaConfig
from zinnia.config.optimization_config import OptimizationConfig

CIRC_FOLDER = "/home/zhantong/circ"
RESULT_PATH = 'results-circ-optimizer.json'
# Note: please comment or uncomment the circ's ./examples/circ.rs 335-336 lines accordingly when toggling this flag.
# Be reminded to recompile circ before running the benchmark.
ENABLE_OPTIMIZATIONS = True


def run_setup_to_collect_data(_):
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


def run_prove(name: str, source: str) -> dict:
    original_directory = os.getcwd()
    try:
        # set pwd to circ folder
        os.chdir(CIRC_FOLDER)
        with open(os.path.join(CIRC_FOLDER, "./example.zok"), "w") as f:
            f.write(source)
        # run the command
        my_env = os.environ.copy()
        my_env['RUST_MIN_STACK'] = '536870912'  # 512 MiB
        my_env['LOOKUP_BITS'] = '6'
        setup_process = subprocess.run(['./target/release/examples/circ', 'example.zok', 'r1cs', '--action', 'setup'], capture_output=True, text=True, env=my_env)
        setup_feedback = setup_process.stdout + setup_process.stderr
        assert setup_process.returncode == 0, setup_feedback
        match = re.search(r"Final r1cs: \s*([\d\.]+) constraints", setup_feedback)
        no_of_constraints = int(match.group(1))
    except Exception as e:
        os.chdir(original_directory)
        raise e
    os.chdir(original_directory)
    return {
        "name": name,
        "no_of_constraints": no_of_constraints,
    }


def run_evaluate(dataset: str, problem: str) -> dict:
    module = importlib.import_module('.' + dataset + '.' + problem + '.sol', 'benchmarking')
    # Get the method from the module
    method = getattr(module, 'verify_solution')
    chips = getattr(module, 'chips', [])
    # Get the circuit
    config = ZinniaConfig(backend=ZinniaConfig.BACKEND_CIRC_ZOK)
    circuit = ZKCircuit.from_method(method, chips=chips, config=config)
    # Compile the circuit
    source = circuit.compile().source
    # Run
    result = run_prove(f"{dataset}::{problem}.py", source)
    return result


# removed floating point problems and crypt problems --- either not implemented or because Zokrates' p value difference
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
    # "p492",
    "p2125"
]
LEETCODE_MATRIX = [
    "p73",
    "p2133"
]
DS1000 = ['case295', 'case296', 'case297', 'case299', 'case301', 'case302', 'case303', 'case304', 'case309', 'case310',
          'case313', 'case318', 'case319', 'case322', 'case335', 'case336', 'case337', 'case338', 'case339', # 'case329' removed for power function
          'case353', 'case354', 'case360', 'case368', 'case369', 'case370', 'case385', 'case387', 'case388', 'case389',
          'case390', 'case391', 'case392', 'case393', 'case406', 'case407', 'case408', 'case409', 'case415', 'case433',
          'case434', 'case435', 'case436', 'case437', 'case441', 'case480', 'case501', 'case507', 'case510']

DATASETS = {
    "ds1000": DS1000,
    "leetcode_array": LEETCODE_ARRAY,
    "leetcode_dp": LEETCODE_DP,
    "leetcode_graph": LEETCODE_GRAPH,
    "leetcode_math": LEETCODE_MATH,
    "leetcode_matrix": LEETCODE_MATRIX,
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
                # if ENABLE_OPTIMIZATIONS and "circ_optimization_enabled" in results_dict[f"{dataset}::{problem}"].keys():
                    # continue
                if not ENABLE_OPTIMIZATIONS and "circ_optimization_disabled" in results_dict[f"{dataset}::{problem}"].keys():
                    continue
            try:
                print('Evaluating', f"{dataset}::{problem}")
                result = run_evaluate(dataset, problem)
                if f"{dataset}::{problem}" not in results_dict.keys():
                    results_dict[f"{dataset}::{problem}"] = {}
                if ENABLE_OPTIMIZATIONS:
                    results_dict[f"{dataset}::{problem}"]["circ_optimization_enabled"] = result
                else:
                    results_dict[f"{dataset}::{problem}"]["circ_optimization_disabled"] = result
            except AssertionError as e:
                print(f"Failed to evaluate {dataset}::{problem}. Skipping...")
                raise e
            finally:
                with open(RESULT_PATH, 'w') as f:
                    f.write(json.dumps(results_dict, indent=2))


if __name__ == '__main__':
    main()
