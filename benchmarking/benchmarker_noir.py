import importlib
import json
import os.path
import re
import subprocess
import time

from zinnia import ZinniaConfig, ZKCircuit
from zinnia.config import OptimizationConfig

NOIR_FOLDER = "/Users/zhantong/Projects/hello_noir"
TIME_MEASURE_REPETITIONS = 100


def run_prove(name: str, input_source: str, program_source: str):
    original_directory = os.getcwd()
    try:
        with open(os.path.join(NOIR_FOLDER, "Prover.toml"), "w") as f:
            f.write(input_source)
        with open(os.path.join(NOIR_FOLDER, "src/main.nr"), "w") as f:
            f.write(program_source)
        # set pwd to sp1 folder
        os.chdir(NOIR_FOLDER)
        # run the command
        my_env = os.environ.copy()
        circuit_size = 0
        total_gates = 0
        opcode_count = 0
        proving_times = []
        verifying_times = []
        compilation_times = []
        for i in range(TIME_MEASURE_REPETITIONS):
            start_time = time.time()
            execute_process = subprocess.run(['nargo', 'execute'], capture_output=True, text=True, env=my_env)
            execute_process_feedback = execute_process.stdout + execute_process.stderr
            assert execute_process.returncode == 0, execute_process_feedback
            assert "Circuit witness successfully solved" in execute_process_feedback
            end_time = time.time()
            compilation_times.append(end_time - start_time)
            start_time = time.time()
            prove_process = subprocess.run(
                ['bb', 'prove', '-b', './target/hello_noir.json', '-w', './target/hello_noir.gz', '-o', './target'],
                capture_output=True, text=True, env=my_env)
            prove_process_feedback = prove_process.stdout + prove_process.stderr
            assert prove_process.returncode == 0, prove_process_feedback
            match = re.search(r"Finalized circuit size: \s*([\d\.]+)", prove_process_feedback)
            assert match
            circuit_size = int(match.group(1))
            end_time = time.time()
            proving_times.append(end_time - start_time)
            write_vk_process = subprocess.run(['bb', 'write_vk', '-b', './target/hello_noir.json', '-o', './target'],
                                              capture_output=True, text=True, env=my_env)
            assert write_vk_process.returncode == 0
            start_time = time.time()
            verify_process = subprocess.run(['bb', 'verify', '-k', './target/vk', '-p', './target/proof'],
                                            capture_output=True, text=True, env=my_env)
            assert verify_process.returncode == 0
            end_time = time.time()
            verifying_times.append(end_time - start_time)
            profiler_process = subprocess.run(
                ['noir-profiler', 'gates', '--artifact-path', './target/hello_noir.json', '--backend-path', 'bb',
                 '--output', './target', '--', '--include_gates_per_opcode'], capture_output=True, text=True,
                env=my_env)
            assert profiler_process.returncode == 0
            profiler_process_feedback = profiler_process.stdout + profiler_process.stderr
            match = re.search(r"Total gates by opcodes: \s*([\d\.]+),", profiler_process_feedback)
            assert match
            total_gates = int(match.group(1))
            match = re.search(r"Opcode count: \s*([\d\.]+),", profiler_process_feedback)
            assert match
            opcode_count = int(match.group(1))
    except Exception as e:
        os.chdir(original_directory)
        raise e
    os.chdir(original_directory)
    return {
        "name": name,
        "circuit_size": circuit_size,
        "opcode_count": opcode_count,
        "total_gates": total_gates,
        "proving_time": sum(proving_times) / len(proving_times),
        "verifying_time": sum(verifying_times) / len(verifying_times),
        "nargo_compilation_time": sum(compilation_times) / len(compilation_times),
    }


def run_evaluate(dataset: str, problem: str):
    # Get the driver source
    with open(os.path.join('../benchmarking', dataset, problem, 'Prover.toml'), 'r') as f:
        baseline_input_source = f.read()
    with open(os.path.join('../benchmarking', dataset, problem, 'Prover.zinnia.toml'), 'r') as f:
        ours_input_source = f.read()
    # Get the program source
    with open(os.path.join('../benchmarking', dataset, problem, 'main.nr'), 'r') as f:
        baseline_program_source = f.read()
    module = importlib.import_module('.' + dataset + '.' + problem + '.sol', 'benchmarking')
    # Get the method from the module
    method = getattr(module, 'verify_solution')
    chips = getattr(module, 'chips', [])
    # Get the circuit
    config = ZinniaConfig(backend="noir")
    circuit = ZKCircuit.from_method(method, chips=chips, config=config)
    compiled_program = circuit.compile()
    ours_program_source = compiled_program.source
    # Run
    result_ours = run_prove(f"{dataset}::{problem}.ours.noir", ours_input_source, ours_program_source)
    result_baseline = run_prove(f"{dataset}::{problem}.baseline.noir", baseline_input_source, baseline_program_source)
    return {
        "ours_on_noir": result_ours,
        "baseline_on_noir": result_baseline,
        "zinnia_compilation_time": compiled_program.eval_data
    }


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
DS1000 = ['case295', 'case296', 'case297', 'case299', 'case301', 'case302', 'case303', 'case304', 'case309', 'case310',
          'case313', 'case318', 'case319', 'case322', 'case329', 'case335', 'case336', 'case337', 'case338', 'case339',
          'case353', 'case354', 'case360', 'case368', 'case369', 'case370', 'case385', 'case387', 'case388', 'case389',
          'case390', 'case391', 'case392', 'case393', 'case406', 'case407', 'case408', 'case409', 'case415', 'case433',
          'case434', 'case435', 'case436', 'case437', 'case441', 'case480', 'case501', 'case507', 'case510']
CRYPT = [
    "ecc",
    "poseidon",
    "mimc",
    "elgamal",
    "merkle"
]

DATASETS = {
    "crypt": CRYPT,
    "leetcode_array": LEETCODE_ARRAY,
    "leetcode_dp": LEETCODE_DP,
    "leetcode_graph": LEETCODE_GRAPH,
    "leetcode_math": LEETCODE_MATH,
    "leetcode_matrix": LEETCODE_MATRIX,
    "ds1000": DS1000
}


def main():
    if not os.path.exists('results-noir.json'):
        results_dict = {}
    else:
        with open('results-noir.json', 'r') as f:
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
                raise e
            finally:
                with open('results-noir.json', 'w') as f:
                    f.write(json.dumps(results_dict, indent=2))
        with open('results-noir.json', 'w') as f:
            f.write(json.dumps(results_dict, indent=2))
    with open('results-noir.json', 'w') as f:
        f.write(json.dumps(results_dict, indent=2))


if __name__ == '__main__':
    main()
