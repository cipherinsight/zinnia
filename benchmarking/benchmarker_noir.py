import json
import os.path
import re
import subprocess


NOIR_FOLDER = "/Users/zhantong/Projects/hello_noir"
TIME_MEASURE_REPETITIONS = 10


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
        for i in range(TIME_MEASURE_REPETITIONS):
            subprocess.run(['nargo', 'execute'], capture_output=True, text=True, env=my_env)
            prove_process = subprocess.run(['bb', 'prove', '-b', './target/hello_noir.json', '-w', './target/hello_noir.gz', '-o', './target'], capture_output=True, text=True, env=my_env)
            prove_process_feedback = prove_process.stdout + prove_process.stderr
            assert prove_process.returncode == 0, prove_process_feedback
            match = re.search(r"Finalized circuit size: \s*([\d\.]+)", prove_process_feedback)
            assert match
            circuit_size = int(match.group(1))
            write_vk_process = subprocess.run(['bb', 'write_vk', '-b', './target/hello_noir.json', '-o', './target'], capture_output=True, text=True, env=my_env)
            assert write_vk_process.returncode == 0
            verify_process = subprocess.run(['bb', 'verify', '-k', './target/vk', '-p', './target/proof'], capture_output=True, text=True, env=my_env)
            assert verify_process.returncode == 0
            profiler_process = subprocess.run(['noir-profiler', 'gates', '--artifact-path', './target/hello_noir.json', '--backend-path', 'bb', '--output', './target', '--', '--include_gates_per_opcode'], capture_output=True, text=True, env=my_env)
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
        "total_gates": total_gates
    }


def run_evaluate(dataset: str, problem: str):
    # Get the driver source
    with open(os.path.join('../benchmarking', dataset, problem, 'Prover.toml'), 'r') as f:
        input_source = f.read()
    # Get the program source
    with open(os.path.join('../benchmarking', dataset, problem, 'main.nr'), 'r') as f:
        program_source = f.read()
    # Run
    return run_prove(f"{dataset}::{problem}.py", input_source, program_source)


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
    "case360",
    "case387",
    "case501",
    "case510",
]
CRYPT = [
    "ecc",
    "poseidon",
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
                with open('results-noir.json', 'w') as f:
                    f.write(json.dumps(results_dict, indent=2))
                raise e
        with open('results-noir.json', 'w') as f:
            f.write(json.dumps(results_dict, indent=2))

    with open('results-noir.json', 'w') as f:
        f.write(json.dumps(results_dict, indent=2))


if __name__ == '__main__':
    main()
