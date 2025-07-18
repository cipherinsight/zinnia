import json
import os.path
import re
import subprocess
import time

CAIRO_FOLDER = "/home/cairo_projects/hello_world"
TIME_MEASURE_REPETITIONS = 10


def run_prove(name: str, program_source: str):
    original_directory = os.getcwd()
    try:
        with open(os.path.join(CAIRO_FOLDER, "src/hello_world.cairo"), "w") as f:
            f.write(program_source)
        # set pwd to cairo folder
        os.chdir(CAIRO_FOLDER)
        # run the command
        my_env = os.environ.copy()
        stark_proving_time_in_seconds = 0
        stark_verifying_time_in_seconds = 0
        for i in range(TIME_MEASURE_REPETITIONS):
            execute_process = subprocess.run(['scarb', 'execute', '-p', 'hello_world'], capture_output=True, text=True, env=my_env)
            execute_process_feedback = execute_process.stdout + execute_process.stderr
            assert execute_process.returncode == 0, execute_process_feedback
            match = re.search(r"execution(\w+)", execute_process_feedback)
            assert match
            execution_id = int(match.group(1))
            start_time = time.time()
            prove_process = subprocess.run(['scarb', 'prove', '--execution-id', str(execution_id)], capture_output=True, text=True, env=my_env)
            prove_process_feedback = prove_process.stdout + prove_process.stderr
            assert prove_process.returncode == 0, prove_process_feedback
            proving_time = time.time() - start_time
            stark_proving_time_in_seconds += proving_time
            start_time = time.time()
            verify_process = subprocess.run(['scarb', 'verify', '--execution-id', str(execution_id)], capture_output=True, text=True, env=my_env)
            verify_process_feedback = verify_process.stdout + verify_process.stderr
            assert verify_process.returncode == 0, verify_process_feedback
            verifying_time = time.time() - start_time
            stark_verifying_time_in_seconds += verifying_time
    except Exception as e:
        os.chdir(original_directory)
        raise e
    os.chdir(original_directory)
    return {
        "name": name,
        "stark_proving_time": stark_proving_time_in_seconds / TIME_MEASURE_REPETITIONS,
        "stark_verify_time": stark_verifying_time_in_seconds / TIME_MEASURE_REPETITIONS,
    }


def run_evaluate(dataset: str, problem: str):
    # Get the program source
    with open(os.path.join('../benchmarking', dataset, problem, 'main.cairo'), 'r') as f:
        program_source = f.read()
    # Run
    return run_prove(f"{dataset}::{problem}.py", program_source)


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
    # "case330",
    "case360",
    "case387",
    # "case418",
    # "case453",
    # "case459",
    "case501",
    "case510",
]
CRYPT = [
    "ecc",
    "poseidon",
]

DATASETS = {
    # "crypt": CRYPT,
    # "mlalgo": MLALGO,
    # "leetcode_array": LEETCODE_ARRAY,
    # "leetcode_dp": LEETCODE_DP,
    # "leetcode_graph": LEETCODE_GRAPH,
    # "leetcode_math": LEETCODE_MATH,
    # "leetcode_matrix": LEETCODE_MATRIX,
    "ds1000": DS1000
}


def main():
    if not os.path.exists('results-cairo.json'):
        results_dict = {}
    else:
        with open('results-cairo.json', 'r') as f:
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
                with open('results-cairo.json', 'w') as f:
                    f.write(json.dumps(results_dict, indent=2))
                raise e
        with open('results-cairo.json', 'w') as f:
            f.write(json.dumps(results_dict, indent=2))

    with open('results-cairo.json', 'w') as f:
        f.write(json.dumps(results_dict, indent=2))


if __name__ == '__main__':
    main()
