import json
import os.path
import re
import subprocess
import time

CAIRO_FOLDER = "/home/zhantong/cairo_projects/hello_world"
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
DS1000 = ['case295', 'case296', 'case297', 'case299', 'case301', 'case302', 'case303', 'case304', 'case309', 'case310',
          'case313', 'case318', 'case319', 'case322', 'case329', 'case335', 'case336', 'case337', 'case338', 'case339',
          'case353', 'case354', 'case360', 'case368', 'case369', 'case370', 'case385', 'case387', 'case388', 'case389',
          'case390', 'case391', 'case392', 'case393', 'case406', 'case407', 'case408', 'case409', 'case415', 'case433',
          'case434', 'case435', 'case436', 'case437', 'case441', 'case480', 'case501', 'case507', 'case510']
CRYPT = [
    # "ecc",  # Cairo's p value is different
    "poseidon",
]

DATASETS = {
    "crypt": CRYPT,
    # "mlalgo": MLALGO,
    "leetcode_array": LEETCODE_ARRAY,
    "leetcode_dp": LEETCODE_DP,
    "leetcode_graph": LEETCODE_GRAPH,
    "leetcode_math": LEETCODE_MATH,
    "leetcode_matrix": LEETCODE_MATRIX,
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
