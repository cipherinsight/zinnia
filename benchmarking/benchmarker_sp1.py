import json
import os.path
import re
import subprocess


SP1_FOLDER = "/home/zhantong/sp1proj"
TIME_MEASURE_REPETITIONS = 10


def run_prove(name: str, driver_source: str, program_source: str):
    original_directory = os.getcwd()
    try:
        with open(os.path.join(SP1_FOLDER, "script/src/bin/main.rs"), "w") as f:
            f.write(driver_source)
        with open(os.path.join(SP1_FOLDER, "program/src/main.rs"), "w") as f:
            f.write(program_source)
        # set pwd to sp1 folder
        os.chdir(SP1_FOLDER)
        # run the command
        my_env = os.environ.copy()
        stark_proving_time_in_seconds = 0
        snark_proving_time_in_seconds = 0
        verify_time_in_seconds = 0
        snark_size = 0
        for i in range(TIME_MEASURE_REPETITIONS):
            prove_process = subprocess.run(['cargo', 'run', '--release', '--', '--prove'], capture_output=True, text=True, env=my_env)
            prove_process_feedback = prove_process.stdout + prove_process.stderr
            assert prove_process.returncode == 0, prove_process_feedback
            match = re.search(r"Prove time \(zk-STARK\) \(ms\): \s*([\d\.]+)", prove_process_feedback)
            assert match
            proving_time_1 = float(match.group(1))
            match = re.search(r"Prove time \(ms\): \s*([\d\.]+)", prove_process_feedback)
            assert match
            proving_time_2 = float(match.group(1))
            match = re.search(r"Verify time \(ms\): \s*([\d\.]+)", prove_process_feedback)
            assert match
            verify_time = float(match.group(1))
            stark_proving_time_in_seconds += proving_time_1 / 1000
            snark_proving_time_in_seconds += proving_time_2 / 1000
            verify_time_in_seconds += verify_time / 1000
            snark_size = os.path.getsize(os.path.join(SP1_FOLDER, "proof-with-pis.bin"))
    except Exception as e:
        os.chdir(original_directory)
        raise e
    os.chdir(original_directory)
    return {
        "name": name,
        "stark_proving_time": stark_proving_time_in_seconds / TIME_MEASURE_REPETITIONS,
        "snark_proving_time": snark_proving_time_in_seconds / TIME_MEASURE_REPETITIONS,
        "snark_size": snark_size,
        "verify_time": verify_time_in_seconds / TIME_MEASURE_REPETITIONS,
    }


def run_evaluate(dataset: str, problem: str):
    # Get the driver source
    with open(os.path.join('../benchmarking', dataset, problem, 'sp1.driver.rs'), 'r') as f:
        driver_source = f.read()
    # Get the program source
    with open(os.path.join('../benchmarking', dataset, problem, 'sp1.prog.rs'), 'r') as f:
        program_source = f.read()
    # Run
    return run_prove(f"{dataset}::{problem}.py", driver_source, program_source)


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
    if not os.path.exists('results-sp1.json'):
        results_dict = {}
    else:
        with open('results-sp1.json', 'r') as f:
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
                with open('results-sp1.json', 'w') as f:
                    f.write(json.dumps(results_dict, indent=2))
                raise e


    with open('results-sp1.json', 'w') as f:
        f.write(json.dumps(results_dict, indent=2))


if __name__ == '__main__':
    main()
