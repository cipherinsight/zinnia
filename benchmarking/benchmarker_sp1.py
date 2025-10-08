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
        stark_verify_time_in_seconds = 0
        snark_verify_time_in_seconds = 0
        snark_size = 0
        stark_size = 0
        for i in range(TIME_MEASURE_REPETITIONS):
            prove_process = subprocess.run(['cargo', 'run', '--release', '--', '--prove'], capture_output=True, text=True, env=my_env)
            prove_process_feedback = prove_process.stdout + prove_process.stderr
            assert prove_process.returncode == 0, prove_process_feedback
            match = re.search(r"Prove time \(zk-STARK\) \(ms\): \s*([\d\.]+)", prove_process_feedback)
            assert match
            proving_time_stark = float(match.group(1))
            match = re.search(r"Prove time \(zk-SNARK\) \(ms\): \s*([\d\.]+)", prove_process_feedback)
            assert match
            proving_time_snark = float(match.group(1))
            match = re.search(r"Verify time \(zk-STARK\) \(ms\): \s*([\d\.]+)", prove_process_feedback)
            assert match
            verify_time_stark = float(match.group(1))
            match = re.search(r"Verify time \(zk-SNARK\) \(ms\): \s*([\d\.]+)", prove_process_feedback)
            assert match
            verify_time_snark = float(match.group(1))
            stark_proving_time_in_seconds += proving_time_stark / 1000
            snark_proving_time_in_seconds += proving_time_snark / 1000
            stark_verify_time_in_seconds += verify_time_stark / 1000
            snark_verify_time_in_seconds += verify_time_snark / 1000
            snark_size = os.path.getsize(os.path.join(SP1_FOLDER, "proof-with-pis.bin"))
            stark_size = os.path.getsize(os.path.join(SP1_FOLDER, "proof-with-pis-stark.bin"))
    except Exception as e:
        os.chdir(original_directory)
        raise e
    os.chdir(original_directory)
    return {
        "name": name,
        "stark_proving_time": stark_proving_time_in_seconds / TIME_MEASURE_REPETITIONS,
        "snark_proving_time": snark_proving_time_in_seconds / TIME_MEASURE_REPETITIONS,
        "snark_size": snark_size,
        "stark_size": stark_size,
        "stark_verify_time": stark_verify_time_in_seconds / TIME_MEASURE_REPETITIONS,
        "snark_verify_time": snark_verify_time_in_seconds / TIME_MEASURE_REPETITIONS,
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

    with open('results-sp1.json', 'w') as f:
        f.write(json.dumps(results_dict, indent=2))


if __name__ == '__main__':
    main()
