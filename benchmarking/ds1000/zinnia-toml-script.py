import sys
import os
import json


def recursively_convert_json_integers_to_strings(obj):
    if isinstance(obj, dict):
        return {k: recursively_convert_json_integers_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursively_convert_json_integers_to_strings(elem) for elem in obj]
    elif isinstance(obj, int):
        return str(obj)
    else:
        return obj


if __name__ == "__main__":
    all_cases = os.listdir(os.getcwd())
    all_cases = [case for case in all_cases if case.startswith("case")]
    noir_cases = []
    for case in all_cases:
        if os.path.exists(f"{case}/Prover.toml"):
            noir_cases.append(case)
            with open(f"{case}/sol.py.in", "r") as f:
                data_obj = json.load(f)
            with open(f"{case}/sol.py.in", "w") as f:
                json.dump(recursively_convert_json_integers_to_strings(data_obj), f)
            with open(f"{case}/Prover.zinnia.toml", "w") as f:
                for key, value in data_obj.items():
                    f.write(f'{key} = "{value}"\n')
            print(f"Generated {case}/Prover.zinnia.toml")
    print('No. of DS1000', len(all_cases))
    print(sorted(all_cases))
    print(sorted(noir_cases))