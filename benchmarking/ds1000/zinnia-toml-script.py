import sys
import os
import json


if __name__ == "__main__":
    all_cases = os.listdir(os.getcwd())
    all_cases = [case for case in all_cases if case.startswith("case")]
    noir_cases = []
    for case in all_cases:
        if os.path.exists(f"{case}/Prover.toml"):
            noir_cases.append(case)
            with open(f"{case}/sol.py.in", "r") as f:
                data_obj = json.load(f)
            with open(f"{case}/Prover.zinnia.toml", "w") as f:
                for key, value in data_obj.items():
                    f.write(f'{key} = "{value}"\n')
            print(f"Generated {case}/Prover.zinnia.toml")
    print('No. of DS1000', len(all_cases))
    print(sorted(all_cases))
    print(sorted(noir_cases))