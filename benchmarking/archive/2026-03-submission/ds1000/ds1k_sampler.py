import random

from datasets import load_dataset

ds1000 = list(load_dataset("xlangai/DS-1000")["test"])
# add an index to each example
for i, _ex in enumerate(ds1000):
    _ex['index'] = i


def filter_test(ex) -> bool:
    if ex['metadata']['library'] != 'Numpy':
        return False
    if 'import scipy' in ex['reference_code'] + ex['prompt']:
        return False
    if 'from scipy' in ex['reference_code'] + ex['prompt']:
        return False
    if 'import pandas' in ex['reference_code'] + ex['prompt']:
        return False
    if 'from pandas' in ex['reference_code'] + ex['prompt']:
        return False
    if 'import sklearn' in ex['reference_code'] + ex['prompt']:
        return False
    if 'from sklearn' in ex['reference_code'] + ex['prompt']:
        return False
    if 'random' in ex['reference_code'] + ex['prompt']:
        return False
    if 'string' in ex['reference_code'] + ex['prompt']:
        return False
    if 'unravel' in ex['reference_code'] + ex['prompt']:
        return False
    if 'nan' in (ex['reference_code'] + ex['prompt']).lower():
        return False
    if 'inf' in (ex['reference_code'] + ex['prompt']).lower():
        return False
    if '"' in (ex['reference_code']).lower():
        return False
    if "'" in (ex['reference_code']).lower():
        return False
    if "np.linalg" in (ex['reference_code'] + ex['prompt']).lower():
        return False
    if "np.lib" in (ex['reference_code'] + ex['prompt']).lower():
        return False
    if "np.polyfit" in (ex['reference_code'] + ex['prompt']).lower():
        return False
    if "try:" in (ex['reference_code']).lower():
        return False
    if ".imag" in (ex['reference_code']).lower():
        return False
    if "<<" in (ex['reference_code']).lower():
        return False
    if ">>" in (ex['reference_code']).lower():
        return False
    return True


# filter out all the examples that have the library as 'numpy'
ds1000 = [
    ex for ex in ds1000 if filter_test(ex)
]
print('Total numbers of filtered cases:', len(ds1000))
filtered_cases = []
trivial_cases = [291, 451, 400, 401, 317, 360, 361, 359, 362, 363, 364, 365, 366, 367, 377, 378, 379, 380, 500]

for i in range(len(ds1000)):
    prompt = ds1000[i]['prompt']
    reference_code = ds1000[i]['reference_code']
    filtered_cases.append({'prompt': prompt, 'reference_code': reference_code, 'index': ds1000[i]['index']})

# save the filtered cases to a markdown file for easy viewing
with open('./ds1k_sampled.md', 'w') as f:
    for i, case in enumerate(filtered_cases):
        if case['index'] in trivial_cases:
            continue
        f.write(f"### Case {i} (Index: {case['index']})\n")
        f.write("**Prompt:**\n")
        f.write("```\n")
        f.write(case['prompt'] + '\n')
        f.write("```\n")
        f.write("**Reference Code:**\n")
        f.write("```\n")
        f.write(case['reference_code'] + '\n')
        f.write("```\n\n")

