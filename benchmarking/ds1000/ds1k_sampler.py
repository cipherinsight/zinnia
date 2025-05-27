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
    return True


# filter out all the examples that have the library as 'numpy'
ds1000 = [
    ex for ex in ds1000 if filter_test(ex)
]
print('Total numbers of filtered cases:', len(ds1000))

ds1000 = random.sample(ds1000, 10)

for i in range(10):
    print("No.", ds1000[i]['index'])
    print(ds1000[i]['prompt'])
    print('---' * 20)
    print(ds1000[i]['reference_code'])
    print('===' * 20)
