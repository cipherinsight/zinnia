import random

from datasets import load_dataset

ds1000 = list(load_dataset("xlangai/DS-1000")["test"])


def filter_test(ex) -> bool:
    if ex['metadata']['library'] != 'Numpy':
        return False
    if 'import scipy' in ex['reference_code']:
        return False
    if 'from scipy' in ex['reference_code']:
        return False
    if 'import sklearn' in ex['reference_code']:
        return False
    if 'from sklearn' in ex['reference_code']:
        return False
    return True


# filter out all the examples that have the library as 'numpy'
ds1000 = [
    ex for ex in ds1000 if filter_test(ex)
]
print(len(ds1000))

# randomly sample 10 examples with the seed
random.seed(0)
ds1000 = random.sample(ds1000, 10)

for i in range(10):
    print(ds1000[i]['prompt'])
    print('---' * 20)
    print(ds1000[i]['reference_code'])
    print('===' * 20)
