"""A 12-person, six-task incomplete crossover: each task is performed once."""
from collections import Counter

DESIGN_ID = 'crossover-12x6-v1'

def schedules(tasks):
    if len(tasks)!=6 or len(set(tasks))!=6:
        raise ValueError('The crossover requires six distinct tasks')
    # Even-order Williams square: every task occupies every position once;
    # every ordered pair of distinct tasks is adjacent once in the six rows.
    first=[0,1,5,2,4,3]
    base=['zinnia','zinnia','rust','zinnia','rust','rust']
    rows=[]
    for row in range(6):
        order=[tasks[(index+row)%6] for index in first]
        sequence=base[row:]+base[:row]
        for opposite in (False,True):
            languages=[('rust' if c=='zinnia' else 'zinnia') if opposite else c for c in sequence]
            training_first=languages[0] if row%2==0 else ('rust' if languages[0]=='zinnia' else 'zinnia')
            rows.append({'slot':len(rows)+1,'order':order,'task_conditions':dict(zip(order,languages)),
                         'practice_order':[training_first,'rust' if training_first=='zinnia' else 'zinnia']})
    assert all(Counter(r['task_conditions'].values())=={'zinnia':3,'rust':3} for r in rows)
    return rows
