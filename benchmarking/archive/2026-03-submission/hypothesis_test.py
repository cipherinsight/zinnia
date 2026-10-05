import numpy as np
import scipy.stats as stats

group_a_data = [[15, 8, 18], [12, 6, 25], [5, 7, 10]]
group_b_data = [[5, 13, 45], [16, 13, 60], [11, 15, 60]]
group_a_data = np.asarray(group_a_data)
group_b_data = np.asarray(group_b_data)

# use 60 minus the time taken to complete the task
# group_a_data = [100 - row for row in group_a_data]
# group_b_data = [100 - row for row in group_b_data]
group_a_data = [sum(row) for row in group_a_data]
group_b_data = [sum(row) for row in group_b_data]
print(group_a_data)
print(group_b_data)

u_statistic, p_value = stats.mannwhitneyu(group_a_data, group_b_data, alternative='less')

print("Mann-Whitney U Statistic:", u_statistic)
print("P-value:", p_value)

alpha = 0.05

if p_value <= alpha:
    print("Reject the null hypothesis. Group A performs significantly better.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference.")
