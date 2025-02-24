# 740. Delete and Earn
# Medium
# Topics
# Companies
# Hint
#
# You are given an integer array nums. You want to maximize the number of points you get by performing the following operation any number of times:
#
#     Pick any nums[i] and delete it to earn nums[i] points. Afterwards, you must delete every element equal to nums[i] - 1 and every element equal to nums[i] + 1.
#
# Return the maximum number of points you can earn by applying the above operation some number of times.
# Constraints:
#
#     1 <= nums.length <= 2 * 10^4
#     1 <= nums[i] <= 10^3

from zinnia import *


@zk_circuit
def verify_solution(
    nums: NDArray[int, 20],
    result: int
):
    n = 1001
    values = [0] * n
    for num in nums:
        values[num] += num
    take = 0
    skip = 0
    for i in range(n):
        takei = skip + values[i]
        skipi = max(skip, take)
        take = takei
        skip = skipi
    assert result == max(take, skip)
