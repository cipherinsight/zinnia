# 204. Count Primes
# Medium
# Topics
# Companies
# Hint
#
# Given an integer n, return the number of prime numbers that are strictly less than n.
# Constraints:
#     0 <= n <= 5 * 10^6
import json

from zinnia import *


@zk_circuit
def verify_solution(n: int, result: int):
    is_prime = [1] * 1001
    number_of_primes = 0
    assert 1000 >= n >= 0
    if n == 1 or n == 0:
        assert result == 0
    else:
        for i in range(2, 1001):
            if is_prime[i] == 1:
                number_of_primes += 1
                for j in range(i, 1001, i):
                    is_prime[j] = 0
            assert i != n or number_of_primes == result


# assert verify_solution(1000, 168)

# Compile and get the source code
# zinnia_config = ZinniaConfig(backend=ZinniaConfig.BACKEND_HALO2)
# program = ZKCircuit.from_method(verify_solution, config=zinnia_config).compile()
# halo2_source_code = program.source
# print(halo2_source_code)

# # Parse inputs
# inputs = (1000, 168)
# parsed_inputs = program.argparse(*inputs)
# json_dict = {}
# for entry in parsed_inputs.entries:
#     json_dict[entry.get_key()] = entry.get_value()
# print(json.dumps(json_dict, indent=2))
#
# # Mock Prove (Optional)
# mock_executor = MockProgramExecutor(program.get_execution_context(), program, zinnia_config)
# mock_result = mock_executor.exec(*inputs)
# assert mock_result
