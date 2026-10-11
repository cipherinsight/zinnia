@zk_circuit
def kernel(data: Float[Array, '8'], bin_size: Integer):
    assert bin_size == 4
    bin_size = 4
    new_data = data[::-1]
    bin_data_mean = new_data[:data.size // bin_size * bin_size].reshape(-1, bin_size).mean(axis=1)
    return bin_data_mean