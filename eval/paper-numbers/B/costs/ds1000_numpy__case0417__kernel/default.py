@zk_circuit
def kernel(data: Float[Array, '8'], bin_size: Integer):
    new_data = data[::-1]
    bin_data_mean = new_data[:data.size // bin_size * bin_size].reshape(-1, bin_size).mean(axis=1)
    return bin_data_mean