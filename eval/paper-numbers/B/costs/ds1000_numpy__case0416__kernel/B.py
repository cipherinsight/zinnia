@zk_circuit
def kernel(data: Float[Array, '8 6'], bin_size: Integer):
    assert bin_size == 2
    bin_size = 2
    bin_data_mean = data[:, :data.shape[1] // bin_size * bin_size].reshape(data.shape[0], -1, bin_size).mean(axis=-1)
    return bin_data_mean