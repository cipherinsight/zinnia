def kernel(x):
    windows = np.lib.stride_tricks.sliding_window_view(x, (2,2), axis=(1,2))
    return windows[:,::2,::2,:,:,:].sum(axis=(-1,-2))
