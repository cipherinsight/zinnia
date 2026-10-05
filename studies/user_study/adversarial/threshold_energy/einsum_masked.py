def kernel(x, floor):
    active = np.where(x > floor, x, 0)
    return np.einsum('nc,nc->n', active, active)
