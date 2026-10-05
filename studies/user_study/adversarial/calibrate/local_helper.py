def kernel(x):
    def clamp(v):
        if v < 0:
            return 0
        if v > 255:
            return 255
        return v
    return np.array([clamp(2*v+5) for v in x])
