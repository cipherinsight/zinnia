def kernel(a: Integer[Array, "5 6"], key: Integer[Array, "6"]):
    p = np.argsort(key, kind="stable")
    c = np.empty_like(p)
    c[p] = np.arange(6)
    im = a[:, c]
    rows = np.flatnonzero(im.any(axis=1))
    cols = np.flatnonzero(im.any(axis=0))
    return im[rows[0]:rows[-1]+1,
              cols[0]:cols[-1]+1]
