def kernel(x, y):
    xx = np.einsum('id,id->i', x, x)
    yy = np.einsum('jd,jd->j', y, y)
    xy = np.einsum('id,jd->ij', x, y)
    return xx[:, None] + yy[None, :] - 2 * xy
