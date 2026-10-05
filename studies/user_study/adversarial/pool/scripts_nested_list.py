def kernel(x):
    b,h,w,c = x.shape
    return np.array([[[[sum(x[n,2*i+di,2*j+dj,k] for di in range(2) for dj in range(2)) for k in range(c)] for j in range(w//2)] for i in range(h//2)] for n in range(b)])
