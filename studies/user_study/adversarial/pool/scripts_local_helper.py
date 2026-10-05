def kernel(x):
    def add_pairs(a):
        return a[:,::2,:,:] + a[:,1::2,:,:]
    rows = add_pairs(x)
    return rows[:,:,::2,:] + rows[:,:,1::2,:]
