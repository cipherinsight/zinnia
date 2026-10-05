def kernel(x):
    y=2*x+5
    return (255+np.abs(y)-np.abs(y-255))//2
