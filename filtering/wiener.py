import numpy as np
from scipy.linalg import toeplitz
from scipy import signal

class WienerFilterFIR:
    def __init__(self, length=int):
        self.length = length
        self.fitted = False
    
    def cross_correlation(self, x:np.ndarray, y:np.ndarray, length:int):
        assert (x.shape==y.shape), "Missmatch in dimensions of signal and noisy signal ({})!=({})".format(x.shape, y.shape)
        assert (length<=x.shape[0]), "Filter length can't be greater than the sequence signal ({}), ({})".format(length, x.shape)

        N = x.shape[0]
        r_xy = np.zeros(2*length+1,)

        mval = np.transpose(np.arange(-length,length+1))
        rxy = np.zeros(2*length+1,)
        rxy[length] = np.sum(np.multiply(x[:N], y[:N]))/N
        rxyp = np.zeros(length,)
        ryxp = np.zeros(length,)
        for i in range(length):
            rxyp[i]=np.sum(np.multiply(x[i+1:N], y[:N-i-1]))/N # solution
            ryxp[i]=np.sum(np.multiply(y[i+1:N], x[:N-i-1]))/N # solution
            rxy[:length] = np.flipud(ryxp)
            rxy[length+1:2*length+1] = rxyp
        return rxy, mval
    
    def fit(self, x:np.ndarray, y:np.ndarray, length:int=None):
        if not length:
            length = self.length
        else:
            self.length = length
        self.y = y
        r_xx, _ = self.cross_correlation(x,x,length-1)
        r_dx, _ = self.cross_correlation(y,x,length-1)
        iv = np.zeros((length, length))

        for n in range(length):
            iv[n,:] = np.array(np.arange(length-n-1, 2*length-n-1))

        iv = iv.astype(int)
        R_xx = r_xx[iv]
        w = np.linalg.solve(R_xx, r_dx[length-1:2*length-1])
        self.w = w
        self.fitted = True
        return w
    
    def MSE(self, y:np.ndarray, yhat:np.ndarray):
        assert (y.shape == yhat.shape), "Missmatch in dimension of original and reconstructed signal {y.shape}!={yhat.shape}."
        N = y.shape[0]
        return np.sum(np.power(y-yhat,2))/N
    
    def convmtx(h, n):
        ''' Convolution matrix, same as convmtx does in matlab'''
        return toeplitz(
                np.hstack([h, np.zeros(n-1)]),
                np.hstack([h[0], np.zeros(n-1)]),
                )

    def transform(self, x:np.ndarray):
        assert (self.fitted), "Weights to deconvolve are unknown, use .fit() method first."
        yhat = signal.lfilter(self.w, 1, x)
        mse = self.MSE(self.y, yhat)
        return yhat, mse

