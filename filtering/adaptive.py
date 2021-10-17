import numpy as np
from scipy.linalg import toeplitz

class AdaptiveFilter:
    def __init__(self, length:int=None, method:str="nlms"):
        assert (method.lower() in ["nlms", "lms", "rls"]), "Method chosen not supported, try method='nlms', 'lms', 'rls'."
        self.method = method.lower()
        self.length = length
    
    def fit_transform(self, x:np.ndarray, y:np.ndarray, L:int, mu:float=None, delta:float=None, beta:float=None, lambda_:float=None, method:str=None):
        assert (x.shape == y.shape), "Missmatch in the dimensions of input signal x, and desired signal y. {}!={}".format(x.shape, y.shape)
        if not method:
            method = self.method
        else:
            self.method = method.lower()
            method = method.lower()
            assert (method.lower() in ["nlms", "lms", "rls"]), "Method chosen not supported, try method='nlms', 'lms', 'rls'."

        if method == "nlms":
            assert (mu) and (L) and (delta), "Missing value for step_size/length_filter/stability_parameter: {},{},{}".format(mu, L, delta)
            yhat, mse = self.nlms(x=x, y=y, L=L, mu=mu, delta=delta)
            return yhat, mse
        
        elif method == "lms":
            assert (mu) and (L), "Missing value for step_size/length_filter: {},{},{}".format(mu, L)
            yhat, mse = self.lms(x=x, y=y, L=L, mu=mu)
            return yhat, mse
        
        elif method == "rls":
            assert (L) and (beta) and (lambda_), "Missing value for length_filter/forgetting_factor/regularization_parameter: {},{},{}".format(L, beta, lambda_)
            yhat, mse = self.rls(x=x, y=y, L=L, beta=beta, lambda_=lambda_)
            return yhat, mse
    
    def convmtx(self, h, n):
        ''' Convolution matrix, same as convmtx does in matlab'''
        return toeplitz(
                np.hstack([h, np.zeros(n-1)]),
                np.hstack([h[0], np.zeros(n-1)]),
                )

    def nlms(self, x:np.ndarray, y:np.ndarray, L:int, mu:float, delta:float):
        ''' 
        :param x: input signal
        :type x: numpy.ndarray
        :param y: desired signal
        :type y: numpy.ndarray
        :param L: filter length
        :type L: int
        :param mu: step size
        :type mu: float
        :param delta: small stability value
        :type delta: float
        :return yhat: filter output
        :type yhat: numpy.ndarray
        '''
        yhat = np.zeros(len(y))
        e = np.zeros(len(y))
        # if x is a one-dimensional vector we assume x is a time series and we
        # create X accordingly using X = convmtx(x, L), otherwise we assume it
        # already is a convolution matrix
        if x.ndim == 1:
            X = self.convmtx(x, L)
        else:
            X = x.T
        # start NLMS
        w = np.zeros(L)
        for n in range(len(y)):
            x_n = X[n, :].T
            yhat[n] = w.T@x_n
            e[n] = y[n] - yhat[n]
            w = w + mu/(delta + np.dot(x_n, x_n))*x_n*e[n]
        mse = e**2
        return yhat, mse

    def lms(self, x:np.ndarray, y:np.ndarray, L:int, mu:float):
        ''' 
        :param x: input signal
        :type x: numpy.ndarray
        :param y: desired signal
        :type y: numpy.ndarray
        :param L: filter length
        :type L: int
        :param mu: step size
        :type mu: float
        :return yhat: filter output
        :type yhat: numpy.ndarray
        '''
        yhat = np.zeros(len(y))
        e = np.zeros(len(y))
        X = self.convmtx(x, L)
        # start LMS
        w = np.zeros(L)
        for n in range(len(y)):
            x_n = np.transpose(X[n, :])
            yhat[n] = np.dot(w, x_n)
            e[n] = np.subtract(y[n], yhat[n])
            w = w + mu*e[n]*x_n
        mse = e**2
        return yhat, mse

    def rls(self, x:np.ndarray, y:np.ndarray, L:int, beta:float, lambda_:float):
        '''
        :param x: input signal
        :type x: numpy.ndarray
        :param y: desired signal
        :type y: numpy.ndarray
        :param L: filter length
        :type L: int
        :param beta: forget factor
        :type beta: float
        :param lambda_: regularization parameter
        :type lambda_: float
        :return yhat: filter output
        :type yhat: numpy.ndarray
        '''
        yhat = np.zeros(len(y))
        e = np.zeros(len(y))
        # if x is a one-dimensional vector we assume x is a time series and we
        # create X accordingly using X = convmtx(x, L), otherwise we assume it
        # already is a convolution matrix
        if x.ndim == 1:
            X = self.convmtx(x, L)
        else:
            X = x.T
        # start RLS
        w = np.zeros(L)
        P = 1/lambda_*np.eye(L)
        for n in range(len(y)):
            x_n = X[n, :].T
            yhat[n] = w.T@x_n
            e[n] = y[n] - yhat[n]
            denum = 1 / (beta + x_n.T@P@x_n)
            K_n = P@x_n*denum
            w = w + K_n*e[n]
        P = (P - (np.outer(K_n, x_n.T))@P)/beta
        mse = e**2
        return yhat, mse