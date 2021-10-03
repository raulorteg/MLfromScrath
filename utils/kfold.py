import numpy as np

class Kfold:
    """
    k-fold cross-validation. Given the whole dataset X (features), y (labels).
    splits the dataset randomly into K independent, equal subsets without overlap.
    Of the k subsamples, a single subsample is retained as the validation data for testing the model
    and the remaining k − 1 subsamples are used as training data. Cross-validation process is then
    repeated k times, with each of the k subsamples used exactly once as the validation data.
    The k results can then be averaged to produce a single estimation.

    :param X: numpy.ndarray containing the samples and the features of the dataset
    :type X: numpy.ndarray

    :param y: numpy.ndarray containing the labels of the dataset
    :type y: numpy.ndarray

    :param K: Number of equal-sized subsets to partition the original dataset
    :type K: int
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, K: int = 5):
        assert (X.shape[0] == y.shape[0]), "Missmatch in number of samples. X and y must math in the first dimension."
        self.X = X
        self.y = y
        self.K = K+1
        self.n_samples, self.n_features = self.X.shape
        self.__partition()

    def __partition(self):
        """ Partition the original dataset into K equal-sized datasets """
        p = np.random.permutation(self.n_samples) 
        length_partition = int(np.floor(self.n_samples/self.K))
        Xtrain, Xtest, ytrain, ytest = [], [], [], []
        for i in range(self.K):
            idxs_test = p[i*length_partition:(i+1)*length_partition]
            idxs_train = list(set(p) - set(idxs_test))
            Xtest.append(self.X[idxs_test])
            Xtrain.append(self.X[idxs_train])
            ytest.append(self.y[idxs_test])
            ytrain.append(self.y[idxs_train])
    
        self.Xtrain = np.array(Xtrain)
        self.Xtest = np.array(Xtest)
        self.ytrain = np.array(ytrain)
        self.ytest = np.array(ytest)
    
    def get_data(self):
        return self.Xtrain, self.ytrain, self.Xtest, self.ytest

def compute_accuracy(predicted, target):
    """ return fraction of predicted values matching the known targets
    :param predicted: array containing the predicted index of the classes (Multiclass not implemented)
    :type predicted: numpy.ndarray
    :param target: array containing the known index of the classes (Multiclass not implemented)
    :type target: numpy.ndarray
    """
    assert (len(predicted) == len(target)), "Predictions and Targets missmatch in the dimension."
    assert (predicted.ndim == 1), "Multiclass prediction not yet implemented, predicted values should be single values of class index"
    assert (target.ndim == 1), "Multiclass prediction not yet implemented, target values should be single values of class index"
    return (predicted == target).sum()/len(predicted)