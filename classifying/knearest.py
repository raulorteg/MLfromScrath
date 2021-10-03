import numpy as np
from copy import deepcopy

class KnearestClassifier:
    """
    algorithm:
    1. Take {x^(1), x^(2), ..., x^(m)} d-dimensional samples with known labels {c^(1), c^(2), ...}, X_known
    2. Take {x^(1), x^(2), ..., x^(n)}, d-dimensional samples with unknown label X_unknown
    3. Given the number K of neirest neighbors to look at.
    4. Classify the X_unkown samples with the class with highest frequency across k-Nearest neighbors from X_known
    """
    def __init__(self, X, C, K=3, display: bool = False):
        """
        :param X: array of M d-dimensional samples of known class
        :type X: numpy.ndarray
        :param C: array cotaining the classes for the X samples (1d)
        :type C: numpy.ndarray
        :param K: number of nearest neighbours to look at
        :type K: int
        """
        assert (K < X.shape[0]), "Can't look at more neighbors than samples are in the labeled data."
        assert (K > 0), "Number of nearest neighbors must be postive integer."
        assert (C.shape[0] == X.shape[0]), "All samples must be labeled. X and C must math in the dimension 0."
        assert (C.ndim == 1), "Multiclass classification is not supported."
        self.X = X
        self.C = C
        self.num_K = K
        self.display = display
        self.n_samples, self.n_dim = X.shape
        self.images_buffer = []

    def run(self, X_unknown: np.ndarray, K: int = None):
        """ Classify the unlabeled data using the labeled data using the K nearest neighbors, 
        if K is missing use the one defined in the constructor.
        :param X_unknown: array of samples of unknown label class.
        :type X_unknown: numpy.array
        :param K: number of nearest neighbours to look at. (Default None). If None then use the K defined in the constructor method.
        :type K: int
        """
        if K:
            self.num_K = K
        
        assigned_classes = np.empty(X_unknown.shape[0])
        for i, unlabeled_sample in enumerate(X_unknown):
            c_i = self.__compute_label(unlabeled_sample)
            assigned_classes[i] = c_i
        return X_unknown, assigned_classes
        
    def __compute_label(self, sample):
        """ compute euclidean distance between uunlabeled samples (sample) and labeled samples in d-dimensional space
        :param sample: x^(i) sample from X_unknown. x^(i) is a d-dimensional array.
        :type sample: numpy.ndarray
        """
        distances = np.empty(self.n_samples)
        for i, labeled_sample in enumerate(self.X):
            distances[i] = np.linalg.norm(sample - labeled_sample)
        idxs = np.argpartition(distances, self.num_K)[:self.num_K]
        unique, counts = np.unique(self.C[idxs], return_counts=True)
        class_idx = np.argmax(counts)
        return unique[class_idx]