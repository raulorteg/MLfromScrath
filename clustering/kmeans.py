import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from PIL import Image
import imageio
import io

class Kmeans:
    """
    algorithm:
    1. Take {x^(1), x^(2), ..., x^(m)} d-dimensional samples, and K number of clusters
    2. Randomly initialize K cluster centroids mu_1, mu_2, ..., mu_k in d-dimensional space
    3. do while (centroids dont change):
        3.1 Assign each sample to a centroid based on <criteria> (e.g closest centroid)
        3.2 Move centroids to the mean position of the samples assigned to each centroid
    end while
    """
    def __init__(self, X, K=3, display: bool = False):
        """
        :param X: array of M d-dimensional samples
        :type X: numpy.ndarray
        :param K: number of centroids (clusters)
        :type K: int
        """
        assert (K < X.shape[0]), "Can't have more centroids than samples."
        assert (K > 0), "Number of centroids must be postive integer."
        self.X = X
        self.num_K = K
        self.display = display
        self.n_samples, self.n_dim = X.shape
        self.bounds = np.array(list(zip(X.min(axis=0), X.max(axis=0))))
        self.converged = False
        self.classes = None
        self.num_iterations = 0
        self.images_buffer = []
        self.__init_K()

    def __init_K(self):
        """ init the position of the centroids """
        self.mu_k = np.empty((self.num_K, self.n_dim))
        for d in range(self.n_dim):
            min_, max_ = self.bounds[d]
            self.mu_k[:, d] = np.random.uniform(low=min_, high=max_, size=self.num_K)
    
    def __d_euclidean_distance(self, sample):
        """ compute euclidean distance between sample and centroids in d-dimensional space
        :param sample: x^(i) sample from X. x^(i) is a d-dimensional array.
        :type sample: numpy.ndarray
        """
        distances = np.empty(self.num_K)
        for i, centroid in enumerate(self.mu_k):
            distances[i] = np.linalg.norm(sample - centroid)
        c_idx = np.argmin(distances)
        return c_idx, distances[c_idx]
    
    def __assign_centroids(self):
        """ given the samples X return the assigned index of the centroids that
        each sample is closest to """
        assignations  = np.empty(self.n_samples)
        overall_cost = 0.0
        for sample_idx, sample in enumerate(self.X):
            c_idx, c_value = self.__d_euclidean_distance(sample=sample)
            assignations[sample_idx] = c_idx
            overall_cost += c_value
        return assignations, overall_cost/self.n_samples
    
    def __move_centroids(self, assignations):
        """ Using the samples assigned to each centroid update the coordinates
        of the centroid to be the mean of the samples assigned to it.
        :param assigned: Index of assigned centroids to each sample point
        :type assigned: numpy.ndarray """
        mu_k = np.empty((self.num_K, self.n_dim))
        counts_assigned = np.empty(self.num_K)
        for c_idx in range(self.num_K):
            sample_idxs = np.where(assignations == c_idx)
            counts_assigned[c_idx] = len(sample_idxs)
            mu_k[c_idx] = self.X[sample_idxs].mean(axis=0)
        return mu_k, counts_assigned

    def __single_iteration(self):
        """ Perform a single iteration on the kmeans algorithm """
        if self.num_iterations == 0:
            self.num_iterations += 1
            assignations, overall_cost = self.__assign_centroids()
            mu_k, counts_assigned = self.__move_centroids(assignations)

            if self.display:
                if self.n_dim == 2:
                    self.images_buffer.append(self.__display_2d(self.X, assignations, overall_cost))
                else:
                    # coming when pca is done
                    raise NotImplementedError("Display is currently only Implemented for 2d Data. Upcoming changes!")
        
        else:
            self.num_iterations += 1
            assignations, overall_cost = self.__assign_centroids()
            self.mu_k, counts_assigned = self.__move_centroids(assignations)

            if self.display:
                if self.n_dim == 2:
                    self.images_buffer.append(self.__display_2d(self.X, assignations, overall_cost))
                else:
                    # Coming when pca is done
                    raise NotImplementedError("Display is currently only Implemented for 2d Data. Upcoming changes!")
        return assignations, counts_assigned

    def run(self):
        """ Execute the kmeans algorithm """
        while not self.converged:
            mu_k_old = deepcopy(self.mu_k)
            assignations, _ = self.__single_iteration()
            if (mu_k_old == self.mu_k).all() and (self.num_iterations > 1):
                self.converged = True

        self.classes = assignations

        if self.display:
            self.__generate_gif()
    
    def __display_2d(self, X, assignations, overall_cost):
        """ Create the .png figure to be used in the .gif visualization
        :param X: array of samples in 2d space
        :type X: numpy.ndarray
        :param assignations: Index of assigned centroids to each sample point
        :type assignations: numpy.ndarray
        :param overall_cost: sum of the minimum distance to all centroids,
        normalized divided by the total number of samples
        :type overall_cost: float
        """
        color = cm.rainbow(np.linspace(0, 1, self.num_K))
        fig, ax = plt.subplots(figsize=(10,5))
        for c_idx in range(self.num_K):
            sample_idxs = np.where(assignations == c_idx)
            ax.scatter(X[sample_idxs][:,0], X[sample_idxs][:,1], color=color[c_idx], marker=".", label=c_idx)
            ax.plot(self.mu_k[c_idx][0], self.mu_k[c_idx][1], color=color[c_idx], marker="^", label=c_idx)
        ax.set(xlabel='x_{1}', ylabel='x_{2}', title='2d Clustering Plot (iter={})'.format(self.num_iterations))

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img
    
    def __generate_gif(self):
        """ generate .gif from the buffer of images if display flag is true. """
        kwargs_write = {'fps':1.0, 'quantizer':'nq'}
        imageio.mimsave('kmeans.gif', self.images_buffer, fps=1)

    def get_centroids(self):
        return self.mu_k
    
    def get_labeled_data(self):
        return self.X, self.classes