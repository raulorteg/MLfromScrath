import numpy as np

class Dummy:
    """ Generates <m> dummy samples {x^(1), x^(2), ..., x^(m)} of dimension <d>
    each from within the bounds for each dimension
    """
    def __init__(self, n_samples=50, n_dim=2, bounds=(0,1)):
        """
        :param n_samples: number of <m> dummy samples to generate.
        :type n_samples: int
        :param n_dim: dimension <d> of the <m> dummy samples.
        :type n_dim: int
        :param bounds: minimum and maximum values for every dimension.
        """
        self.n_samples = n_samples
        self.n_dim = n_dim
        self.bounds = bounds
        self.__generate()
    
    def __generate(self):
        """ Generate the random dummy data from uniform distribution """
        self.X = np.empty((self.n_samples, self.n_dim))
        if isinstance(self.bounds, tuple):
            self.X = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_samples, self.n_dim))
        else:
            for i, bounds_i in enumerate(self.bounds):
                self.X[i] = np.random.uniform(bounds_i[0], bounds_i[1], self.n_samples)
    
    def get_dummy(self):
        return self.X