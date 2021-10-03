import numpy as np
import pandas as pd

class IrisDataset:
    """
    The classic Iris Dataset, this class formats the dataset to be able to use it easyly.
    the dataset .csv file was downloaded from  https://gist.github.com/netj/8836201.
    :param one_hot: boolean to set up mode to convert variety types into one_hot vectors (TODO:)
    :type one_hot: bool
    """
    def __init__(self, one_hot: bool = False, shuffle: bool = True):
        self.one_hot = one_hot
        self.shuffle = shuffle

        self.data = pd.read_csv("data/iris.csv")
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)

        self.features_names = self.data.columns.to_list()
        self.X = self.data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].to_numpy()
        self.n_samples, self.n_dim = self.X.shape
        self.bounds = np.array(list(zip(self.X.min(axis=0), self.X.max(axis=0))))
        self.y = self.data['variety']

        if self.one_hot:
            raise NotImplementedError('One hot encoding for iris variety types not yet implemented.')
        else:
            self.__generate_dicts()
            self.y = self.y.apply(lambda x: self.variety_to_idx[x]).to_numpy()
        
    def __generate_dicts(self):
        """ Create the dictionaries used to translate from variety string to idx and vice-versa"""
        self.variety_to_idx = {'Setosa':0, 'Versicolor':1, 'Virginica':2}
        self.idx_to_variety = {0: 'Setosa', 1: 'Versicolor', 2:'Virginica'}

    def variety_to_idx(self, variety):
        return self.variety_to_idx[variety]
    
    def idx_to_variety(self, idx):
        return self.idx_to_variety[idx]
    
    def synthetic_data(self, n_samples=100):
        """ Generate new unlabeled data within the bounds of the Iris dataset """
        synthetic = np.empty((n_samples, self.n_dim))
        for d in range(self.n_dim):
            min_, max_ = self.bounds[d]
            synthetic[:,d] = np.random.uniform(min_, max_, n_samples)
        return synthetic
    
    def get_data(self):
        return self.X, self.y