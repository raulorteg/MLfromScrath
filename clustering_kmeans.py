from clustering.kmeans import Kmeans
from data.dummy import Dummy

def main():
    """ Test example of clustering_kmeans with given number of clusters K """
    data = Dummy(n_samples=500, n_dim=3)
    X = data.get_dummy()
    clustering = Kmeans(X, K=5, display=False)
    clustering.run()
    print(f"Number of iterations: {clustering.num_iterations}\n")

    """ Test example of clustering_kmeans with unknown number of clusters K """
    clustering = Kmeans(X,)
    clustering.silhouette_find_k()
    print(f"Number of centroids found: {clustering.num_K}")

if __name__ == "__main__":
    main()