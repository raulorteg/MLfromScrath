from clustering.kmeans import Kmeans
from data.dummy import Dummy

def main():
    """ Test example of clustering_kmeans """
    data = Dummy(n_samples=10000, n_dim=2)
    X = data.get_dummy()
    clustering = Kmeans(X, K=10, display=False)
    clustering.run()
    print(f"Number of iterations: {clustering.num_iterations}")

if __name__ == "__main__":
    main()