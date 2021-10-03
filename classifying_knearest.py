from classifying.knearest import KnearestClassifier
from data.iris import IrisDataset

def main():
    """ Test example of classifying k_nearest """
    data = IrisDataset()
    X, y = data.get_data()
    classifier = KnearestClassifier(X=X, C=y, K=3)
    unlabeled_data = data.synthetic_data(n_samples=5)
    _, labels = classifier.run(unlabeled_data, 5)
    print(f"Predicted labels: {labels} with k=5 neighbors")

if __name__ == "__main__":
    main()