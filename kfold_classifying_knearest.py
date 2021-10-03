from classifying.knearest import KnearestClassifier
from data.iris import IrisDataset
from utils.kfold import Kfold, compute_accuracy

def main():
    """ Test example of classifying k_nearest and validation using k-fold """
    data = IrisDataset()
    X, y = data.get_data()

    Xtrain, ytrain, Xtest, ytest = Kfold(X=X, y=y, K=10).get_data()
    k_fold_accuraccy = []
    for i in range(len(Xtrain)):
        classifier = KnearestClassifier(X=Xtrain[i], C=ytrain[i], K=10)
        _, labels = classifier.run(Xtest[i])
        accuracy_perc = compute_accuracy(labels, ytest[i])
        k_fold_accuraccy.append(accuracy_perc)
        print(f" --- accuracy on k={i+1}: {round(accuracy_perc*100, 2)}%")
    print(f"mean k-fold accuracy: {round(sum(k_fold_accuraccy)*100/len(k_fold_accuraccy),2)}%")

if __name__ == "__main__":
    main()