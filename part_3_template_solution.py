import numpy as np
from numpy.typing import NDArray
from typing import Any
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(uniq, counts))
        num_classes = len(uniq)
        total_count = np.sum(counts)
        
        print(f"Unique Classes: {uniq}")
        print(f"Counts per Class: {counts}")
        print(f"Total Count of Elements: {total_count}")

        return {
            "class_counts": class_counts,  # Replace with actual class counts
            "num_classes": num_classes,  # Replace with the actual number of classes
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """
        # Enter code and return the `answer`` dictionary

        answer = {}

        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """

        clf = LogisticRegression(max_iter=1000)  # Instantiate the logistic regression classifier

        # Train the classifier
        clf.fit(Xtrain, ytrain)

        # Initialize lists to store top-k accuracy scores
        topk_train_scores = []
        topk_test_scores = []

        # Calculate top-k accuracy scores for k=1,2,3,4,5
        for k in range(1, 6):
            # Calculate top-k accuracy scores for training set
            topk_train_score = self.calculate_topk_accuracy(clf, Xtrain, ytrain, k)
            topk_train_scores.append((k, topk_train_score))

            # Calculate top-k accuracy scores for testing set
            topk_test_score = self.calculate_topk_accuracy(clf, Xtest, ytest, k)
            topk_test_scores.append((k, topk_test_score))

        # Plot k vs. score for both training and testing data
        self.plot_topk_accuracy(topk_train_scores, "Training")
        self.plot_topk_accuracy(topk_test_scores, "Testing")

        # Comment on the rate of accuracy change
        text_rate_accuracy_change = "The rate of accuracy change for testing data seems to decrease as k increases."

        # Comment on the usefulness of this metric for the dataset
        text_is_topk_useful_and_why = "Top-k accuracy is useful for this dataset as it provides insights into how well the model performs when considering multiple possible predictions. This is especially relevant for applications where the exact prediction might not be as critical as having a set of likely predictions."

        # Fill the answer dictionary
        answer["clf"] = clf
        answer["plot_k_vs_score_train"] = topk_train_scores
        answer["plot_k_vs_score_test"] = topk_test_scores
        answer["text_rate_accuracy_change"] = text_rate_accuracy_change
        answer["text_is_topk_useful_and_why"] = text_is_topk_useful_and_why

        return answer, Xtrain, ytrain, Xtest, ytest

    def calculate_topk_accuracy(
        self,
        clf: Any,  # classifier object
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        k: int
    ) -> float:
        """
        Calculate top-k accuracy score.

        Parameters:
        - clf: The classifier object.
        - X: Input data.
        - y: True labels.
        - k: Value of k for top-k accuracy.

        Returns:
        - float: Top-k accuracy score.
        """
        # Predict probabilities
        probs = clf.predict_proba(X)

        # Get top-k predictions
        topk_preds = np.argsort(-probs, axis=1)[:, :k]

        # Check if true labels are in top-k predictions
        correct = np.array([y[i] in topk_preds[i] for i in range(len(y))])

        # Calculate top-k accuracy score
        topk_accuracy = np.mean(correct)
        return topk_accuracy
    
    def plot_topk_accuracy(self, scores: list[tuple[int, float]], title: str):
        """
        Plot k vs. top-k accuracy scores.

        Parameters:
        - scores: List of tuples (k, score).
        - title: Title of the plot.
        """
        ks, accuracies = zip(*scores)
        plt.plot(ks, accuracies, marker='o', linestyle='-')
        plt.xlabel('k')
        plt.ylabel('Top-k Accuracy')
        plt.title(f'{title} Top-k Accuracy')
        plt.show()

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}

        # Answer is a dictionary with the same keys as part 1.B

        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""

        # Enter your code and fill the `answer` dictionary
        answer = {}

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        
        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer
