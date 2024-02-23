import numpy as np
from numpy.typing import NDArray
from typing import Any
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC

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
        
        uniq, counts = np.unique(y, return_counts=True)
        
        #UNIQUE CLASSES AND COUNTS
        print(f"Unique Classes: {uniq}")
        print(f"Counts per class: {counts}")
        print(f"Total count of classes: {np.sum(counts)}")
        
        #CONVERTING COUNTS TO DICTIONARY - KEYS ARE UNIQUE CLASSES
        class_counts = dict(zip(uniq, counts))

        return {
            "class_counts": class_counts,
            "num_classes": len9(uniq),
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
        
        answer = {}

        clf = self.train_classifier(Xtrain, ytrain)
        
        plot_k_vs_score_train = []
        plot_k_vs_score_test = []
        
        k_values = [1, 2, 3, 4, 5]
        
        for k in k_values:
            score_train = self.top_k_accuracy(clf, Xtrain, ytrain, k)
            score_test = self.top_k_accuracy(clf, Xtest, ytest, k)
            
            plot_k_vs_score_train.append((k, score_train))
            plot_k_vs_score_test.append((k, score_test))
            
            answer[k] = {
                "score_train": score_train,
                "score_test": score_test
            }
            
        
        # Comment on the rate of accuracy change for testing data
        rate_of_accuracy_change_test = "The rate of accuracy change for testing data decreases as k increases."

        
        # Comment on the rate of accuracy change
        "text_is_topk_useful_and_why" : "This dataset benefits from top-k accuracy as it offers valuable insights into the model's performance when considering multiple potential predictions. Such a metric holds particular relevance for applications where the precision of exact predictions is not critical, but rather having a range of probable predictions is important."
        
        
        answer["clf"] = clf
        answer["plot_k_vs_score_train"] = plot_k_vs_score_train
        answer["plot_k_vs_score_test"] = plot_k_vs_score_test
        answer["text_rate_accuracy_change"] = rate_of_accuracy_change
        answer["text_is_topk_useful_and_why"] = topk_useful_and_why
        
        
        
        return answer, Xtrain, ytrain, Xtest, ytest

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
        
        X, y, Xtest, ytest = u.prepare_data()
        
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        
        Xtrain = Xtrain/255.0
        Xtest = Xtest/255.0
        
        indices_of_9 = np.where(ytrain == 9)[0]
        indices_to_remove = np.random.choice(indices_of_9, size=int(0.9 * len(indices_of_9)), replace=False)
        Xtrain = np.delete(Xtrain, indices_to_remove, axis=0)
        ytrain = np.delete(ytrain, indices_to_remove)
        ytrain = np.where(ytrain == 7, 0, ytrain)
        ytrain = np.where(ytrain == 9, 1, ytrain)
        
        ytest = np.where(ytest == 7, 0, ytest)
        ytest = np.where(ytest == 9, 1, ytest)
        
        
        # Answer is a dictionary with the same keys as part 1.B   
        answer["length_Xtrain"] = len(Xtrain)
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = np.max(Xtrain)
        answer["max_Xtest"] = np.max(Xtest)
       

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

        # Enter your code and fill the answer dictionary
        answer = {}
        
        #CROSS VALIDATION SETUP
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        #CLASSIFIER
        clf = SVC(random_state=42)
        
        #DEFINING SCORES
        scorers = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1': make_scorer(f1_score, average='macro')
        }
        
        #PERFORMING CROSS VALIDATION
        cv_results = cross_validate(clf, X, y, cv=cv_strategy, scoring=scorers)
        
        #TRAIN THE CLASSIFIER
        clf.fit(X, y)
        
        answer['cv'] = cv_strategy
        answer['clf'] = clf
        answer['scores'] = {
            'mean_accuracy': np.mean(cv_results['test_accuracy']),
            'std_accuracy': np.std(cv_results['test_accuracy']),
            'mean_precision': np.mean(cv_results['test_precision']),
            'std_precision': np.std(cv_results['test_precision']),
            'mean_recall': np.mean(cv_results['test_recall']),
            'std_recall': np.std(cv_results['test_recall']),
            'mean_f1': np.mean(cv_results['test_f1']),
            'std_f1': np.std(cv_results['test_f1']),
        }
        answer['is_precision_higher_than_recall'] = answer['scores']['mean_precision'] > answer['scores']['mean_recall']
        answer['explain_is_precision_higher_than_recall'] = ("Precision is higher than recall." if answer['is_precision_higher_than_recall'] 
    else "Recall is higher than precision.")
        answer['confusion_matrix_train'] = confusion_matrix(y, clf.predict(X))
        answer['confusion_matrix_test'] = confusion_matrix(ytest, clf.predict(Xtest))
        

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the compute_class_weight argument of the estimator to compute the class weights. 
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
        
        #COMPUTE CLASS WEIGHT
        unique_classes = np.unique(y)
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y)
        class_weights_dict = dict(zip(unique_classes, class_weights))

        #CROSS VALIDATION SETUP
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        #CLASSIFIER WITH WEIGHTS
        clf = SVC(class_weight=class_weights_dict, random_state=42)
        
        #DEFINING SCORES
        scorers = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1': make_scorer(f1_score, average='macro')
        }
        
        #PERFORM CROSS VALIDATION
        cv_results = cross_validate(clf, X, y, cv=cv_strategy, scoring=scorers)
        
        #TRAIN CLASSIFIER
        clf.fit(X, y)
        
        
        answer['class_weights'] = class_weights_dict
        answer['cv'] = cv_strategy
        answer['clf'] = clf
        answer['scores'] = {
            'mean_accuracy': np.mean(cv_results['test_accuracy']),
            'std_accuracy': np.std(cv_results['test_accuracy']),
            'mean_precision': np.mean(cv_results['test_precision']),
            'std_precision': np.std(cv_results['test_precision']),
            'mean_recall': np.mean(cv_results['test_recall']),
            'std_recall': np.std(cv_results['test_recall']),
            'mean_f1': np.mean(cv_results['test_f1']),
            'std_f1': np.std(cv_results['test_f1']),
        }
        answer['confusion_matrix_train'] = confusion_matrix(y, clf.predict(X))
        answer['confusion_matrix_test'] = confusion_matrix(ytest, clf.predict(Xtest))
        answer['explain_purpose_of_class_weights'] = (
            "Class weights are employed to mitigate the challenge of class imbalance by assigning greater importance to less prevalent classes."
            "This aids in directing the classifier's focus towards the minority classes during the training process."
        )
        answer['explain_performance_difference'] = (
            "An explanation derived from the observed performance disparities between employing default and weighted loss functions."
            "Generally, the application of class weights tends to enhance recall for minority classes while potentially impacting precision."
        )

        return answer
