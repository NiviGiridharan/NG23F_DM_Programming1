# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from typing import Any
from numpy.typing import NDArray

import numpy as np
import utils as u

# Initially empty. Use for reusable functions across
# Sections 1-3 of the homework
import new_utils as nu


# ======================================================================
class Section1:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None

        Notes: notice the argument `seed`. Make sure that any sklearn function that accepts
        `random_state` as an argument is initialized with this seed to allow reproducibility.
        You change the seed ONLY in the section of run_part_1.py, run_part2.py, run_part3.py
        below `if __name__ == "__main__"`
        """
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly and 
       that you have all the required packages installed. For information about setting up 
       Python please consult the following link: https://www.anaconda.com/products/individual. 
       To test that your environment is set up correctly, simply execute `starter_code` in 
       the `utils` module. This is done for you. 
    """

    def partA(self):
        # Return 0 (ran ok) or -1 (did not run ok)
        answer = u.starter_code()
        return answer

    # ----------------------------------------------------------------------
    """
    B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
       functions in utils.py, to obtain a data matrix X consisting of only the digits 7 and 9. Make sure that 
       every element in the data matrix is a floating point number and scaled between 0 and 1 (write
       a function `def scale() in new_utils.py` that returns a bool to achieve this. Checking is not sufficient.) 
       Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
       and the maximum value of 𝑋 for both training and test sets. Use the routines provided in utils.
       When testing your code, I will be using matrices different than the ones you are using to make sure 
       the instructions are followed. 
    """

    def partB(
        self,
    ):
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)

        answer = {}

        # Enter your code and fill the `answer` dictionary

        answer["length_Xtrain"] = len(Xtrain)
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = Xtrain.max()
        answer["max_Xtest"] = Xtest.max()
        return answer, Xtrain, ytrain, Xtest, ytest

    """
    C. Train your first classifier using k-fold cross validation (see train_simple_classifier_with_cv 
       function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
       for the accuracy scores in each validation set in cross validation. (with k splits, cross_validate
       generates k accuracy scores.)  
       Remember to set the random_state in the classifier and cross-validator.
    """

    # ----------------------------------------------------------------------
    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary

        answer = {}
        clf = DecisionTreeClassifier(random_state=42)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(clf, X, y, cv=cv, return_train_score=False)
        
        mean_fit_time = np.mean(cv_results['fit_time'])
        std_fit_time = np.std(cv_results['fit_time'])
        mean_accuracy = np.mean(cv_results['test_score'])
        std_accuracy = np.std(cv_results['test_score'])
        
        answer["clf"] = clf
        answer["cv"] = cv
        answer["scores"] = {'mean_fit_time':mean_fit_time, 'std_fit_time':std_fit_time, 'mean_accuracy':mean_accuracy, 'std_accuracy':std_accuracy}
        return answer

    # ---------------------------------------------------------
    """
    D. Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator.
    Explain the pros and cons of using Shuffle-Split versus 𝑘-fold cross-validation.
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary
        clf = DecisionTreeClassifier(random_state=42)
        cv = ShuffleSplit(n_splits=5, random_state=42)
        cv_results = cross_validate(clf, X, y, cv=cv, return_train_score=False)

        mean_fit_time = np.mean(cv_results['fit_time'])
        std_fit_time = np.std(cv_results['fit_time'])
        mean_accuracy = np.mean(cv_results['test_score'])
        std_accuracy = np.std(cv_results['test_score'])
        # Answer: same structure as partC, except for the key 'explain_kfold_vs_shuffle_split'

        answer = {
        "clf": clf,
        "cv": cv,
        "scores": {
            "mean_fit_time": mean_fit_time,
            "std_fit_time": std_fit_time,
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy
        },
        "explain_kfold_vs_shuffle_split": (
            "K-fold cross-validation splits the data into k subsets and uses k-1 subsets for training "
            "and the remaining one subset for testing, repeating this process k times. Shuffle-split cross-validation "
            "randomly splits the data into train/test sets multiple times, which can be useful when the dataset "
            "is small or has imbalanced classes. However, it might not cover all data points in the test set "
            "if the number of splits is small, and it might not be suitable for tasks with time series data."
            )
        }
        return answer

    # ----------------------------------------------------------------------
    """
    E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. 
       Note that this may take a long time (2–5 mins) to run. Do you notice 
       anything about the mean and/or standard deviation of the scores for each k?
    """

    def partE(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        k_values = [2, 5, 8, 16]
        answer = {}

        for k in k_values:
            clf = DecisionTreeClassifier(random_state=42)
            cv = KFold(n_splits=k, shuffle=True, random_state=42)
            cv_results = cross_validate(clf, X, y, cv=cv, return_train_score=False)

            mean_accuracy = np.mean(cv_results['test_score'])
            std_accuracy = np.std(cv_results['test_score'])

            answer[k] = {
                "clf": clf,
                "cv": cv,
                "scores": {
                    "mean_accuracy": mean_accuracy,
                    "std_accuracy": std_accuracy
                }
            }

    return answer

    # ----------------------------------------------------------------------
    """
    F. Repeat part D with a Random-Forest classifier with default parameters. 
       Make sure the train test splits are the same for both models when performing 
       cross-validation. (Hint: use the same cross-validator instance for both models.)
       Which model has the highest accuracy on average? 
       Which model has the lowest variance on average? Which model is faster 
       to train? (compare results of part D and part F)

       Make sure your answers are calculated and not copy/pasted. Otherwise, the automatic grading 
       will generate the wrong answers. 
       
       Use a Random Forest classifier (an ensemble of DecisionTrees). 
    """

    def partF(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ) -> dict[str, Any]:
        """ """

        answer = {}

        # Decision Tree Classifier
        clf_DT = DecisionTreeClassifier(random_state=42)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results_DT = cross_validate(clf_DT, X, y, cv=cv, return_train_score=False)

        # Random Forest Classifier
        clf_RF = RandomForestClassifier(random_state=42)
        cv_results_RF = cross_validate(clf_RF, X, y, cv=cv, return_train_score=False)

        # Extracting mean accuracy and standard deviation for both classifiers
        mean_accuracy_DT = np.mean(cv_results_DT['test_score'])
        std_accuracy_DT = np.std(cv_results_DT['test_score'])

        mean_accuracy_RF = np.mean(cv_results_RF['test_score'])
        std_accuracy_RF = np.std(cv_results_RF['test_score'])

        # Determining which model has the highest accuracy on average
        model_highest_accuracy = "Random Forest" if mean_accuracy_RF > mean_accuracy_DT else "Decision Tree"

        # Determining which model has the lowest variance on average
        model_lowest_variance = "Random Forest" if std_accuracy_RF < std_accuracy_DT else "Decision Tree"

        # Comparing training time between the two models
        mean_fit_time_DT = np.mean(cv_results_DT['fit_time'])
        mean_fit_time_RF = np.mean(cv_results_RF['fit_time'])

        # Determining which model is faster to train
        model_fastest = "Random Forest" if mean_fit_time_RF < mean_fit_time_DT else "Decision Tree"

        # Constructing the answer dictionary
        answer["clf_DT"] = clf_DT
        answer["clf_RF"] = clf_RF
        answer["cv"] = cv
        answer["scores_DT"] = {'mean_fit_time': mean_fit_time_DT, 'std_fit_time': std_fit_time_DT,
                               'mean_accuracy': mean_accuracy_DT, 'std_accuracy': std_accuracy_DT}
        answer["scores_RF"] = {'mean_fit_time': mean_fit_time_RF, 'std_fit_time': std_fit_time_RF,
                               'mean_accuracy': mean_accuracy_RF, 'std_accuracy': std_accuracy_RF}
        answer["model_highest_accuracy"] = model_highest_accuracy
        answer["model_lowest_variance"] = model_lowest_variance
        answer["model_fastest"] = model_fastest

        return answer

    # ----------------------------------------------------------------------
    """
    G. For the Random Forest classifier trained in part F, manually (or systematically, 
       i.e., using grid search), modify hyperparameters, and see if you can get 
       a higher mean accuracy.  Finally train the classifier on all the training 
       data and get an accuracy score on the test set.  Print out the training 
       and testing accuracy and comment on how it relates to the mean accuracy 
       when performing cross validation. Is it higher, lower or about the same?

       Choose among the following hyperparameters: 
         1) criterion, 
         2) max_depth, 
         3) min_samples_split, 
         4) min_samples_leaf, 
         5) max_features 
    """

    def partG(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """
        Perform classification using the given classifier and cross validator.

        Parameters:
        - clf: The classifier instance to use for classification.
        - cv: The cross validator instance to use for cross validation.
        - X: The test data.
        - y: The test labels.
        - n_splits: The number of splits for cross validation. Default is 5.

        Returns:
        - y_pred: The predicted labels for the test data.

        Note:
        This function is not fully implemented yet.
        """

        # refit=True: fit with the best parameters when complete
        # A test should look at best_index_, best_score_ and best_params_
        """
        List of parameters you are allowed to vary. Choose among them.
         1) criterion,
         2) max_depth,
         3) min_samples_split, 
         4) min_samples_leaf,
         5) max_features 
         5) n_estimators
        """

        answer = {}

        # Random Forest Classifier with default parameters
        clf = RandomForestClassifier(random_state=42)
    
        # Grid of hyperparameters to search
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        # Grid search to find best hyperparameters
        grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X, y)

        # Extracting best estimator and its parameters
        best_estimator = grid_search.best_estimator_
        best_parameters = grid_search.best_params_

        # Training the classifier on all training data
        best_estimator.fit(X, y)

        # Predicting on training and test data
        y_train_pred = best_estimator.predict(X)
        y_test_pred = best_estimator.predict(Xtest)

        # Computing accuracy scores
        accuracy_train_orig = accuracy_score(y, y_train_pred)
        accuracy_test_orig = accuracy_score(ytest, y_test_pred)

        # Confusion matrices
        confusion_matrix_train_orig = confusion_matrix(y, y_train_pred)
        confusion_matrix_test_orig = confusion_matrix(ytest, y_test_pred)

        # Extracting accuracy scores from cross-validation
        mean_accuracy_cv = grid_search.best_score_

        # Constructing the answer dictionary
        answer["clf"] = clf
        answer["default_parameters"] = clf.get_params()
        answer["best_estimator"] = best_estimator
        answer["grid_search"] = grid_search
        answer["mean_accuracy_cv"] = mean_accuracy_cv
        answer["confusion_matrix_train_orig"] = confusion_matrix_train_orig
        answer["confusion_matrix_test_orig"] = confusion_matrix_test_orig
        answer["accuracy_orig_full_training"] = accuracy_train_orig
        answer["accuracy_orig_full_testing"] = accuracy_test_orig

        return answer
