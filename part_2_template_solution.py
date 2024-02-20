# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
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
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        #loading the MNIST dataset
        mnist = fetch_openml('mnist_784')
        X, y = mnist.data, mnist.target
        
        # Split X and y into train and test sets
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
        
        answer = {}
        # Enter your code and fill the `answer`` dictionary

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set

        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary


        
        # Print out the number of elements in each class y
        unique_train, counts_train = np.unique(ytrain, return_counts=True)
        unique_test, counts_test = np.unique(ytest, return_counts=True)

        # Print out the number of classes for both training and testing datasets
        num_classes_train = len(unique_train)
        num_classes_test = len(unique_test)

        # Print out the number of elements in each class in the training and testing set
        class_count_train = dict(zip(unique_train, counts_train))
        class_count_test = dict(zip(unique_test, counts_test))

        # Print out the length of Xtrain, Xtest, ytrain, ytest
        length_Xtrain = len(Xtrain)
        length_Xtest = len(Xtest)
        length_ytrain = len(ytrain)
        length_ytest = len(ytest)

        # Print out the maximum value in the training and testing set
        max_Xtrain = Xtrain.max()
        max_Xtest = Xtest.max()

        # Fill the answer dictionary
        answer["nb_classes_train"] = num_classes_train
        answer["nb_classes_test"] = num_classes_test
        answer["class_count_train"] = class_count_train
        answer["class_count_test"] = class_count_test
        answer["length_Xtrain"] = length_Xtrain
        answer["length_Xtest"] = length_Xtest
        answer["length_ytrain"] = length_ytrain
        answer["length_ytest"] = length_ytest
        answer["max_Xtrain"] = max_Xtrain
        answer["max_Xtest"] = max_Xtest

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        """ """
        # Enter your code and fill the `answer`` dictionary
        answer = {}

        """
        `answer` is a dictionary with the following keys:
           - 1000, 5000, 10000: each key is the number of training samples

           answer[k] is itself a dictionary with the following keys
            - "partC": dictionary returned by partC section 1
            - "partD": dictionary returned by partD section 1
            - "partF": dictionary returned by partF section 1
            - "ntrain": number of training samples
            - "ntest": number of test samples
            - "class_count_train": number of elements in each class in
                               the training set (a list, not a numpy array)
            - "class_count_test": number of elements in each class in
                               the training set (a list, not a numpy array)
        """
        for ntrain in ntrain_list:
            Xtrain = X[:ntrain, :]
            ytrain = y[:ntrain]

            class_count_train = [np.sum(ytrain == i) for i in range(10)]

            for ntest in ntest_list:
                Xtest_sub = Xtest[:ntest, :]
                ytest_sub = ytest[:ntest]

                # Part C
                partC_result = self.partC(Xtrain, ytrain)

                # Part D
                partD_result = self.partD(Xtrain, ytrain)

                # Part F
                partF_result = self.partF(Xtrain, ytrain, Xtest_sub, ytest_sub)

                # Storing results in the answer dictionary
                answer[(ntrain, ntest)] = {
                    "partC": partC_result,
                    "partD": partD_result,
                    "partF": partF_result,
                    "ntrain": ntrain,
                    "ntest": ntest,
                    "class_count_train": class_count_train,
                    "class_count_test": [np.sum(ytest_sub == i) for i in range(10)],
                }        

        return answer
