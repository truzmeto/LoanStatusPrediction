import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm

#neural network
def ann((x_train, x_test, y_train, y_test):
        """

        """
        MLP = MLPClassifier(hidden_layer_sizes = (10,4),
                            activation = 'logistic',
                            solver = 'sgd',
                            learning_rate_init = 0.001,
                            max_iter = 500,
                            shuffle = True,
                            random_state = 10,
                            early_stopping = True,
                            validation_fraction=0.2)

        MLP.fit(x_train,y_train)
        pred_test_mlp = MLP.predict(x_test)
        #print(confusion_matrix(y_test,pred_test_mlp))
        #print(classification_report(y_test,pred_test_mlp))
        return pred_test_mlp



#SVM rbf kernel with Gid search CV
def svmCV(x_train, x_test, y_train, y_test):
        """

        """

        parameters = {'kernel': ['rbf'],
                      'C':[1, 4]}
        svc = svm.SVC(gamma="scale")
        clf = GridSearchCV(svc, parameters, cv=4, n_jobs=8)
        clf.fit(x_train, y_train)
        pred_test_svm = svm.predict(x_test)
        
        return pred_test_svm

