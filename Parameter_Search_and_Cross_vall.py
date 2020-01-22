from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import KFold


def svc_param_selection(x, y, n_folds, is_wrapper=False):
    Cs = [0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1]
    if is_wrapper:
        kernels = ['linear']
    else:
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    degrees = [3, 4, 5, 10, 15]
    param_grid = {'C': Cs, 'gamma': gammas, 'kernel':kernels, 'degree':degrees}
    clf = GridSearchCV(SVC(), param_grid, cv=n_folds)
    clf.fit(x, y)
    return clf.best_params_

def model_scoring(high_model_params,classes, X_last,y_last, name):
    acc_list = []
    precision_list = []
    recall_list = []
    kf = KFold(n_splits=10)
    for train_ind, test_ind in kf.split(X_last, y_last):
        final_svm = SVC(C=high_model_params['C'], gamma=high_model_params['gamma'], kernel=high_model_params['kernel'],
                            degree=high_model_params['degree'], decision_function_shape='ovo')
        final_svm.fit(X_last[train_ind], y_last[train_ind])
        predicted = final_svm.predict(X_last[test_ind])
        acc_list.append(accuracy_score(y_last[test_ind], predicted))
        precision_list.append(precision_score(y_last[test_ind], predicted, labels=classes))
        recall_list.append(recall_score(y_last[test_ind], predicted, labels=classes))
        with open(name+'_svm.txt', 'a+') as file:
            result_string = 'ACC: ' + str(acc_list[-1]) + '  Precision: ' + str(precision_list[-1]) + '   Recall: ' + str(
                recall_list[-1]) + '\n'
            file.write(result_string)
            print(result_string)
        del final_svm


    x_train_conf, x_test_conf, y_train_conf, y_test_conf = train_test_split(X_last, y_last,
                                                                            test_size=0.25, random_state=42)
    final_svm = SVC(C=high_model_params['C'], gamma=high_model_params['gamma'], kernel=high_model_params['kernel'],
                        degree=high_model_params['degree'], decision_function_shape='ovo')
    final_svm.fit(x_train_conf, y_train_conf)
    plot_confusion_matrix(final_svm, x_test_conf, y_test_conf)

