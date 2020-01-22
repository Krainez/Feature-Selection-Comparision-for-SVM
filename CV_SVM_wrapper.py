from sklearn import svm
from Parameter_Search_and_Cross_vall import svc_param_selection,model_scoring
from sklearn.feature_selection import RFE
import time as t
import numpy as np
from sklearn.model_selection import train_test_split
from EDA_main import prepare_input_data,prepare_output_data
import pandas as pd



#========DATA READ============#
data_1 = pd.read_csv('data_set_ALL_AML_train.csv')
data_2 = pd.read_csv('data_set_ALL_AML_independent.csv')
target = pd.read_csv('actual.csv')
data_all = prepare_input_data(data_1, data_2)
target_all, classes = prepare_output_data(target)
#==============================#

x_train, x_test, y_train, y_test = train_test_split(data_all.values, target_all.values, test_size=0.25, random_state=42)

nof_list = np.arange(1, x_train[0].shape[0])
high_score = 0
nof = 0
high_score_n_feature = x_train.shape[1]
score_list = []
early_stopping = 30
counter = 0
for n in nof_list:
    start_wrapper = t.time()
    rfe = RFE(svm.SVC(kernel='linear'), n)
    X_train_rfe = rfe.fit_transform(x_train, y_train)
    X_test_rfe = rfe.transform(x_test)
    wrapper_time = t.time()-start_wrapper
    svc_best_params = svc_param_selection(X_train_rfe, y_train, n_folds=3, is_wrapper=True)
    svm_classifier = svm.SVC(C=svc_best_params['C'], gamma=svc_best_params['gamma'], kernel=svc_best_params['kernel'])
    svm_classifier.fit(X_train_rfe, y_train)
    score = svm_classifier.score(X_test_rfe, y_test)
    score_list.append(score)
    if score > high_score or (score == high_score and high_score_n_feature < n):
        high_score = score
        high_rank = rfe.ranking_
        nof = n
        high_model_params = svm_classifier.get_params()
        high_score_x_train = X_train_rfe
        high_score_x_test = X_test_rfe
        high_score_n_feature = n
        high_score_time = wrapper_time
    else:
        counter = counter + 1
        if counter >= early_stopping:
            break
    print('For number', str(n), ' feature score =', str(score))

print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
list_of_selected_indexes = list(np.argsort(high_rank)[-high_score_n_feature:])
column_names = data_all.columns.values
selected_feature_names = column_names[list_of_selected_indexes]
with open('Selected_feature_names.txt', 'a+') as file:
    for gene_expression in selected_feature_names:
        added_line = gene_expression+'\n'
        file.write(added_line)

X_last = np.concatenate((high_score_x_train, high_score_x_test), axis=0)
y_last = np.concatenate((y_train, y_test), axis=0)


model_scoring(high_model_params=high_model_params, classes=classes, X_last=X_last, y_last=y_last, name='wrapper')