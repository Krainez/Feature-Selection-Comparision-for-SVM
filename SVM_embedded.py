from sklearn.linear_model import LassoCV
from EDA_main import prepare_input_data, prepare_output_data
import pandas as pd
import numpy as np
from Parameter_Search_and_Cross_vall import svc_param_selection, model_scoring
from sklearn.model_selection import train_test_split
import time as t


#========DATA READ============#
data_1 = pd.read_csv('data_set_ALL_AML_train.csv')
data_2 = pd.read_csv('data_set_ALL_AML_independent.csv')
target = pd.read_csv('actual.csv')
data_all = prepare_input_data(data_1, data_2)
target_all, classes = prepare_output_data(target)
x_train, x_test, y_train, y_test = train_test_split(data_all.values, target_all.values, test_size=0.25, random_state=42)
#==============================#
start_embedded = t.time()
reg = LassoCV(cv=10)
reg.fit(x_train, y_train)
embedded_time = t.time()-start_embedded
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(x_train, y_train))
coef = pd.Series(reg.coef_)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()

indexes = np.where(coef == 0)
x_train = np.delete(x_train, indexes, axis=1)
x_test = np.delete(x_test, indexes, axis=1)

X_last = np.concatenate((x_train, x_test), axis=0)
y_last = np.concatenate((y_train, y_test), axis=0)

list_of_selected_indexes = list(np.where(coef!=0))
column_names = data_all.columns.values
selected_feature_names = column_names[list_of_selected_indexes]
with open('Selected_feature_names.txt', 'a+') as file:
    for gene_expression in selected_feature_names:
        added_line = gene_expression+'\n'
        file.write(added_line)
params = svc_param_selection(x_train, y_train, 10)
model_scoring(high_model_params=params, classes=classes, X_last=X_last, y_last=y_last, name='embedded')

