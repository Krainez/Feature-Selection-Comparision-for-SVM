import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from collections import Counter


def prepare_input_data(train, test):
    scaler = MinMaxScaler()
    data_1 = train[[x for x in train.columns if "call" not in x]]
    data_2 = test[[x for x in test.columns if "call" not in x]]
    data_1 = data_1.set_index('Gene Accession Number')
    data_2 = data_2.set_index('Gene Accession Number')

    data = pd.concat([data_1, data_2], axis=1)
    data = data.drop('Gene Description', axis=1)
    data = data.T
    data_last = pd.DataFrame(scaler.fit_transform(data.values), columns=data.columns, index=data.index)

    return data_last


def prepare_output_data(target):

    target = target.set_index('patient')
    encoder = LabelEncoder()
    encoder.fit(target)
    labels = pd.Series(encoder.transform(target), index=target.index)
    return labels, encoder.classes_


def histogram_plotter_for_data(data, target):
    data['class'] = data.index
    temp = target.to_dict()
    data['class'] = data['class'].astype(int)
    data['class'].replace(temp['cancer'], inplace=True)
    data['class'].replace({'ALL': 0, 'AML': 1}, inplace=True)
    data['class'].plot(kind='hist')
def histogram_plotter_selected_features(name):
    with open(name, 'r') as file1:
        list_of_selected_features = file1.readlines()
    features_counts = Counter(list_of_selected_features)
    df = pd.DataFrame.from_dict(features_counts, orient='index')
    df.plot(kind='bar')


def cluster_map_plotter(data):
    diff = data.groupby('class').mean().apply(lambda x: x[0] - x[1])
    my_columns = diff.sort_values().index.tolist()
    selected = my_columns[:5] + my_columns[-5:]
    small_data = data[selected + ['class']]
    sns.set(font_scale=0.5)
    sns.clustermap(small_data.corr(), cmap='RdBu_r', figsize=(15, 15))


if __name__ == '__main__':
    data_train = pd.read_csv('data_set_ALL_AML_train.csv')
    data_test = pd.read_csv('data_set_ALL_AML_independent.csv')
    target_all = pd.read_csv('actual.csv')
    target_all = target_all.set_index('patient')
    data_all = prepare_input_data(data_train, data_test)
    histogram_plotter_selected_features('Selected_feature_names.txt')
    histogram_plotter_for_data(data_all, target_all)
    cluster_map_plotter(data_all)