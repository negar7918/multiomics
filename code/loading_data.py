import numpy as np
import pandas as pd
from sklearn import preprocessing


def load_data(disease):
    if disease == 'brca':
        view1_data = pd.read_csv('../../data/brca/1_all.csv', header=None)  # mRNA
        view2_data = pd.read_csv('../../data/brca/2_all.csv', header=None)  # DNAmeth
        view3_data = pd.read_csv('../../data/brca/3_all.csv', header=None)  # miRNA
        label = pd.read_csv('../../data/brca/labels_all.csv', header=None)
    elif disease == 'lihc':
        view1_data = pd.read_csv('../../data/LIHC/data/gene.csv')
        view1_data = view1_data.T # mRNA
        view1_data = view1_data.iloc[1:, 1:]
        view2_data = pd.read_csv('../../data/LIHC/data/methyl.csv', header=None)
        view2_data = view2_data.T # DNAmeth
        view2_data = view2_data.iloc[1:, 1:]
        view3_data = pd.read_csv('../../data/LIHC/data/miRNA.csv', header=None)
        view3_data = view3_data.T # miRNA
        view3_data = view3_data.iloc[1:, 1:]
        label = pd.read_csv('../../data/LIHC/label.csv', header=None)
        label = label.to_numpy()[1:, 1:]
    elif disease == 'coad':
        view1_data = pd.read_csv('../../data/COAD/COAD_mRNA.csv')
        view1_data = view1_data.T # mRNA
        view1_data = view1_data.iloc[1:, 1:]
        view2_data = pd.read_csv('../../data/COAD/COAD_Methy.csv', header=None)
        view2_data = view2_data.T # DNAmeth
        view2_data = view2_data.iloc[1:, 1:]
        view3_data = pd.read_csv('../../data/COAD/COAD_miRNA.csv', header=None)
        view3_data = view3_data.T # miRNA
        view3_data = view3_data.iloc[1:, 1:]
        label = pd.read_csv('../../data/COAD/COAD_label.csv', header=None)
        label = label.to_numpy()[1:, 0]
        label[label == 'CIN'] = 0
        label[label == 'GS'] = 1
        label[label == 'MSI'] = 2
        label[label == 'POLE'] = 3
    elif disease == 'kirc':
        view1_data = pd.read_csv('../../data/KIRC/data/gene1.csv')
        view1_data = view1_data.T # mRNA
        view1_data = view1_data.iloc[1:, 1:]
        view2_data = pd.read_csv('../../data/KIRC/data/methyl.csv', header=None)
        view2_data = view2_data.T # DNAmeth
        view2_data = view2_data.iloc[1:, 1:]
        view3_data = pd.read_csv('../../data/KIRC/data/miRNA1.csv', header=None)
        view3_data = view3_data.T # miRNA
        view3_data = view3_data.iloc[1:, 1:]
        label = pd.read_csv('../../data/KIRC/label.csv', header=None)
        label = label.to_numpy()[1:, 1:]

    print(view1_data.shape)
    print(view2_data.shape)
    print(view3_data.shape)

    # Transform features by scaling each feature to a given range (default range is (0, 1))
    scaler = preprocessing.MinMaxScaler()
    view1_data = scaler.fit_transform(view1_data)
    view2_data = scaler.fit_transform(view2_data)
    view3_data = scaler.fit_transform(view3_data)

    view_train_concatenate = np.concatenate((view1_data, view2_data, view3_data), axis=1)
    if disease == 'lihc' or 'kirc':
        y_true = label
    else:
        y_true = label.to_numpy().flatten()
    print('label dim')
    print(y_true.shape)
    return view1_data, view2_data, view3_data, view_train_concatenate, y_true