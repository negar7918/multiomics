import numpy as np
import pandas as pd
from sklearn import preprocessing


def load_data(disease):
    # Load BRCA_data
    # Dataset    mRNA expression        DNA methylation    miRNA expression  Samples     Subtypes
    # BRCA         1000                   1000               503              875           5
    if disease == 'brca':
        view1_data = pd.read_csv('../../data/brca/1_all.csv', header=None)  # mRNA
        view2_data = pd.read_csv('../../data/brca/2_all.csv', header=None)  # DNAmeth
        label = pd.read_csv('../../data/brca/labels_all.csv', header=None)
    elif disease == 'brca2':
        view1_data = pd.read_csv('../../dataset/test/BRCA_mRNA.csv')
        view1_data = view1_data.T # mRNA
        view1_data = view1_data.iloc[1:, 1:]
        view2_data = pd.read_csv('../../dataset/test/BRCA_Methy.csv', header=None)
        view2_data = view2_data.T # DNAmeth
        view2_data = view2_data.iloc[1:, 1:]
        view3_data = pd.read_csv('../../dataset/test/BRCA_miRNA.csv', header=None)
        view3_data = view3_data.T # miRNA
        view3_data = view3_data.iloc[1:, 1:]
        label = pd.read_csv('../../dataset/test/BRCA_label.csv', header=None)
        label = label.to_numpy()[1:, 0]
        label[label == 'LumA'] = 0
        label[label == 'LumB'] = 1
        label[label == 'Normal'] = 2
        label[label == 'Basal'] = 3
        label[label == 'Her2'] = 4
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
    elif disease == '5cancer':
        view1_data = pd.read_csv('../../dataset/5cancer/mRNA.csv')
        view1_data = view1_data.T # mRNA
        view1_data = view1_data.iloc[1:, 1:]
        view2_data = pd.read_csv('../../dataset/5cancer/Methy.csv', header=None)
        view2_data = view2_data.T # DNAmeth
        view2_data = view2_data.iloc[1:, 1:]
        view3_data = pd.read_csv('../../dataset/5cancer/miRNA.csv', header=None)
        view3_data = view3_data.T # miRNA
        view3_data = view3_data.iloc[1:, 1:]
        label = pd.read_csv('../../dataset/5cancer/label.csv', header=None)
        label = label.to_numpy()[2:, 0]
        label[label == 'BRCA'] = 0
        label[label == 'COAD'] = 1
        label[label == 'KIRC'] = 2
        label[label == 'LUAD'] = 3
        label[label == 'LUSC'] = 4
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
    elif disease == 'kidney':
        kirp_mrna = pd.read_csv('../../dataset/kidney/KIRP_mRNA.csv')
        kirp_mrna = kirp_mrna.T.iloc[1:, :]
        kirc_mrna = pd.read_csv('../../dataset/kidney/KIRC_mRNA.csv')
        kirc_mrna = kirc_mrna.T.iloc[1:, :]
        view1_data = pd.concat((kirp_mrna, kirc_mrna), axis=0) # mRNA
        kirp_meth = pd.read_csv('../../dataset/kidney/KIRP_Methy.csv')
        kirp_meth = kirp_meth.T.iloc[1:, :]
        kirc_meth = pd.read_csv('../../dataset/kidney/KIRC_Methy.csv')
        kirc_meth = kirc_meth.T.iloc[1:, :]
        view2_data = pd.concat((kirp_meth, kirc_meth), axis=0)  # DNAmeth
        kirp_mirna = pd.read_csv('../../dataset/kidney/KIRP_miRNA.csv')
        kirp_mirna = kirp_mirna.T.iloc[1:, :]
        kirc_mirna = pd.read_csv('../../dataset/kidney/KIRC_miRNA.csv')
        kirc_mirna = kirc_mirna.T.iloc[1:, :]
        view3_data = pd.concat((kirp_mirna, kirc_mirna), axis=0)  # miRNA
        kirp, kirc = kirp_mirna.shape[0], kirc_mirna.shape[0]
        label = np.concatenate((np.full(kirp, 0, dtype=int), np.full(kirc, 1, dtype=int)), axis=0)
    elif disease == 'luad':
        view1_data = pd.read_csv('../../dataset/LUAD/data/gene.csv')
        view1_data = view1_data.T # mRNA
        view1_data = view1_data.iloc[1:, 1:]
        view2_data = pd.read_csv('../../dataset/LUAD/data/methyl.csv', header=None)
        view2_data = view2_data.T # DNAmeth
        view2_data = view2_data.iloc[1:, 1:]
        view3_data = pd.read_csv('../../dataset/LUAD/data/miRNA.csv', header=None)
        view3_data = view3_data.T # miRNA
        view3_data = view3_data.iloc[1:, 1:]
        label = pd.read_csv('../../dataset/LUAD/label.csv', header=None)
        label = label.to_numpy()[1:, 1:]
    elif disease == 'ad':
        view1_data_1 = pd.read_csv('../../dataset/AD/1_te.csv', header=None)
        view1_data_2 = pd.read_csv('../../dataset/AD/1_tr.csv', header=None)
        view1_data = pd.concat((view1_data_2, view1_data_1), axis=0) # mRNA
        view2_data_1 = pd.read_csv('../../dataset/AD/2_te.csv', header=None)
        view2_data_2 = pd.read_csv('../../dataset/AD/2_tr.csv', header=None)
        view2_data = pd.concat((view2_data_2, view2_data_1), axis=0) # DNAmeth
        view3_data_1 = pd.read_csv('../../dataset/AD/3_te.csv', header=None)
        view3_data_2 = pd.read_csv('../../dataset/AD/3_tr.csv', header=None)
        view3_data = pd.concat((view3_data_2, view3_data_1), axis=0) # miRNA
        label_1 = pd.read_csv('../../dataset/AD/labels_te.csv', header=None)
        label_2 = pd.read_csv('../../dataset/AD/labels_tr.csv', header=None)
        label = pd.concat((label_2, label_1), axis=0)

    print(view1_data.shape)
    print(view2_data.shape)

    # Transform features by scaling each feature to a given range (default range is (0, 1))
    scaler = preprocessing.MinMaxScaler()
    view1_data = scaler.fit_transform(view1_data)
    view2_data = scaler.fit_transform(view2_data)

    view_train_concatenate = np.concatenate((view1_data, view2_data), axis=1)
    if disease == 'lihc' or 'kirc':
        y_true = label
    else:
        y_true = label.to_numpy().flatten()
    print('label dim')
    print(y_true.shape)
    return view1_data, view2_data, view_train_concatenate, y_true