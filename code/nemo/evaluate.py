import pandas as pd
import multiomics.code.evaluation as e
from multiomics.code.loading_data import load_data
from sklearn.model_selection import train_test_split
import numpy as np

view1_data, view2_data, view3_data, view_train_concatenate, y_true = load_data("coad") # brca

# split data
indices = np.arange(len(view_train_concatenate))
y_train, y_test, train_idx, test_idx = train_test_split( y_true, indices, test_size=0.2, random_state=1)


# truth = pd.read_csv('../../data/brca/labels_all.csv', header=None) # brca
# y_pred = pd.read_csv('./brca.csv', header=None)
# y = y_pred.iloc[1:, 1].astype(int).to_numpy()[test_idx]
# t = truth.to_numpy(dtype=int).flatten()[test_idx]

truth = pd.read_csv('../../data/coad/COAD_label.csv', header=None)

truth = truth.to_numpy()[1:, 0]
truth[truth == 'CIN'] = 0
truth[truth == 'GS'] = 1
truth[truth == 'MSI'] = 2
truth[truth == 'POLE'] = 3

y_pred = pd.read_csv('./coad.csv', header=None)

y = y_pred.iloc[1:, 1].astype(int).to_numpy()[test_idx]
t = truth.astype(int)[test_idx]

nmi_, ari_, f_score_, acc_, v_, ch = e.evaluate(t, y)
print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f,  '
                               'ch_index: %.4f  <==|' % (nmi_, ari_, f_score_, acc_, v_, ch))

