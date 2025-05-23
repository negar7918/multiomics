#%%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
from loading_data import load_data
from torch import nn
import util
import math
import torch.nn.functional as F
import torch.optim
import torch.nn.init as init
from contrastive_loss import InstanceLoss
import argparse
from sklearn.cluster import KMeans
import evaluation as evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
from vae.mocss_vae import SharedAndSpecificEmbedding as SASEvae
from ae.mocss_original_refactored import SharedAndSpecificEmbedding as SASEae
from prod_gamma_dirvae.prod_gamma_dirvae_cancer import SharedAndSpecificEmbedding as SASEpgdv
from gamma_dirvae.gamma_dirvae_cancer import SharedAndSpecificEmbedding as SASEgdv
from Laplace_dirvae.lap_dirvae_cancer import SharedAndSpecificEmbedding as SASElap
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import itertools
#from neurhci.uhci import UHCI
#from neurhci.hierarchical import HCI2layers
#from neurhci.marginal_utilities import MonotonicSelector, MarginalUtilitiesLayer, IdentityClipped, Identity
import xgboost
import os

device = 'cpu'
POSITION = 0
OMICS_NAMES = ['mRNA', 'DNAmethyl', 'miRNA', 'Shared']

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

setup_seed(2)

#%%
def get_data(name_model, disease):
    omics_shape = {'brca': [1000,1000,503], 'kirc': [58315, 22928, 1879], 'lihc': [20530, 5000, 1046], 'coad': [17260, 19052, 375]}[disease]
    group_numbers = {'brca': 4, 'coad': 5, 'lihc':4, 'kirc': 4}[disease]
    model_sas = {'vae': SASEvae(
                    view_size=[omics_shape[0], omics_shape[1], omics_shape[2]],
                    n_units_1=[512, 256, 128, 32], n_units_2=[512, 256, 128, 32],
                    n_units_3=[256, 128, 64, 32], mlp_size=[32, 8]
                ), 
                'ae':SASEae(
                    view_size=[omics_shape[0], omics_shape[1], omics_shape[2]],
                    n_units_1=[512, 256, 128, 32], n_units_2=[512, 256, 128, 32],
                    n_units_3=[256, 128, 64, 32], mlp_size=[32, 8]
                ), 
                'ProdGammaDirVae':SASEpgdv(
                    "ProdGamDirVae", K=group_numbers, view_size=[omics_shape[0], omics_shape[1], omics_shape[2]],
                    n_units_1=[512, 256, 128, 8], n_units_2=[512, 256, 128, 8],
                    n_units_3=[256, 128, 64, 8], mlp_size=[32, 8]
                ), 
                'GammaDirVae':SASEgdv(
                    view_size=[omics_shape[0], omics_shape[1], omics_shape[2]],
                    n_units_1=[512, 256, 128, 32], n_units_2=[512, 256, 128, 32],
                    n_units_3=[256, 128, 64, 32], mlp_size=[32, 8]
                ), 
                'lapdirvae': SASElap(view_size=[omics_shape[0], omics_shape[1], omics_shape[2]],
                    n_units_1=[512, 256, 128, 32], n_units_2=[512, 256, 128, 32],
                    n_units_3=[256, 128, 64, 32], mlp_size=[32, 8]
                )
                }[name_model]
    model_embedding = model_sas.to(device)
    path_all = f'../results/{disease}/'
    model_embedding.load_state_dict(torch.load(path_all+f'model_{disease}_{name_model}', weights_only=False))
    path = f'../../data/data_test' +'/'
    X_whole_test = np.load(os.path.join(path, f'test_data_{disease}.npy'), allow_pickle=True)
    all_labels = np.load(os.path.join(path, f'test_label_{disease}.npy'), allow_pickle=True)
    if disease == 'brca':
        y_whole_test = all_labels.flatten()
    elif disease == 'lihc':
        all_labels_str = all_labels[:,-1]
        y_whole_test = np.array([len(k)-1 for k in all_labels_str])
    elif disease == 'kirc':
        y_whole_test = all_labels[:,1].astype(int)
    else:
        y_whole_test = all_labels
    X_subtrain, X_subtest, y_subtrain, y_subtest = train_test_split(X_whole_test, y_whole_test, test_size=0.25, random_state=12)

    X_whole_test_omics = torch.from_numpy(X_whole_test.astype(float)).float().to(device)
    Y_whole_test = y_whole_test.astype(int)
    X_subtest_omics = torch.from_numpy(X_subtest.astype(float)).float().to(device)
    Y_subtest = y_subtest.astype(int)
    X_subtrain_omics = torch.from_numpy(X_subtrain.astype(float)).float().to(device)
    Y_subtrain = y_subtrain.astype(int)

    Xs = []
    with torch.no_grad():
        for X_loader in [X_subtrain_omics, X_subtest_omics, X_whole_test_omics]:
            if name_model == 'vae':
                (view1_specific_em_new, view1_specific_mu_new, view1_specific_sigma_new, view1_shared_em_new,
                view2_specific_em_new, view2_specific_mu_new, view2_specific_sigma_new, view2_shared_em_new,
                view3_specific_em_new, view3_specific_mu_new, view3_specific_sigma_new, view3_shared_em_new,
                view1_specific_rec_new, view1_shared_rec_new, view2_specific_rec_new,
                view2_shared_rec_new, view3_specific_rec_new, view3_shared_rec_new,
                view1_shared_mlp_new, view2_shared_mlp_new, view3_shared_mlp_new) = (
                    model_embedding(X_loader[:,:omics_shape[0]], 
                    X_loader[:,omics_shape[0]:omics_shape[0]+omics_shape[1]],
                    X_loader[:,omics_shape[0]+omics_shape[1]:]))
            elif name_model == 'ae':
                view1_specific_em_new, view1_shared_em_new, view2_specific_em_new, \
                view2_shared_em_new, view3_specific_em_new,  \
                view3_shared_em_new, view1_specific_rec_new, view1_shared_rec_new, view2_specific_rec_new, \
                view2_shared_rec_new, view3_specific_rec_new, view3_shared_rec_new, view1_shared_mlp_new, view2_shared_mlp_new, \
                view3_shared_mlp_new = model_embedding(X_loader[:,:omics_shape[0]], 
                    X_loader[:,omics_shape[0]:omics_shape[0]+omics_shape[1]],
                    X_loader[:,omics_shape[0]+omics_shape[1]:])
            elif name_model == 'GammaDirVae':
                view1_specific_em_new, view1_specific_alpha_new, view1_shared_em_new, view2_specific_em_new, \
                view2_specific_alpha_new, view2_shared_em_new, view3_specific_em_new, view3_specific_alpha_new, \
                view3_shared_em_new, view1_specific_rec_new, view1_shared_rec_new, view2_specific_rec_new, \
                view2_shared_rec_new, view3_specific_rec_new, view3_shared_rec_new, view1_shared_mlp_new, view2_shared_mlp_new, \
                view3_shared_mlp_new = model_embedding(X_loader[:,:omics_shape[0]], 
                    X_loader[:,omics_shape[0]:omics_shape[0]+omics_shape[1]],
                    X_loader[:,omics_shape[0]+omics_shape[1]:])
            elif name_model == 'ProdGammaDirVae':
                view1_specific_em_new, view1_specific_alpha_new, view1_shared_em_new, view2_specific_em_new, \
                view2_specific_alpha_new, view2_shared_em_new, view3_specific_em_new, view3_specific_alpha_new, \
                view3_shared_em_new, view1_specific_rec_new, view1_shared_rec_new, view2_specific_rec_new, \
                view2_shared_rec_new, view3_specific_rec_new, view3_shared_rec_new, view1_shared_mlp_new, view2_shared_mlp_new, \
                view3_shared_mlp_new = model_embedding(X_loader[:,:omics_shape[0]], 
                    X_loader[:,omics_shape[0]:omics_shape[0]+omics_shape[1]],
                    X_loader[:,omics_shape[0]+omics_shape[1]:])
            elif name_model == 'lapdirvae':
                view1_specific_em_new, view1_specific_mu_new, view1_specific_sig_new, view1_shared_em_new, view2_specific_em_new, \
                view2_specific_mu_new, view2_specific_sig_new, view2_shared_em_new, view3_specific_em_new, view3_specific_mu_new, \
                view3_specific_sig_new, view3_shared_em_new, view1_specific_rec_new, view1_shared_rec_new, view2_specific_rec_new, \
                view2_shared_rec_new, view3_specific_rec_new, view3_shared_rec_new, view1_shared_mlp_new, view2_shared_mlp_new, \
                view3_shared_mlp_new = model_embedding(X_loader[:,:omics_shape[0]], 
                    X_loader[:,omics_shape[0]:omics_shape[0]+omics_shape[1]],
                    X_loader[:,omics_shape[0]+omics_shape[1]:])
            view_shared_common = (view1_shared_em_new + view2_shared_em_new + view3_shared_em_new) / 3
            final_embedding = torch.cat(
                (view1_specific_em_new, view2_specific_em_new, view3_specific_em_new, view_shared_common), dim=1)
            out_shapes = [view1_specific_em_new.shape[1], view2_specific_em_new.shape[1], view3_specific_em_new.shape[1], view_shared_common.shape[1]]
            final_embedding = final_embedding
            print(final_embedding.shape)
            Xs.append(final_embedding.detach().numpy())

    return X_subtrain, Y_subtrain, Xs[0], X_subtest, Y_subtest, Xs[1], X_whole_test, Y_whole_test, Xs[2], out_shapes, model_embedding

#%%
def extract_omics(x, omics):
    xs = [x[:,:32]*1., x[:,32:64]*1., x[:,64:96]*1., x[:,96:]*1.]
    return np.concatenate([xs[i] for i in omics], axis=1)

def one_knn(X_subtrain, Y_subtrain, X_subtest, Y_subtest, X_whole_test, Y_whole_test, disease):
    nb_classes = {
            'brca': 5,
            'lihc': 2,
            'coad': 4,
            'kirc': 2}[disease]    
    best_inertia = float("inf")
    best_labels = None
    for i in range(30):
        kmeans = KMeans(n_clusters=nb_classes, init='k-means++', random_state=i)
        labels = kmeans.fit_predict(X_whole_test)
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_labels = labels
    nmi_, ari_, f_score_, acc_, v_, ch = evaluation.evaluate(Y_whole_test, best_labels)
    #print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f,  '
    #                        'ch_index: %.4f  <==|' % (nmi_, ari_, f_score_, acc_, v_, ch))
    
    knn = KNeighborsClassifier(n_neighbors=nb_classes)
    # Train the model
    knn.fit(X_subtrain, Y_subtrain)
    # Predict on test set
    y_pred = knn.predict(X_subtest)
    accuracy = accuracy_score(Y_subtest, y_pred)
    #print(f"kNN acc: {accuracy:.2f}")
    return nmi_, accuracy

def one_ablation(X_subtrain, Y_subtrain, X_subtest, Y_subtest, X_whole_test, Y_whole_test, which_omics, disease):
    xe = extract_omics(X_subtest, which_omics)
    xr = extract_omics(X_subtrain, which_omics)
    xw = extract_omics(X_whole_test, which_omics)
    nmi, acc = one_knn(xr, Y_subtrain, xe, Y_subtest, xw, Y_whole_test, disease)
    return nmi, acc

def insert_i_in_tuple(tup, i):
    assert i not in tup
    return tuple(sorted(list(tup) + [i]))

def shapley_size_4_dirty(score_dict):
    print(score_dict)
    shapley_values = [0., 0., 0., 0.]
    for i in range(4):
        all_combinations = []
        print(i)
        shifted = [a for a in range(4) if not a==i]
        print(shifted)
        for j in range(4):
            all_combs_j = list(itertools.combinations(shifted, j))
            all_combinations += all_combs_j
        print(all_combinations)
        for comb in all_combinations:
            print(comb)
            s = len(comb)
            n = 4
            coef = math.factorial(s)*math.factorial(n-s-1)/math.factorial(n)
            combi = insert_i_in_tuple(comb, i)
            shapley_values[i] += coef*(score_dict[combi] - score_dict[comb])
    return shapley_values
            
#%%
def all_ablations(X_subtrain, Y_subtrain, X_subtest, Y_subtest, X_whole_test, Y_whole_test, name_model, disease):
    log_file = f'results/logs_{disease}.txt'
    scores = []
    score_dict = {():0.}
    nmi_dict = {(): 0.}
    all_combinations = []
    for i in range(4):
        all_combs_i = itertools.combinations(range(4), i+1)
        all_combinations += all_combs_i
    for omics_list in all_combinations:
        #print(omics_list)
        nmi, acc = one_ablation(X_subtrain, Y_subtrain, X_subtest, Y_subtest, X_whole_test, Y_whole_test, omics_list, disease)
        names = ', '.join([OMICS_NAMES[i] for i in omics_list])
        scores.append(f'{names}: KNN accuracy: {acc}, nmi: {nmi}\n')
        score_dict[omics_list] = acc
        nmi_dict[omics_list] = nmi
   
    shapley_values_acc = shapley_size_4_dirty(score_dict)
    print(shapley_values_acc)
    print(sum(shapley_values_acc), score_dict[(0,1,2,3)])
    shapley_values_nmi = shapley_size_4_dirty(nmi_dict)
    print(shapley_values_nmi)
    print(sum(shapley_values_nmi), nmi_dict[(0,1,2,3)])
    
    with open(log_file, 'a') as file:
        file.write(name_model + '\n')
        for s in scores:
            file.write(s)
        file.write('Shapley values accuracies: \n')
        for o,s in zip(OMICS_NAMES, shapley_values_acc):
            file.write(o + ': ' + '{:.2f}'.format(s) + '\n')
        file.write('Shapley values nmi: \n')
        for o,s in zip(OMICS_NAMES, shapley_values_nmi):
            file.write(o + ': ' + '{:.2f}'.format(s) + '\n')
        file.write('\n')

    return shapley_values_acc, shapley_values_nmi, score_dict, nmi_dict

def all_expes(disease):
    score_dicts = {}
    for name_model in ['ae', 'vae', 'lapdirvae', 'GammaDirVae', 'ProdGammaDirVae']:
        print(name_model)
        _, Y_subtrain, X_subtrain, _, Y_subtest, X_subtest, _, Y_whole_test, X_whole_test, out_shapes, _ = get_data(name_model, disease)
        score_dicts[name_model] = all_ablations(X_subtrain, Y_subtrain, X_subtest, Y_subtest, X_whole_test, Y_whole_test, name_model, disease)
    return score_dicts


#%%
shaps_all = {}
for disease in ['kirc', 'coad', 'brca', 'lihc']:
    shaps_all[disease] = all_expes(disease)
# %%
def plot_shaps(shaps, name_model):
    x = np.arange(4)
    width = 6
    colors = ['red', 'blue', 'green', 'orange']
    name_model_clean = {'ae':'MOCSS (AE)', 'vae':'VAE', 'lapdirvae':'LapDirVae', 'GammaDirVae':'GamDirVae', 'ProdGammaDirVae':'ProdGamDirVae'}[name_model]
    fig,ax = plt.subplots(figsize=(13,4), dpi=80)
    x = np.arange(4)
    width = 6
    bar_handles = {}
    for i,disease in enumerate(['brca', 'coad', 'lihc', 'kirc']):
        shapley_values = shaps[disease][name_model][0]
        bars = ax.bar(x-1.5+i*width, shapley_values, color=colors)
        for j, bar in enumerate(bars):
            if OMICS_NAMES[j] not in bar_handles:
                bar_handles[OMICS_NAMES[j]] = bar
        ax.text(
            i * width, 
            -0.05,  # Negative position moves the text below bars
            disease, 
            ha='center', 
            fontsize=15
        )
    #plt.xlabel('Omics Names')
    plt.ylabel('Shapley Value for Accuracy')
    plt.title(f"{name_model_clean}")
    ax.set_xticks([])
    ax.legend(bar_handles.values(), bar_handles.keys())
    plt.grid()
    plt.ylim(0, 0.5)
    plt.savefig(f'../results/shapley_acc_{name_model}.pdf')
    
    plt.clf()

    fig,ax = plt.subplots(figsize=(13,4), dpi=120)
    bar_handles = {}
    for i,disease in enumerate(['brca', 'coad', 'lihc', 'kirc']):
        shapley_values = shaps[disease][name_model][1]
        bars = ax.bar(x-1.5+i*width, shapley_values, color=colors)
        for j, bar in enumerate(bars):
            if OMICS_NAMES[j] not in bar_handles:
                bar_handles[OMICS_NAMES[j]] = bar
        ax.text(
            i * width, 
            -0.05,  # Negative position moves the text below bars
            disease, 
            ha='center', 
            fontsize=15
        )
    #plt.xlabel('Omics Names')
    plt.ylabel('Shapley Value for NMI')
    plt.title(f"{name_model_clean}")
    ax.set_xticks([])
    ax.legend(bar_handles.values(), bar_handles.keys())
    plt.grid()
    plt.savefig(f'../results/shapley_nmi_{name_model}.pdf')
    
    plt.clf()

plot_shaps(shaps_all, 'ae')
plot_shaps(shaps_all, 'vae')
plot_shaps(shaps_all, 'lapdirvae')
plot_shaps(shaps_all, 'GammaDirVae')
plot_shaps(shaps_all, 'ProdGammaDirVae')

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# brca, coad, lihc, kirc
def plot_ablations(shaps_all, name_model):
    name_model_clean = {'ae':'MOCSS (AE)', 'vae':'VAE', 'lapdirvae':'LapDirVae', 'GammaDirVae':'GamDirVae', 'ProdGammaDirVae':'ProdGamDirVae'}[name_model]
    n_data = 4
    vals = []
    names = []
    plt.figure(figsize=(12, 6))
    diseases = ['brca', 'coad', 'lihc', 'kirc']
    for i,disease in enumerate(diseases):
        dict_values = shaps_all[disease][name_model][2]
        for j,(k,v) in enumerate(dict_values.items()):
            if len(k):
                if i==0:
                    names.append(', '.join([OMICS_NAMES[omic_idx] for omic_idx in k]))
                    vals.append([])
                vals[j-1].append(v)
    vals = np.array(vals).T
    
    markers = ['o', 'x', 's', '*']
    colors = ['blue', 'orange', 'red', 'green']
    for i,v in enumerate(vals):
        plt.plot(range(len(v)), v, label=diseases[i], linestyle='None', color=colors[i], marker=markers[i])
    
    plt.xticks(ticks=np.arange(15), labels=names, rotation=45)
    plt.title(f"Ablation Study for {name_model_clean}")
    plt.ylabel("k-NN Accuracy")
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(f'ablation_dots{name_model}.pdf')
    plt.clf()

    # Plot
    data_titled = {d:vals.T[i] for i,d in enumerate(names)}
    #return data_titled
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data_titled, color='tab:blue')
    plt.grid()

    # Only draw a horizontal line at the median for group "B"
    plt.xticks(rotation=45)  # Rotate x labels if needed
    plt.title(f"Ablation Study for {name_model_clean}")
    plt.tight_layout()
    plt.savefig(f'ablation_boxplots_{name_model}.pdf')
# %%
    
plot_ablations(shaps_all, 'ae')
plot_ablations(shaps_all, 'vae')
plot_ablations(shaps_all, 'lapdirvae')
plot_ablations(shaps_all, 'GammaDirVae')
plot_ablations(shaps_all, 'ProdGammaDirVae')
# %%
