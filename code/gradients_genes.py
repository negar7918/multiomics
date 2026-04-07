#%%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import math
import torch.optim
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
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
device = 'cpu'
POSITION = 0
OMICS_NAMES = ['mRNA', 'DNA', 'miRNA', 'Shared']

path = "./results"
os.makedirs(path, exist_ok=True)

#%%
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
                'ProdGamDirVae':SASEpgdv(
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
    #model_embedding.load_state_dict(torch.load(path_all+f'model_{disease}_{name_model}', weights_only=False))

    ls2 = [{'loss': 100000000, 'config': 'test'}]
    model_path = ('results/models/'+name_model)
    for (dir_path, dir_names, file_names) in os.walk(model_path):
        for config in dir_names:
            name = os.path.join(config, 'loss.npy')
            f = os.path.join(model_path, name)
            if os.path.exists(f):
                l = np.load(f)
                dict = {'loss': l, 'config': config}
                ls2 = np.append(ls2, dict)
    loss_min2 = min(ls2, key=lambda x: x['loss'])
    model_embedding.load_state_dict(torch.load(model_path + '/model_{}'.format(disease)))
    model_embedding.eval()


    #path = f'../data/data_test' +'/'
    X_whole_test = np.load(os.path.join(model_path, f'test_data_{disease}.npy'), allow_pickle=True)
    all_labels = np.load(os.path.join(model_path, f'test_label_{disease}.npy'), allow_pickle=True)
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
            elif name_model == 'ProdGamDirVae':
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

    return X_subtrain, Y_subtrain, Xs[0], X_subtest, Y_subtest, Xs[1], X_whole_test, Y_whole_test, Xs[2], out_shapes, model_embedding, X_subtrain_omics, X_subtest_omics, X_whole_test_omics, omics_shape

def model_forward(model, x_omics, omics_shape):
    if isinstance(model, (SASEvae, SASElap)):
        (view1_specific_em_new, view1_specific_mu_new, view1_specific_sigma_new, view1_shared_em_new,
        view2_specific_em_new, view2_specific_mu_new, view2_specific_sigma_new, view2_shared_em_new,
        view3_specific_em_new, view3_specific_mu_new, view3_specific_sigma_new, view3_shared_em_new,
        view1_specific_rec_new, view1_shared_rec_new, view2_specific_rec_new,
        view2_shared_rec_new, view3_specific_rec_new, view3_shared_rec_new,
        view1_shared_mlp_new, view2_shared_mlp_new, view3_shared_mlp_new) = (
            model(x_omics[:,:omics_shape[0]], 
            x_omics[:,omics_shape[0]:omics_shape[0]+omics_shape[1]],
            x_omics[:,omics_shape[0]+omics_shape[1]:]))
    else:
        view1_specific_em_new, view1_specific_alpha_new, view1_shared_em_new, view2_specific_em_new, \
        view2_specific_alpha_new, view2_shared_em_new, view3_specific_em_new, view3_specific_alpha_new, \
        view3_shared_em_new, view1_specific_rec_new, view1_shared_rec_new, view2_specific_rec_new, \
        view2_shared_rec_new, view3_specific_rec_new, view3_shared_rec_new, view1_shared_mlp_new, view2_shared_mlp_new, \
        view3_shared_mlp_new = model(x_omics[:,:omics_shape[0]],                    
            x_omics[:,omics_shape[0]:omics_shape[0]+omics_shape[1]],
            x_omics[:,omics_shape[0]+omics_shape[1]:])
    view_shared_common = (view1_shared_em_new + view2_shared_em_new + view3_shared_em_new) / 3
    final_embedding = torch.cat(
            (view1_specific_em_new, view2_specific_em_new, view3_specific_em_new, view_shared_common), dim=1)
    return final_embedding

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
    nearest_neighbors = knn.kneighbors(X_subtest)
    #print(f"kNN acc: {accuracy:.2f}")
    return nmi_, accuracy, nearest_neighbors

def gradient_distance(base_example, nearest_neighbors, model, omics_shape):
    grads = []
    embeddings_neighbors = model_forward(model, nearest_neighbors, omics_shape)
    for i in range(nearest_neighbors.shape[0]):
        embedding_base = model_forward(model, base_example.unsqueeze(0), omics_shape)
        embedding_neighbor = embeddings_neighbors[i].unsqueeze(0)
        norm = torch.nn.functional.mse_loss(embedding_base, embedding_neighbor, reduction='mean')
        grad = torch.autograd.grad(norm, base_example, retain_graph=False)[0]
        grads.append(grad.detach().cpu().numpy())
    grads = np.array(grads)
    return grads

def one_exp(disease, name_model):
    X_train, Y_subtrain, X_subtrain, Y_train, Y_subtest, X_subtest, _, Y_whole_test, X_whole_test, out_shapes, model, X_subtrain_omics, X_subtest_omics, X_whole_test_omics, omics_shape = get_data(name_model, disease)
    nmi, acc, nearest_neighbors = one_knn(X_subtrain, Y_subtrain, X_subtest, Y_subtest, X_whole_test, Y_whole_test, disease)
    gradient_distances_list = []
    for i in range(X_subtest_omics.shape[0]):
        base_example = X_subtest_omics[i].clone().detach().requires_grad_(True)
        neighbor_indices = nearest_neighbors[1][i]
        neighbors = X_subtrain_omics[neighbor_indices].clone().detach().requires_grad_(True)
        grads = gradient_distance(base_example, neighbors, model, omics_shape)
        gradient_distances_list.append(grads)
    gradient_distances = np.array(gradient_distances_list)
    return gradient_distances, nmi, acc, nearest_neighbors

def all_expes(disease):
    score_dicts = {}
    for name_model in ['ae', 'vae', 'lapdirvae', 'GammaDirVae', 'ProdGamDirVae']:
        print(name_model)
        X_train, Y_subtrain, X_subtrain, Y_train, Y_subtest, X_subtest, _, Y_whole_test, X_whole_test, out_shapes, model = get_data(name_model, disease)
    return model

# %%
gradient_distances, nmi, acc, nearest_neighbors = one_exp('brca', 'ProdGamDirVae')
# %%
import seaborn as sn
all_avgs_by_omics = []
nb_examples = gradient_distances.shape[0]
for index_test_value in range(nb_examples):
    print(Y_subtest[index_test_value])
    print([Y_subtrain[i] for i in nearest_neighbors[1][index_test_value]])
    #sn.heatmap(np.abs(gradient_distances[index_test_value]), cmap='viridis', vmin=0)
    #plt.show()
    split_by_omics = []
    curr_index = 0
    for k in omics_shape:
        split_by_omics.append(gradient_distances[index_test_value][:,curr_index:curr_index+k])
        curr_index += k

    avgs_by_omics = [np.abs(split_by_omics[i]).mean(1) for i in range(len(split_by_omics))]
    all_avgs_by_omics.append(np.stack(avgs_by_omics).T*1000)

#%%
# Create the grid of subplots
fig, axes = plt.subplots(9, 5, figsize=(10, 15))

vmax = np.concatenate(all_avgs_by_omics).max()

# Flatten axes for easy iteration
axes = axes.flatten()

# Keep a reference to one heatmap for the colorbar
hm = None

class_names = ["Normal-like", "Basal-like", "HER2-enriched", "Luminal A", "Luminal B"]

for index_test_value, ax in enumerate(axes[:len(all_avgs_by_omics)]):
    ax.set_title(f'Label: {class_names[Y_subtest[index_test_value]]}', fontsize=10)
    avgs_by_omics = all_avgs_by_omics[index_test_value]
    hm = sn.heatmap(
        avgs_by_omics,
        cmap='inferno',
        vmin=0,
        vmax=vmax,
        yticklabels=False,
        xticklabels=OMICS_NAMES[:-1] if index_test_value >= 39 else False,
        cbar=False,
        ax=ax
    )
fig.delaxes(axes[44])
# Add one big colorbar to the right
cbar_ax = fig.add_axes([0.25, -0.02, 0.5, 0.02])  # centered below
fig.colorbar(hm.collections[0], cax=cbar_ax, orientation='horizontal')

plt.tight_layout()
plt.show()
# %%
import csv

list_feat_names = csv.reader(open('./feat_names.csv', 'r'))
list_feat_names = [row[0] for row in list_feat_names]
# %%
index_test_value = 2
print(Y_subtest[index_test_value])
print([Y_subtrain[i] for i in nearest_neighbors[1][index_test_value]])
top_indices_arr = np.argsort(np.abs(gradient_distances[index_test_value]), axis=1)
top_indices = top_indices_arr[:,-10:].tolist()[::-1]
all_top_features = []
for neighbour in range(len(top_indices)):
    top_features = [list_feat_names[i] for i in top_indices[neighbour]]
    all_top_features.append(top_features)

# %%

# %%
np.sort(counts)
# %%
all_all_top_features = []
for index_test_value in range(nb_examples):
    print(Y_subtest[index_test_value])
    print([Y_subtrain[i] for i in nearest_neighbors[1][index_test_value]])
    top_indices = np.argsort(np.abs(gradient_distances[index_test_value]), axis=1)
    top_indices = top_indices[:,-10:].tolist()[::-1]
    all_top_features = []
    for neighbour in range(len(top_indices)):
        top_features = [list_feat_names[i] for i in top_indices[neighbour]]
        all_top_features += (top_features)
    all_all_top_features += (all_top_features)

# %%
all_appearing = list(set(all_all_top_features))
counts = []
for gene in all_appearing:
    count = all_all_top_features.count(gene)
    counts.append(count)
# %%
print(np.array(all_appearing)[np.argsort(counts)[-20:]])
print(np.array(counts)[np.argsort(counts)[-20:]])
# %%
relevant_ones = ['hsa-mir-187', 'hsa-mir-20b', 'hsa-mir-204']
indices_relevant = [all_appearing.index(gene) for gene in relevant_ones]
counts_relevant = [counts[idx] for idx in indices_relevant]
# %%
