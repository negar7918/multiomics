#%%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
from loading_data import load_data
from torch import nn
import util
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
#from neurhci.uhci import UHCI
#from neurhci.hierarchical import HCI2layers
#from neurhci.marginal_utilities import MonotonicSelector, MarginalUtilitiesLayer, IdentityClipped, Identity
import xgboost
import os

device = 'cpu'
disease = 'coad'

omics_shape = {'brca': [1000,1000,503], 'kric': [58315, 22928, 1879], 'lihc': [20530, 5000, 1046], 'coad': [17260, 19052, 375]}[disease]
group_numbers = {'brca': 4, 'coad': 5, 'lihc':4, 'kirc': 4}[disease]
nb_classes = {
    'brca': 5,
    'lihc': 2,
    'coad': 4,
    'kric': 2}[disease]

def get_data(name_model):
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
                    "ProdGamDirVae", group_numbers, view_size=[omics_shape[0], omics_shape[1], omics_shape[2]],
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
    path = f'../results/data_{disease}' +'/'
    data_base = load_data('coad')
    X_train, X_test, y_train, y_test = train_test_split(data_base[3], data_base[4], test_size=0.2, random_state=1)

    X_test_omics = torch.from_numpy(X_test.astype(float)).float().to(device)
    Y_test = y_test.astype(int)
    X_train_omics = torch.from_numpy(X_train.astype(float)).float().to(device)
    Y_train = y_train.astype(int)

    Xs = []
    with torch.no_grad():
        for X_loader in [X_train_omics, X_test_omics]:
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

    return X_train, Y_train, Xs[0], X_test, Y_test, Xs[1], out_shapes

#%%
OMICS_NAMES = {'brca': ['mRNA', 'DNAmethyl', 'miRNA', 'Shared'], 'kric':['gene1', 'methyl', 'miRNA1', 'Shared'], 'lihc':['gene', 'methyl', 'miRNA', 'Shared'], 'coad':['mRNA', 'Methy', 'miRNA', 'Shared']}[disease]
Separations = [32, 64, 96]

def extract_omics(x, omics):
    xs = [x[:,:32]*1., x[:,32:64]*1., x[:,64:96]*1., x[:,96:]*1.]
    return np.concatenate([xs[i] for i in omics], axis=1)

def one_knn(X_train, Y_train, X_test, Y_test):
    best_inertia = float("inf")
    best_labels = None
    for i in range(30):
        kmeans = KMeans(n_clusters=nb_classes, init='k-means++', random_state=i)
        labels = kmeans.fit_predict(X_test)
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_labels = labels
    nmi_, ari_, f_score_, acc_, v_, ch = evaluation.evaluate(Y_test, best_labels)
    print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f,  '
                            'ch_index: %.4f  <==|' % (nmi_, ari_, f_score_, acc_, v_, ch))
    
    knn = KNeighborsClassifier(n_neighbors=nb_classes)
    # Train the model
    knn.fit(X_train, Y_train)
    # Predict on test set
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"kNN acc: {accuracy:.2f}")
    return nmi_, accuracy

def one_ablation(X_train, Y_train, X_test, Y_test, which_omics=[0,1,2,3]):
    xe = extract_omics(X_test, which_omics)
    xr = extract_omics(X_train, which_omics)
    print(xe.shape, xr.shape)
    nmi, acc = one_knn(xr, Y_train, xe, Y_test)
    return nmi, acc

def all_ablations(X_train, Y_train, X_test, Y_test, name_model):
    log_file = f'results/logs_{disease}.txt'
    scores = []
    for omics_list in [[0,1,2,3], [0], [1], [2], [3], [0,1,2], [0,1,3], [0,2,3], [1,2,3]]:
        print(omics_list)
        nmi, acc = one_ablation(X_train, Y_train, X_test, Y_test, omics_list)
        names = ', '.join([OMICS_NAMES[i] for i in omics_list])
        scores.append(f'{names}: {acc}\n')
    with open(log_file, 'a') as file:
        file.write(name_model + '\n')
        for s in scores:
            file.write(s)
        file.write('\n')

def all_expes():
    for name_model in ['ae', 'ProdGammaDirVae', 'vae', 'GammaDirVae', 'lapdirvae']:
        X_train, Y_train, _, X_test, Y_test, _, out_shapes = get_data(name_model)
        all_ablations(X_train, Y_train, X_test, Y_test, name_model)

all_expes()

# %%
'''

def one_ablation(Xtr, Y_train, Xval, Y_valid, Xtest, Y_test, name_model, out_shapes, which_omics=[0,1,2,3]):

    X_train = extract_omics(Xtr, which_omics)
    X_valid = extract_omics(Xval, which_omics)
    X_test = extract_omics(Xtest, which_omics)

    score, LR = best_lr(X_train, Y_train, X_valid, Y_valid)
    test_score = LR.score(X_test, Y_test)
    test_score = int(test_score*100)/100
    names = '-'.join([OMICS_NAMES[o] for o in which_omics])

    results_dir = os.path.join('results', disease, name_model, names)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    plt.rcParams["figure.figsize"] = (10,25)
    for i,c in enumerate(LR.coef_):
        plt.subplot(nb_classes,1,i+1)
        plt.grid()
        plt.plot(c)
        plt.vlines(x=Separations, linestyles="dotted", ymin=-1000, ymax=1000)
        plt.ylim(min(c)-0.05, max(c)+0.05)
        plt.title(f'class {i}')
        plt.xlabel('Feature Index')
        plt.ylabel(f'Weight Value')
    #plt.show()
    plt.savefig(f"{results_dir}/weights.pdf")
    plt.clf()
    #plt.show()

    for i,c in enumerate(LR.coef_):
        plt.subplot(nb_classes,1,i+1)
        plt.grid()
        d = np.abs(c)
        plt.plot(d)
        plt.vlines(x=Separations, linestyles="dotted", ymin=-1000, ymax=1000)
        plt.ylim(min(d)-0.05, max(d)+0.05)
        plt.title(f'class {i}')
        plt.xlabel('Feature Index')
        plt.ylabel(f'Weight Value')
    #plt.show()
    plt.savefig(f"{results_dir}/abs_weights.pdf")
    plt.clf()
    #plt.show()

    subtypes = {'brca':{0: "normal-like", 1: "basal", 2: "HER2-enriched", 3: "LumA", 4: "LumB"}, 'coad': {0: 'CIN', 1: 'GS', 2:'MSI', 3:'POLE'}}[disease]
    plt.rcParams["figure.figsize"] = (10, 10)
    xs_b = ([0] + Separations + [sum(out_shapes) + 1])
    nb_embs = len(Separations) + 1
    norm_coef = np.sum(LR.coef_)
    Importances = []
    max_overall = 0.
    for i, c in enumerate(LR.coef_):
        plt.subplot(nb_classes, 1, i + 1)
        plt.grid()
        importances_per_emb = [sum(np.abs(c[xs_b[i]:xs_b[i + 1]])) for i in range(nb_embs)]
        Importances.append(importances_per_emb)
        max_overall = max(max_overall, max(importances_per_emb))
    for i, c in enumerate(Importances):
        plt.subplot(nb_classes, 1, i + 1)
        print(c)
        print(max_overall)
        x_norm = c / max_overall
        plt.bar(OMICS_NAMES, x_norm)
        plt.ylim(0, 1)
        plt.title(subtypes[i]) #(f'class {i}')
        plt.ylabel(f'Summed Weight Values')    #plt.show()
    plt.savefig(f"{results_dir}/abs_mod_weights.pdf")
    plt.clf()
    #plt.show()
    return test_score,names,LR,X_train,X_valid,X_test

def all_ablations(Xtr, Y_train, Xval, Y_valid, X_test, Y_test, name_model, out_shapes):
    log_file = f'results/logs_{disease}.txt'
    scores = []
    for omics_list in [[0,1,2,3], [0], [1], [2], [3], [0,1,2], [0,1,3], [0,2,3], [1,2,3]]:
        print(omics_list)
        score,names,_,_,_,_ = one_ablation(Xtr, Y_train, Xval, Y_valid, X_test, Y_test, name_model, out_shapes, omics_list)
        scores.append(f'{names}: {score}\n')
    with open(log_file, 'a') as file:
        file.write(name_model + '\n')
        for s in scores:
            file.write(s)
        file.write('\n')

def all_expes():
    for name_model in ['ae', 'ProdGammaDirVae', 'vae', 'GammaDirVae', 'lapdirvae']:
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test,out_shapes = get_data(name_model)
        all_ablations(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, name_model, out_shapes)
'''

"""
def make_mlp(in_dim, hidden_dim, out_dim, nb_layers, activation, dropout, norm):
    assert nb_layers>1
    lays = []
    for l in range(nb_layers):
        dim_in = in_dim if l==0 else hidden_dim
        dim_out = out_dim if (l==nb_layers-1) else hidden_dim
        lays.append(nn.Linear(dim_in, dim_out))
        if (l != nb_layers-1):
            lays.append(activation())
            if norm is not None:
                lays.append(norm(dim_out))
            if dropout > 0.:
                lays.append(nn.Dropout(dropout))
    return nn.Sequential(*lays)

mlp_model = make_mlp(
    in_dim=X_valid.shape[1],
    hidden_dim=64,
    out_dim=nb_classes,
    nb_layers=2,
    activation=torch.nn.LeakyReLU,
    dropout=0.,
    norm=None#torch.nn.LayerNorm
).to(device)

list_models = []
for i in range(nb_classes):
    utilities = {i:MonotonicSelector for i in range(128)}
    hierarchy = {-1:[128,129,130,131], 128:list(range(32)), 129:list(range(32,64)), 130:list(range(64, 96)), 131:list(range(96,128))}
    margut = MarginalUtilitiesLayer(list_of_leaves=list(range(128)), types_of_leaves=utilities, nb_sigmoids=3)
    list_models.append(UHCI(hierarchy=hierarchy, marginal_utilities=margut))

class MCC(nn.Module):
    def __init__(self, list_models):
        super().__init__()
        self.elements = torch.nn.ModuleList(list_models)
    
    def forward(self, x):
        outs = [modu(x) for modu in self.elements]
        return torch.cat(outs, 1)

linear_model = torch.nn.Linear(X_train.shape[1], nb_classes).to(device)
#model = MCC(list_models).to(device)
def accuracy(A,B):
    return(torch.sum(A==B)/len(A))

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.75)
for epoch in range(3000):
    mlp_model.train()
    optimizer.zero_grad()
    out = mlp_model(X_train)
    out = torch.nn.functional.log_softmax(out, dim=1)
    target = Y_train
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()
    print("train")
    print(loss.item())
    if not (epoch+1)%1:
        print(epoch+1)
        with torch.no_grad():
            mlp_model.eval()
            #print("test:")
            pred_train = torch.argmax(mlp_model(X_train), dim=1)
            print(accuracy(pred_train, Y_train))
            pred_valid = torch.argmax(mlp_model(X_valid), dim=1)
            print(accuracy(pred_valid, Y_valid))
# %%
torch.save(mlp_model, 'mlp_model_classif')
for i in range(5):
    w = linear_model.weight[i].detach().cpu().numpy()
    plt.plot(w, label=f'class {i}')
    plt.show()
# %%
pred_train = torch.argmax(mlp_model(X_train), dim=1)
ConfMat = ConfusionMatrix("multiclass", num_classes=5).to(device)
print(ConfMat(pred_train, Y_train))
pred_valid = torch.argmax(mlp_model(X_valid), dim=1)
print(ConfMat(pred_valid, Y_valid))


# %%
mlp_model = mlp_model.cpu()
def model_wrapper(x):
    with torch.no_grad():
        x = torch.from_numpy(x)
        out = mlp_model(x)
        out = torch.softmax(x, dim=1)
    return out.detach().numpy()

X_bg = X_train.detach().cpu().numpy()
background = sample(X_bg, 5)

X_in = X_valid.detach().cpu().numpy()[:1]
explainer = KernelExplainer(model_wrapper, background)
# %%
Z = explainer(X_in)
# %%
from ViaSHAP.kanshap import KANSHAP
# %%

"""