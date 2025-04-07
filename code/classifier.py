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
from torchmetrics import ConfusionMatrix
from shap import KernelExplainer, sample
from shap.plots import waterfall
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
#%%
device = 'cpu'
name_model = 'lapdirvae'
omics_shape = [1000,1000,503]
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
                "ProdGamDirVae", 4, view_size=[omics_shape[0], omics_shape[1], omics_shape[2]],
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
params = {'vae':'0.0006_0.0004', 'ProdGammaDirVae': '0.0003_0.0005_4', 'ae': '0.0004_0.0007', 'GammaDirVae': '0.0003_0.0007', 'lapdirvae': '0.0006_0.0007'}
nb_classes = 5

model_embedding = model_sas.to(device)
path = '../results/models_brca_' + name_model + '/' + params[name_model] +'/'
model_embedding.load_state_dict(torch.load(path+'model_brca', weights_only=False))
X_test_omics = torch.from_numpy(np.load(path+'test_data_brca.npy')).float().to(device)
Y_test = torch.from_numpy(np.load(path+'test_label_brca.npy')).long().squeeze().to(device)
X_valid_omics = torch.from_numpy(np.load(path+'val_data_brca.npy')).float().to(device)
Y_valid = torch.from_numpy(np.load(path+'val_label_brca.npy')).long().squeeze().to(device)
X_train_omics = torch.from_numpy(np.load(path+'train_data_brca.npy')).float().to(device)
Y_train = torch.from_numpy(np.load(path+'train_label_brca.npy')).long().squeeze().to(device)


Xs = []
with torch.no_grad():
    for X_loader in [X_train_omics,X_valid_omics]:
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
        Xs.append(final_embedding)

X_train, X_valid = Xs
X_mu = torch.mean(X_train, dim=0)
X_std = torch.std(X_train, dim=0)

X_train = X_train-X_mu
X_train = X_train/X_std

X_valid = X_valid-X_mu
X_valid = X_valid/X_std

index0, index1 = 0,128
X_valid = X_valid[:,index0:index1]
X_train = X_train[:,index0:index1]
Xd = xgboost.XGBClassifier(0.1, max_depth=1)
Xd.fit(X_train, Y_train)
print(Xd.score(X_train, Y_train))
print(Xd.score(X_valid, Y_valid))

print("logistic_regression")
LR = LogisticRegression(penalty='l1', C=0.1, fit_intercept=False, solver='saga', multi_class='multinomial')
LR.fit(X_train, Y_train)
print(LR.score(X_train, Y_train))
print(LR.score(X_valid, Y_valid))

xs_separation = [sum(out_shapes[:i+1]) for i in range(3)]

plt.rcParams["figure.figsize"] = (10,25)
for i,c in enumerate(LR.coef_):
    plt.subplot(5,1,i+1)
    plt.grid()
    plt.plot(c)
    plt.vlines(x=xs_separation, linestyles="dotted", ymin=-1000, ymax=1000)
    plt.ylim(min(c)-0.05, max(c)+0.05)
    plt.title(f'class {i}')
    plt.xlabel('Feature Index')
    plt.ylabel('Weight Value')
#plt.show()
plt.savefig(f"weights_{name_model}.pdf")
plt.show()

for i,c in enumerate(LR.coef_):
    plt.subplot(5,1,i+1)
    plt.grid()
    d = np.abs(c)
    plt.plot(d)
    plt.vlines(x=xs_separation, linestyles="dotted", ymin=-1000, ymax=1000)
    plt.ylim(min(d)-0.05, max(d)+0.05)
    plt.title(f'class {i}')
    plt.xlabel('Feature Index')
    plt.ylabel('Weight Value')
#plt.show()
plt.savefig(f"abs_weights_{name_model}.pdf")
plt.show()

plt.rcParams["figure.figsize"] = (10,10)
xs_b = ([0]+xs_separation+[sum(out_shapes)+1])
nb_embs = len(xs_separation)+1
for i,c in enumerate(LR.coef_):
    plt.subplot(5,1,i+1)
    plt.grid()
    importances_per_emb = [sum(np.abs(c[xs_b[i]:xs_b[i+1]])) for i in range(nb_embs)]
    plt.bar(['Specific 1', 'Specific 2', 'Specific 3', 'Shared'], importances_per_emb)
    plt.title(f'class {i}')
    plt.xlabel('Feature Index')
    plt.ylabel('Summed Weigt Values')
#plt.show()
plt.savefig(f"abs_mod_weights_{name_model}.pdf")
plt.show()

# %%
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

"""
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
"""
linear_model = torch.nn.Linear(X_train.shape[1], nb_classes).to(device)
#model = MCC(list_models).to(device)
"""
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