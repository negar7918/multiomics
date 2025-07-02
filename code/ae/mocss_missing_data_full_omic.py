import numpy as np
from multiomics.code.loading_data import load_data
from torch import nn
from multiomics.code import util
import torch.nn.functional as F
import torch.optim
import torch.nn.init as init
from multiomics.code.contrastive_loss import InstanceLoss
import argparse
from sklearn.cluster import KMeans
import multiomics.code.evaluation as evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SharedAndSpecificLoss(nn.Module):
    def __init__(self):
        super(SharedAndSpecificLoss, self).__init__()

    @staticmethod
    def orthogonal_loss(shared, specific):
        shared = shared - shared.mean()
        specific = specific - specific.mean()
        shared = F.normalize(shared, p=2, dim=1)
        specific = F.normalize(specific, p=2, dim=1)
        # This is not inner product but they use it because the get better result
        correlation_matrix = torch.mul(shared, specific)
        cost = correlation_matrix.mean()
        return cost

    @staticmethod
    def contrastive_loss(shared_1, shared_2, temperature, batch_size):
        assert (shared_1.dim() == 2)
        assert (shared_2.dim() == 2)
        shared_1 = shared_1 - shared_1.mean()
        shared_2 = shared_2 - shared_2.mean()
        shared_1 = F.normalize(shared_1, p=2, dim=1)
        shared_2 = F.normalize(shared_2, p=2, dim=1)
        criterion_instance = InstanceLoss(batch_size=batch_size, temperature=temperature)
        loss = criterion_instance(shared_1, shared_2)
        return loss

    @staticmethod
    def reconstruction_loss(rec, ori):
        assert (rec.dim() == 2)
        assert (ori.dim() == 2)
        rec = rec - rec.mean()
        ori = ori - ori.mean()
        rec = F.normalize(rec, p=2, dim=1)
        ori = F.normalize(ori, p=2, dim=1)
        # This is eucleadian norm: sqrt(sum(a_ij^2))
        loss = torch.linalg.matrix_norm(rec-ori)
        return loss

    def forward(self, params):
        (shared1_output, shared2_output, shared3_output, specific1_output, specific2_output, specific3_output,
        shared1_rec, shared2_rec, shared3_rec, specific1_rec, specific2_rec, specific3_rec, ori1, ori2, ori3,
        shared1_mlp, shared2_mlp, shared3_mlp, temperature, batch_size) = params

        # orthogonal restrict
        orthogonal_loss1 = self.orthogonal_loss(shared1_output, specific1_output)
        orthogonal_loss2 = self.orthogonal_loss(shared2_output, specific2_output)
        orthogonal_loss3 = self.orthogonal_loss(shared3_output, specific3_output)
        orthogonal_loss_all = orthogonal_loss1 + orthogonal_loss2 + orthogonal_loss3

        # Contrastive Loss
        contrastive_loss1 = self.contrastive_loss(shared1_mlp, shared2_mlp, temperature, batch_size)
        contrastive_loss2 = self.contrastive_loss(shared1_mlp, shared3_mlp, temperature, batch_size)
        contrastive_loss3 = self.contrastive_loss(shared2_mlp, shared3_mlp, temperature, batch_size)
        contrastive_loss_all = contrastive_loss1 + contrastive_loss2 + contrastive_loss3

        # reconstruction Loss
        reconst_loss1 = self.reconstruction_loss(shared1_rec, ori1) + self.reconstruction_loss(specific1_rec, ori1)
        reconst_loss2 = self.reconstruction_loss(shared2_rec, ori2) + self.reconstruction_loss(specific2_rec, ori2)
        reconst_loss3 = self.reconstruction_loss(shared3_rec, ori3) + self.reconstruction_loss(specific3_rec, ori3)
        reconstruction_loss_all = reconst_loss1 + reconst_loss2 + reconst_loss3

        loss_total = orthogonal_loss_all + contrastive_loss_all + .7 * reconstruction_loss_all

        return loss_total


class SharedAndSpecificEmbedding(nn.Module):
    def __init__(self, view_size, n_units_1, n_units_2, n_units_3, mlp_size):
        super(SharedAndSpecificEmbedding, self).__init__()

        units = {'1': n_units_1, '2': n_units_2, '3': n_units_3}
        # This dictionary is used to store NN-lineartransformations,
        # where each key follows this pattern [shared/specific]+[omic_view]+'[encoder/decoder_layers]'
        linears = {}
        # Views are i in [1, 2, 3] for three omics data
        for i in range(1,4):
            j = str(i)
            for type in ['shared', 'specific']:
                # encoder (4 layers)
                linears[type+j+'_l1'] = nn.Linear(view_size[i-1], units[j][0])
                linears[type+j+'_l2'] = nn.Linear(units[j][0], units[j][1])
                linears[type+j+'_l3'] = nn.Linear(units[j][1], units[j][2])
                linears[type+j+'_l4'] = nn.Linear(units[j][2], units[j][3])
                # decoder  (4 layers)
                linears[type+j+'_l3_'] = nn.Linear(units[j][3], units[j][2])
                linears[type+j+'_l2_'] = nn.Linear(units[j][2], units[j][1])
                linears[type+j+'_l1_'] = nn.Linear(units[j][1], units[j][0])
                linears[type+j+'_rec'] = nn.Linear(units[j][0], view_size[i-1])

            linears['view'+j+'_mlp1'] = nn.Linear(units[j][3], mlp_size[0])
            linears['view'+j+'_mlp2'] = nn.Linear(mlp_size[0], mlp_size[1])

        # The order is very important since _modules is an OrderedDictionary
        self.shared1_l1, self.shared1_l2, self.shared1_l3, self.shared1_l4, self.shared1_l3_, self.shared1_l2_, \
            self.shared1_l1_, self.shared1_rec, self.specific1_l1, self.specific1_l2, self.specific1_l3, self.specific1_l4,\
            self.specific1_l3_, self.specific1_l2_, self.specific1_l1_, self.specific1_rec, \
            self.shared2_l1, self.shared2_l2, self.shared2_l3, self.shared2_l4, self.shared2_l3_, self.shared2_l2_, \
            self.shared2_l1_, self.shared2_rec, self.specific2_l1, self.specific2_l2, self.specific2_l3, self.specific2_l4,\
            self.specific2_l3_, self.specific2_l2_, self.specific2_l1_, self.specific2_rec, \
            self.shared3_l1, self.shared3_l2, self.shared3_l3, self.shared3_l4, self.shared3_l3_, self.shared3_l2_, \
            self.shared3_l1_, self.shared3_rec, self.specific3_l1, self.specific3_l2, self.specific3_l3, self.specific3_l4,\
            self.specific3_l3_, self.specific3_l2_, self.specific3_l1_, self.specific3_rec, self.view1_mlp1, self.view1_mlp2,\
            self.view2_mlp1, self.view2_mlp2, self.view3_mlp1, self.view3_mlp2 \
            = linears['shared1_l1'], linears['shared1_l2'], linears['shared1_l3'], linears['shared1_l4'], \
              linears['shared1_l3_'], linears['shared1_l2_'], linears['shared1_l1_'], linears['shared1_rec'], \
              linears['specific1_l1'], linears['specific1_l2'], linears['specific1_l3'], linears['specific1_l4'],\
              linears['specific1_l3_'], linears['specific1_l2_'], linears['specific1_l1_'], linears['specific1_rec'], \
              linears['shared2_l1'], linears['shared2_l2'], linears['shared2_l3'], linears['shared2_l4'], \
              linears['shared2_l3_'], linears['shared2_l2_'], linears['shared2_l1_'], linears['shared2_rec'], \
              linears['specific2_l1'], linears['specific2_l2'], linears['specific2_l3'], linears['specific2_l4'],\
              linears['specific2_l3_'], linears['specific2_l2_'], linears['specific2_l1_'], linears['specific2_rec'], \
              linears['shared3_l1'], linears['shared3_l2'], linears['shared3_l3'], linears['shared3_l4'], \
              linears['shared3_l3_'], linears['shared3_l2_'], linears['shared3_l1_'], linears['shared3_rec'], \
              linears['specific3_l1'], linears['specific3_l2'], linears['specific3_l3'], linears['specific3_l4'], \
              linears['specific3_l3_'], linears['specific3_l2_'], linears['specific3_l1_'], linears['specific3_rec'], \
              linears['view1_mlp1'], linears['view1_mlp2'], linears['view2_mlp1'], linears['view2_mlp2'], \
              linears['view3_mlp1'], linears['view3_mlp2']

        def init_weights(s):
            original = s.__dict__
            d = s.__dict__.get('_modules')
            for k, v in d.items():
                init.kaiming_normal_(v.weight)
            # items() creates a new view object and therefore we need to update the original dictionary
            original.update({'_modules': d})

        # Init weight
        init_weights(self)

    def forward(self, view1_input, view2_input, view3_input):
        def encoder_decoder(layers, input_data):
            l1, l2, l3, l4, l3_,l2_, l1_, rec = layers
            # We give batch_size x original_features_dim
            out = F.tanh(l1(input_data))
            out = F.tanh(l2(out))
            out = F.tanh(l3(out))
            # We get batch_size x embedding_dim
            em = F.tanh(l4(out))
            out = F.tanh(l3_(em))
            out = F.tanh(l2_(out))
            out = F.tanh(l1_(out))
            # We get batch_size x original_features_dim
            reconstructed = torch.sigmoid(rec(out))
            return em, reconstructed

        # View1
        l1_specific = (self.specific1_l1, self.specific1_l2, self.specific1_l3, self.specific1_l4, self.specific1_l3_,
                       self.specific1_l2_, self.specific1_l1_, self.specific1_rec)

        view1_specific_em, view1_specific_rec = encoder_decoder(l1_specific, view1_input)
        l1_shared= (self.shared1_l1, self.shared1_l2, self.shared1_l3, self.shared1_l4, self.shared1_l3_,
                    self.shared1_l2_, self.shared1_l1_, self.shared1_rec)
        view1_shared_em, view1_shared_rec = encoder_decoder(l1_shared, view1_input)
        view1_shared_mlp = F.tanh(self.view1_mlp1(view1_shared_em))
        view1_shared_mlp = F.tanh(self.view1_mlp2(view1_shared_mlp))

        # View2
        l2_specific = (self.specific2_l1, self.specific2_l2, self.specific2_l3, self.specific2_l4, self.specific2_l3_,
                       self.specific2_l2_, self.specific2_l1_, self.specific2_rec)
        view2_specific_em, view2_specific_rec = encoder_decoder(l2_specific, view2_input)
        l2_shared= (self.shared2_l1, self.shared2_l2, self.shared2_l3, self.shared2_l4, self.shared2_l3_,
                    self.shared2_l2_, self.shared2_l1_, self.shared2_rec)
        view2_shared_em, view2_shared_rec = encoder_decoder(l2_shared, view2_input)
        view2_shared_mlp = F.tanh(self.view2_mlp1(view2_shared_em))
        view2_shared_mlp = F.tanh(self.view2_mlp2(view2_shared_mlp))

        # View3
        l3_specific = (self.specific3_l1, self.specific3_l2, self.specific3_l3, self.specific3_l4, self.specific3_l3_,
                       self.specific3_l2_, self.specific3_l1_, self.specific3_rec)
        view3_specific_em, view3_specific_rec = encoder_decoder(l3_specific, view3_input)
        l3_shared = (self.shared3_l1, self.shared3_l2, self.shared3_l3, self.shared3_l4, self.shared3_l3_,
                     self.shared3_l2_, self.shared3_l1_, self.shared3_rec)
        view3_shared_em, view3_shared_rec = encoder_decoder(l3_shared, view3_input)
        view3_shared_mlp = F.tanh(self.view3_mlp1(view3_shared_em))
        view3_shared_mlp = F.tanh(self.view3_mlp2(view3_shared_mlp))

        return view1_specific_em, view1_shared_em, view2_specific_em, view2_shared_em, view3_specific_em, \
               view3_shared_em, view1_specific_rec, view1_shared_rec, view2_specific_rec, view2_shared_rec, \
               view3_specific_rec, view3_shared_rec, view1_shared_mlp, view2_shared_mlp, view3_shared_mlp


def reconst_miRNA(view3_test_data, view3_specific_rec_new, path):
    original_values_3 = view3_test_data.detach().numpy()
    recon_values_3 = view3_specific_rec_new.detach().numpy()

    # # Compute distance from x = y line
    distance = np.abs(original_values_3 - recon_values_3)
    # Set a threshold for "closeness"
    threshold = 0.05
    close_mask = distance < threshold

    # Find coordinates where mask is True
    close_coords = np.argwhere(close_mask)

    # Randomly sample a subset of those coordinates
    num_points_to_plot = 300  # adjust as needed
    if len(close_coords) > num_points_to_plot:
        sampled_indices = np.random.choice(len(close_coords), num_points_to_plot, replace=False)
        sampled_coords = close_coords[sampled_indices]
    else:
        sampled_coords = close_coords

    # Extract corresponding values
    x_vals = original_values_3[sampled_coords[:, 0], sampled_coords[:, 1]]
    y_vals = recon_values_3[sampled_coords[:, 0], sampled_coords[:, 1]]

    # Plot
    correlation = np.corrcoef(x_vals, y_vals)[0, 1]
    # Regression line (least squares fit)
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    regression_line = np.polyval([slope, intercept], x_vals)
    plt.scatter(x_vals, y_vals, alpha=.5, s=10, c='black', label=f'corr = {correlation:.2f}')
    plt.plot([x_vals.min(), x_vals.max()], [x_vals.min(), x_vals.max()], c='red', label='x = y')
    # Plot the regression line
    plt.plot(x_vals, regression_line, 'r-', label=f'Regression Line', c='blue', alpha=.5)

    plt.xlabel("Original Values")
    plt.ylabel("Reconstructed Values")
    plt.title("Reconstruction of miRNA using MOCSS (AE)")
    plt.legend()
    plt.savefig(path + '/recon_miRNA.png')


def main(args):
    disease = 'brca' #'kirc' 'coad' 'lihc'
    view1_data, view2_data, view3_data, view_train_concatenate, y_true = load_data(disease)

    # Build Model
    model = SharedAndSpecificEmbedding(
            view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
            n_units_1=[512, 256, 128, 32], n_units_2=[512, 256, 128, 32],
            n_units_3=[256, 128, 64, 32], mlp_size=[32, 8]
        )

    # Load test data
    ls = [{'loss': 100000000, 'config': 'test'}]
    path = ('../../results/models_'+disease+'_ProdGammaDirVae')
    import os
    for (dir_path, dir_names, file_names) in os.walk(path):
        for config in dir_names:
            name = os.path.join(config, 'loss.npy')
            f = os.path.join(path, name)
            if os.path.exists(f):
                l = np.load(f)
                dict = {'loss': l, 'config': config}
                ls = np.append(ls, dict)
    loss_min = min(ls, key=lambda x: x['loss'])
    folder = loss_min['config']
    all_params = {
        'brca': {'vae': '0.0006_0.0004', 'ProdGammaDirVae': '0.0003_0.0005_4', 'ae': '0.0004_0.0007',
                 'GammaDirVae': '0.0003_0.0007', 'lapdirvae': '0.0006_0.0007'},
        'lihc': {'ae': '0.0002_0.0007', 'GammaDirVae': '0.0003_0.0006', 'lapdirvae': '0.0002_0.0005',
                 'ProdGammaDirVae': '0.0005_0.0007_4', 'vae': '0.0005_0.0007'},
        'kirc': {'ae': '0.0002_0.0007', 'GammaDirVae': '0.0001_0.0006', 'lapdirvae': '0.0002_0.0005',
                 'ProdGammaDirVae': '0.0003_0.0005_4', 'vae': '0.0003_0.0007'},
        'coad': {'ae': '0.0002_0.0007', 'GammaDirVae': '0.0001_0.0006', 'lapdirvae': '0.0001_0.0006',
                 'ProdGammaDirVae': '0.0002_0.0003_5', 'vae': '0.0002_0.0006'}}
    folder = all_params[disease]['ProdGammaDirVae']
    desired_path = os.path.join(path, folder)
    data = np.load(desired_path + '/test_data_{}.npy'.format(disease))
    label = np.load(desired_path + '/test_label_{}.npy'.format(disease), allow_pickle=True)

    print(folder)

    # Load model
    ls2 = [{'loss': 100000000, 'config': 'test'}]
    path2 = ('../../results/models_'+disease+'_ae')
    import os
    for (dir_path, dir_names, file_names) in os.walk(path2):
        for config in dir_names:
            name = os.path.join(config, 'loss.npy')
            f = os.path.join(path2, name)
            if os.path.exists(f):
                l = np.load(f)
                dict = {'loss': l, 'config': config}
                ls2 = np.append(ls2, dict)
    loss_min2 = min(ls2, key=lambda x: x['loss'])
    folder2 = loss_min2['config']
    folder2 = all_params[disease]['ae']
    model_path = os.path.join(path2, folder2)
    model.load_state_dict(torch.load(model_path + '/model_{}'.format(disease)))
    model.eval()

    print('\n The name of the folder, that is under the directory name "models_brca_ae", for the reconstruction plot is "{0}".'.format(folder2))

    setup_seed(2)

    # get the omics
    view1_test_data = data[:, :view1_data.shape[1]]
    view1_test_data = torch.tensor(view1_test_data, dtype=torch.float32).clone().detach()
    view2_test_data = data[:, view1_data.shape[1]:view1_data.shape[1] + view2_data.shape[1]]
    view2_test_data = torch.tensor(view2_test_data, dtype=torch.float32).clone().detach()
    view3_test_data = data[:, view1_data.shape[1] + view2_data.shape[1]:]
    view3_test_data = torch.tensor(view3_test_data, dtype=torch.float32).clone().detach()

    # Introduce missing values
    # full miRNA omic removal
    view3_test_data_missing = view3_test_data.clone()
    # Set missing values to zero
    view3_test_data_missing[:, :] = 0

    view1_specific_em_new, view1_shared_em_new, view2_specific_em_new, \
         view2_shared_em_new, view3_specific_em_new,  \
        view3_shared_em_new, view1_specific_rec_new, view1_shared_rec_new, view2_specific_rec_new, \
        view2_shared_rec_new, view3_specific_rec_new, view3_shared_rec_new, view1_shared_mlp_new, view2_shared_mlp_new, \
        view3_shared_mlp_new = model(view1_test_data, view2_test_data, view3_test_data_missing)

    reconst_miRNA(view3_test_data, view3_specific_rec_new, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Method Running')
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    main(args)
