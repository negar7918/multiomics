import torch.optim
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from training import train, validation, EarlyStopper
from multiomics.code_two_omics.loading_data import load_data
from lap_dirvae_cancer import setup_seed, SharedAndSpecificLoss, SharedAndSpecificEmbedding
import os
import warnings
warnings.filterwarnings("ignore")


EPOCHS = 100
LR = [.0003, .0002, .0001, .0004, .0005, .0006] # .0007
BATCH_SIZE = 32
USE_GPU = False
ADJ_PARAMETER = 10 # TODO: adjust it for the dataset
MODEL = "standard"
WEIGHT_DECAY = [5e-4, 4e-4, 3e-4, 6e-4, 7e-4]
#Beta = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 2] is it really necessary? BetaVAE
SEED = 21


def work(p):
    setup_seed(SEED)
    batch, epochs, lr, wd = p
    loss_best = np.inf
    model_path = ''
    early_stopper = EarlyStopper(patience=30, min_delta=10)
    n_clusters = 2
    temperature = 0.4
    method = "lap_dirvae"
    disease = 'lihc'

    view1_data, view2_data, view_train_concatenate, y_true = load_data(disease)

    model = SharedAndSpecificEmbedding(view_size=[view1_data.shape[1], view2_data.shape[1]],
        n_units_1=[512, 256, 128, 32], n_units_2=[512, 256, 128, 32], mlp_size=[32, 8]
    )


    if USE_GPU:
        model = model.cuda()

    # creating directory to save each model
    directory = str(lr) + '_' + str(wd)
    parent_dir = "../../results/models_"+disease+"_lapdirvae"
    path = os.path.join(parent_dir, directory)
    os.makedirs(path, exist_ok=True)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(view_train_concatenate, y_true, test_size=0.2,
                                                        random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    # Data Loader for easy mini-batch return in training
    train_loader = torch.utils.data.DataLoader(dataset=X_train, batch_size=batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=X_val, batch_size=batch, shuffle=False)

    if USE_GPU:
        model = model.cuda()

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_function = SharedAndSpecificLoss()


    for epoch in range(epochs):
        # train
        train(n_clusters, # for Blie version
              train_loader, view1_data, view2_data, model, loss_function, temperature, optimizer, epoch, USE_GPU, epochs)

        # validation
        loss_val = validation(n_clusters, # for Blie version
              val_loader, view1_data, view2_data, model, temperature, loss_function, USE_GPU, epoch, epochs)

        print("epoch {0} done".format(epoch))

        # track the best performance and save the model
        if loss_val < loss_best:
            loss_best = loss_val
            model_path = '{}/model_{}_epoch_{}'.format(path, disease, epoch)

        # early stopping
        if early_stopper.early_stop(loss_val):
            break

    torch.save(model.state_dict(), model_path)
    # the below is needed for later reading the file where we don't know the epoch in the file-name as above
    torch.save(model.state_dict(), '{}/model_{}'.format(path, disease))
    np.save('{}/train_data_{}'.format(f'../../results/data_{disease}', disease), X_train)
    np.save('{}/train_label_{}'.format(f'../../results/data_{disease}', disease), y_train)
    np.save('{}/val_data_{}'.format(f'../../results/data_{disease}', disease), X_val)
    np.save('{}/val_label_{}'.format(f'../../results/data_{disease}', disease), y_val)
    np.save('{}/test_data_{}'.format(f'../../results/data_{disease}', disease), X_test)
    np.save('{}/test_label_{}'.format(f'../../results/data_{disease}', disease), y_test)
    np.save('{}/loss'.format(f'../../results/data_{disease}'), loss_best)


def main(args):
    batch = args.batch_size
    epochs = args.epochs
    if not USE_GPU:
        pool = torch.multiprocessing.Pool(10)
        param = [(batch, epochs, lr, wd) for lr in LR for wd in WEIGHT_DECAY]
        pool.map(work, param)
        pool.close()
    else:
        for lr in LR:
            for wd in WEIGHT_DECAY:
                p = (batch, epochs, lr, wd)
                work(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning')
    parser.add_argument("--epochs", type=int, help='The number of epochs to train', default=EPOCHS)
    parser.add_argument("--batch-size", type=int, help='Batch-size', default=BATCH_SIZE)

    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    parser.add_argument('gpu' if torch.cuda.is_available() else "cpu", type=str, help='Whether to use GPU')
    main(args)
