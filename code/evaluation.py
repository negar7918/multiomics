import numpy as np
from sklearn import metrics
from munkres import Munkres
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import LeaveOneOut


def evaluate(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(pred_adjusted, label)
    v_measure = metrics.v_measure_score(label, pred)
    if type(label) is pd.Series:
        label = label.to_numpy()
    ch_index = metrics.calinski_harabasz_score(label.reshape(-1,1), pred)
    return nmi, ari, f, acc, v_measure, ch_index


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred

def auc_sig_tests(X_test, y_test, X_train, y_train, num_clust):

    knn = KNeighborsClassifier(n_neighbors=num_clust)

    # Train the model
    knn.fit(X_train, y_train)

    # Find classes present in BOTH train and test
    classes_in_test = np.unique(y_test)
    classes_in_both = np.intersect1d(knn.classes_, classes_in_test)

    # Keep only samples belonging to those classes
    mask = np.isin(y_test, classes_in_both)
    y_test_filtered = y_test[mask]
    X_test_filtered = X_test[mask]

    # Get proba columns for valid classes
    class_indices = [list(knn.classes_).index(c) for c in classes_in_both]
    y_prob_filtered = knn.predict_proba(X_test_filtered)[:, class_indices]

    # Renormalize so each row sums to 1.0
    y_prob_filtered = y_prob_filtered / y_prob_filtered.sum(axis=1, keepdims=True)

    if num_clust == 2:
        auc_score = roc_auc_score(
            y_test_filtered,
            y_prob_filtered[:, 0],
            multi_class='ovr',
            labels=classes_in_both
        )

    else:
        auc_score = roc_auc_score(
            y_test_filtered,
            y_prob_filtered,
            multi_class='ovr',
            labels=classes_in_both
        )
    print(f"AUC Score: {auc_score:.4f}")

    k_values = [3, num_clust, 5, 7, 9]

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    results = {}

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)

        scores = cross_val_score(
            knn,
            X_train, y_train,
            cv=cv,
            scoring='f1_macro'
        )

        results[k] = scores.mean()

    print(k_values)
    print(results)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    score, perm_scores, p_value = permutation_test_score(
        knn, X_test, y_test,
        cv=cv,
        n_permutations=1000,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )

    print(f"Observed accuracy : {score:.4f}")
    print(f"Null mean accuracy: {perm_scores.mean():.4f}")
    print(f"p-value           : {p_value:.4f}")


    loo = LeaveOneOut()  # better than k-fold for n=50

    score, perm_scores, p_value = permutation_test_score(
        knn, X_test, y_test,
        cv=loo,
        n_permutations=1000,
        scoring='accuracy',
        random_state=42
    )

    print(f"LOO F1   : {score:.4f}")
    print(f"Null mean: {perm_scores.mean():.4f}")
    print(f"p-value  : {p_value:.4f}")
