import argparse
import numpy as np
import pandas as pd
import os

from util import evaluate_D, evaluate_onehot, load, load_one
from util import evaluate_D_smooth, evaluate_onehot_smooth
from util import generate_data_name


def five(data, X, y, weight=False, tfidf=False, norm='l1', metric='l1'):
    """
    Evaluation for five-fold datasets

    Parameters
    ----------
    data : dict
        Dataset dictionary compatible with the original code
        data['TR'] is the indices of the training samples
        data['TE'] is the indices of the test samples
        The indices are 1-indexed

    X : numpy.array
        BOW vectors
        Shape: (n, d), where n is the number of documents, d is the size of the vocabulary
        X[i, j] is the number of occurences of word j in document i

    y : numpy.array
        Labels
        Shape: (n,), where n is the number of documents
        y[i] is the label of document i

    weight : bool
        wkNN (True) or standard kNN (False)

    tfidf : bool
        TF-IDF (True) or BOW (False)

    norm : {None, 'l1', 'l2'}
        Norm to normalize vectors
        If norm is None, vectors are not normalized.
        Otherwise, this argument is passed to `norm` argument of `sklearn.preprocessing.normalize`.

    metric : {'l1', 'l2'}
        Norm to compare vectors
        This argument is passes to `metric` argument of `sklearn.metrics.pairwise_distances`.

    Returns
    -------
    accs : numpy.array
        accs.shape is (5,)
        Each element represents an accuracy for each fold.
    """

    accs = []
    for i in range(5):
        if data['TR'].shape[0] == 1:
            train = data['TR'][0, i][0] - 1
            test = data['TE'][0, i][0] - 1
        else:
            train = data['TR'][i] - 1
            test = data['TE'][i] - 1
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        if weight:
            accs.append(evaluate_onehot_smooth(X_train, y_train, X_test, y_test, tfidf=tfidf))
        else:
            accs.append(evaluate_onehot(X_train, y_train, X_test, y_test, tfidf=tfidf, norm=norm, metric=metric))
    return np.array(accs)


def fiveD(data, y, D, weight=False):
    """
    Evaluation for five-fold datasets using a distance matrix

    Parameters
    ----------
    data : dict
        Dataset dictionary compatible with the original code
        data['TR'] is the indices of the training samples
        data['TE'] is the indices of the test samples
        The indices are 1-indexed

    y : numpy.array
        Labels
        Shape: (n,), where n is the number of documents
        y[i] is the label of document i

    D : numpy.array
        Distance matrix
        Shape: (n, n), where n is the number of documents
        D[i, j] is the distance between documents i and j

    weight : bool
        wkNN (True) or standard kNN (False)

    Returns
    -------
    accs : numpy.array
        accs.shape is (5,)
        Each element represents an accuracy for each fold.
    """

    accs = []
    for i in range(5):
        if data['TR'].shape[0] == 1:
            train = data['TR'][0, i][0] - 1
            test = data['TE'][0, i][0] - 1
        else:
            train = data['TR'][i] - 1
            test = data['TE'][i] - 1
        y_train = y[train]
        y_test = y[test]
        if weight:
            accs.append(evaluate_D_smooth(y_train, y_test, D[test][:, train], D[train][:, train]))
        else:
            accs.append(evaluate_D(y_train, y_test, D[test][:, train], D[train][:, train]))
    return np.array(accs)


def evaluate_five(filename):
    print(filename)
    print('-' * len(filename))
    is_rpw = "rpw" in filename
    data_name = filename.replace("_rpw", "")
    data, X, y = load('data/{}'.format(data_name))
    D = np.load('distance/{}.npy'.format(filename))
    D_tfidf = np.load('distance/{}-tfidf.npy'.format(filename))

    try:
        results = pd.read_csv('results/{}.csv'.format("results_five"))
    except FileNotFoundError:
        results = pd.DataFrame(columns=['data_name', 'method', 'norm', 'metric', 'accuracy', 'std', 'weight', 'tfidf'])

    if not is_rpw:
        for norm in ['l1', 'l2', None]:
            for metric in ['l1', 'l2']:
                res = (1 - five(data, X, y, norm=norm, metric=metric)) * 100
                print('BOW ({}/{})\t{:.1f} ± {:.1f}'.format(str(norm).upper(), str(metric).upper(), res.mean(), res.std()))
                results = results._append({'data_name': data_name, 'method': 'BOW', 'norm': norm, 'metric': metric, 'accuracy': res.mean(), 'std': res.std(), 'weight': False, 'tfidf': False}, ignore_index=True)
                res = (1 - five(data, X, y, tfidf=True, norm=norm, metric=metric)) * 100
                print('TF-IDF ({}/{})\t{:.1f} ± {:.1f}'.format(str(norm).upper(), str(metric).upper(), res.mean(), res.std()))
                results = results._append({'data_name': data_name, 'method': 'TF-IDF', 'norm': norm, 'metric': metric, 'accuracy': res.mean(), 'std': res.std(), 'weight': False, 'tfidf': True}, ignore_index=True)
        res = (1 - five(data, X, y, weight=True)) * 100
        print('BOW weight\t{:.1f} ± {:.1f}'.format(res.mean(), res.std()))
        results = results._append({'data_name': data_name, 'method': 'BOW', 'norm': None, 'metric': None, 'accuracy': res.mean(), 'std': res.std(), 'weight': True, 'tfidf': False}, ignore_index=True)
        res = (1 - five(data, X, y, weight=True, tfidf=True)) * 100
        print('TF-IDF weight\t{:.1f} ± {:.1f}'.format(res.mean(), res.std()))
        results = results._append({'data_name': data_name, 'method': 'TF-IDF', 'norm': None, 'metric': None, 'accuracy': res.mean(), 'std': res.std(), 'weight': True, 'tfidf': True}, ignore_index=True)

    method = 'WMD' if not is_rpw else 'RPW'
    res = (1 - fiveD(data, y, D)) * 100
    print('{}\t{:.1f} ± {:.1f}'.format(method, res.mean(), res.std()))
    results = results._append({'data_name': data_name, 'method': method, 'norm': None, 'metric': None, 'accuracy': res.mean(), 'std': res.std(), 'weight': False, 'tfidf': False}, ignore_index=True)

    res = (1 - fiveD(data, y, D_tfidf)) * 100
    print('{}-TF-IDF\t{:.1f} ± {:.1f}'.format(method, res.mean(), res.std()))
    results = results._append({'data_name': data_name, 'method': method, 'norm': None, 'metric': None, 'accuracy': res.mean(), 'std': res.std(), 'weight': False, 'tfidf': True}, ignore_index=True)

    res = (1 - fiveD(data, y, D, weight=True)) * 100
    print('{} weight\t{:.1f} ± {:.1f}'.format(method, res.mean(), res.std()))
    results = results._append({'data_name': data_name, 'method': method, 'norm': None, 'metric': None, 'accuracy': res.mean(), 'std': res.std(), 'weight': True, 'tfidf': False}, ignore_index=True)

    res = (1 - fiveD(data, y, D_tfidf, weight=True)) * 100
    print('{}-TF-IDF weight\t{:.1f} ± {:.1f}'.format(method, res.mean(), res.std()))
    results = results._append({'data_name': data_name, 'method': method, 'norm': None, 'metric': None, 'accuracy': res.mean(), 'std': res.std(), 'weight': True, 'tfidf': True}, ignore_index=True)

    if not os.path.exists('results'):
        os.makedirs('results')
    results.to_csv('results/{}.csv'.format("results_five"), index=False)
    print()


def evaluate_one(filename):
    print(filename)
    print('-' * len(filename))
    is_rpw = "rpw" in filename
    X_train, y_train, X_test, y_test = load_one('data/{}'.format(filename.replace("_rpw", "")))
    D = np.load('distance/{}.npy'.format(filename))
    D_train = np.load('distance/{}-train.npy'.format(filename))
    D_tfidf = np.load('distance/{}-tfidf.npy'.format(filename))
    D_train_tfidf = np.load('distance/{}-train-tfidf.npy'.format(filename))

    try:
        results = pd.read_csv('results/{}.csv'.format("results_one"))
    except FileNotFoundError:
        results = pd.DataFrame(columns=['data_name', 'method', 'norm', 'metric', 'accuracy', 'std', 'weight', 'tfidf'])

    if not is_rpw:
        for norm in ['l1', 'l2', None]:
            for metric in ['l1', 'l2']:
                res = (1 - evaluate_onehot(X_train, y_train, X_test, y_test, norm=norm, metric=metric)) * 100
                print('BOW ({}/{})\t{:.1f}'.format(str(norm).upper(), str(metric).upper(), res))
                results = results._append({'data_name': filename, 'method': 'BOW', 'norm': norm, 'metric': metric, 'accuracy': res, 'std': 0, 'weight': False, 'tfidf': False}, ignore_index=True)
                res = (1 - evaluate_onehot(X_train, y_train, X_test, y_test, tfidf=True, norm=norm, metric=metric)) * 100
                print('TF-IDF ({}/{})\t{:.1f}'.format(str(norm).upper(), str(metric).upper(), res))
                results = results._append({'data_name': filename, 'method': 'TF-IDF', 'norm': norm, 'metric': metric, 'accuracy': res, 'std': 0, 'weight': False, 'tfidf': True}, ignore_index=True)
        res = (1 - evaluate_onehot_smooth(X_train, y_train, X_test, y_test)) * 100
        print('BOW weight\t{:.1f}'.format(res))
        results = results._append({'data_name': filename, 'method': 'BOW', 'norm': None, 'metric': None, 'accuracy': res, 'std': 0, 'weight': True, 'tfidf': False}, ignore_index=True)
        res = (1 - evaluate_onehot_smooth(X_train, y_train, X_test, y_test, tfidf=True)) * 100
        print('TF-IDF weight\t{:.1f}'.format(res))
        results = results._append({'data_name': filename, 'method': 'TF-IDF', 'norm': None, 'metric': None, 'accuracy': res, 'std': 0, 'weight': True, 'tfidf': True}, ignore_index=True)

    res = (1 - evaluate_D(y_train, y_test, D, D_train)) * 100
    method = 'WMD' if not is_rpw else 'RPW'
    print('{}\t{:.1f}'.format(method, res))
    results = results._append({'data_name': filename, 'method': method, 'norm': None, 'metric': None, 'accuracy': res, 'std': 0, 'weight': False, 'tfidf': False}, ignore_index=True)

    res = (1 - evaluate_D(y_train, y_test, D_tfidf, D_train_tfidf)) * 100
    print('{}-TF-IDF\t{:.1f}'.format(method, res))
    results = results._append({'data_name': filename, 'method': method, 'norm': None, 'metric': None, 'accuracy': res, 'std': 0, 'weight': False, 'tfidf': True}, ignore_index=True)

    res = (1 - evaluate_D_smooth(y_train, y_test, D, D_train)) * 100
    print('{} weight\t{:.1f}'.format(method, res))
    results = results._append({'data_name': filename, 'method': method, 'norm': None, 'metric': None, 'accuracy': res, 'std': 0, 'weight': True, 'tfidf': False}, ignore_index=True)

    res = (1 - evaluate_D_smooth(y_train, y_test, D_tfidf, D_train_tfidf)) * 100
    print('{}-TF-IDF weight\t{:.1f}'.format(method, res))
    results = results._append({'data_name': filename, 'method': method, 'norm': None, 'metric': None, 'accuracy': res, 'std': 0, 'weight': True, 'tfidf': True}, ignore_index=True)

    if not os.path.exists('results'):
        os.makedirs('results')
    results.to_csv('results/{}.csv'.format("results_one"), index=False)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='recipe2')
    parser.add_argument('--reduced', type=bool, default=False)
    parser.add_argument('--clean', type=bool, default=False)
    parser.add_argument('--rpw', type=bool, default=False)
    args = parser.parse_args()

    if args.clean:
        filename = '{}_clean{}.mat'.format(args.filename, '_reduced' if args.reduced else '')
    else:
        filename = generate_data_name(args.filename, args.clean, args.reduced, args.rpw)

    evaluate_five_dataset_names = {"bbcsport", "twitter", "recipe2", "recipe", "classic", "amazon"}
    evaluate_one_dataset_names_maps = {"ohsumed": "ohsumed-emd_tr_te_ix", "reuter": "r8-emd_tr_te3", "20news": "20ng2_500-emd_tr_te"}

    if args.filename in evaluate_five_dataset_names:
        evaluate_five(filename)
    elif args.filename in evaluate_one_dataset_names_maps.keys():
        if args.clean:
            evaluate_one(filename)
        else:
            filename = evaluate_one_dataset_names_maps[args.filename]
            '{}{}{}.mat'.format(filename, '_reduced' if args.reduced else '', '_rpw' if args.rpw else '')
            evaluate_one(filename)

    # if args.clean:
    #     evaluate_five('bbcsport_clean_reduced.mat')
    #     evaluate_five('bbcsport_clean.mat')
    #     evaluate_five('twitter_clean.mat')
    #     evaluate_five('twitter_clean_reduced.mat')
    #     evaluate_five('recipe_clean.mat')
    #     evaluate_five('recipe_clean_reduced.mat')
    #     evaluate_one('ohsumed_clean.mat')
    #     evaluate_five('classic_clean.mat')
    #     evaluate_one('reuter_clean.mat')
    #     evaluate_five('amazon_clean.mat')
    #     evaluate_one('20news_clean.mat')
    # else:
    #     evaluate_five('bbcsport-emd_tr_te_split_reduced.mat')
    #     evaluate_five('bbcsport-emd_tr_te_split.mat')
    #     evaluate_five('twitter-emd_tr_te_split.mat')
    #     evaluate_five('twitter-emd_tr_te_split_reduced.mat')
    #     evaluate_five('recipe2-emd_tr_te_split.mat')
    #     evaluate_five('recipe2-emd_tr_te_split_reduced.mat')
    #     evaluate_one('ohsumed-emd_tr_te_ix.mat')
    #     evaluate_five('classic-emd_tr_te_split.mat')
    #     evaluate_one('r8-emd_tr_te3.mat')
    #     evaluate_five('amazon-emd_tr_te_split.mat')
    #     evaluate_one('20ng2_500-emd_tr_te.mat')
