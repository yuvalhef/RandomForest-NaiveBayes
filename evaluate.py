import RandomForestClassifierGaussianNB_leaves
import pandas as pd
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
import time
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support as score, precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import csv
from Utils import *
from sklearn.model_selection import train_test_split


def main():
    random_grid = get_random_grid()
    write_headline()
    total_scores={}
    for dataset in data_sets:
        ds = pd.read_csv(data_sets[dataset])
        print("Dataset name: {}".format(dataset))
        ds = ds.dropna()
        # le = LabelEncoder()
        X = ds.values[:, 0:-1]
        Y = ds.values[:, -1]
        # Y =le.fit_transform(Y)
        X, Y = shuffle(X, Y, random_state=10)
        kf = KFold(n_splits=min(5, max([int(ds.__len__()/200),2])))
        kf.get_n_splits(X)
        total_scores.setdefault(dataset, {})
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        for clf_name in classifiers:
            Scores = {
                'best': {'time': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
                'c1': {'time': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
                'c2': {'time': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
                'c3': {'time': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
                'c4': {'time': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
            }
            total_scores[dataset].setdefault(clf_name, Scores)
            total_scores = best(clf_name, random_grid, X_train, Y_train,X_test,Y_test,dataset,total_scores)

        for train_index, test_index in kf.split(X):
            for clf_name in classifiers:
                for configuration in configurations:
                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]
                    start_time = time.time()
                    if clf_name == 'RF':
                        clf=ensemble.RandomForestClassifier(**configurations_paramters[configuration])
                    else:
                        clf=RandomForestClassifierGaussianNB_leaves.RandomForestClassifierNB(**configurations_paramters[configuration])
                    clf.fit(X_train, Y_train)
                    pred = clf.predict(X_test)
                    seconds = (time.time() - start_time)
                    total_scores = evaluate_i_fold(Y_test, pred, seconds, total_scores, dataset, clf_name, configuration, clf.min_samples_leaf, clf.max_depth)
        write_data_set_results_to_csv(dataset, total_scores, classifiers, configurations)


def best(clf_name, random_grid, X_train, Y_train,X_test,Y_test,dataset,total_scores):
    if clf_name=='RF':
        clf = ensemble.RandomForestClassifier()
    else:
        clf = RandomForestClassifierGaussianNB_leaves.RandomForestClassifierNB()
    rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=10, cv=4, random_state=42, verbose=2) #n_jobs=-2)
    # Fit the random search model
    start_time = time.time()
    rf_random.fit(X_train, Y_train)
    clf = rf_random.best_estimator_
    pred = clf.predict(X_test)
    seconds = (time.time() - start_time)
    total_scores = evaluate_i_fold(Y_test, pred, seconds, total_scores, dataset, clf_name, 'best', clf.min_samples_leaf, clf.max_depth)
    return total_scores

def get_random_grid():
    # Hyperparameter for random grid search
    # Number of trees in random forest
    n_estimators = [50] #[int(x) for x in np.linspace(start=60, stop=100, num=20)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [2, 4, 6, 8, 10, 12, 14, 16, 20, 25]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    # min_samples_split = [10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [5, 10, 15, 20, 25, 30, 40, 50]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   # 'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    return random_grid

def evaluate_i_fold(Y_test, pred, time, total_scores_func, dataset, clf,configuration, n_samples, max_depth):
    accuracy = round(accuracy_score(Y_test, pred), 3)
    # precision, recall, f1, _ = score(Y_test, pred)
    precision = round(precision_score(Y_test, pred, average='macro'), 3)
    recall = round(recall_score(Y_test, pred, average='macro'), 3)
    f1 = round(f1_score(Y_test, pred, average='macro'), 3)
    total_scores_func[dataset][clf][configuration]['min_samples_leaf'] = n_samples
    total_scores_func[dataset][clf][configuration]['max_depth'] = max_depth
    total_scores_func[dataset][clf][configuration]['time'].append(round(float(time), 2))
    total_scores_func[dataset][clf][configuration]['accuracy'].append(accuracy)
    total_scores_func[dataset][clf][configuration]['precision'].append(precision)
    total_scores_func[dataset][clf][configuration]['recall'].append(recall)
    total_scores_func[dataset][clf][configuration]['f1'].append(f1)
    return total_scores_func


def write_data_set_results_to_csv(dataset,total_scores,classifiers,configurations_in_f):
    con=[]
    con.extend(configurations_in_f)
    con.append('best')
    for clf in classifiers:
        for configuration in con:
            log_list_test = [dataset,clf,configuration,
                             total_scores[dataset][clf][configuration]['min_samples_leaf'],
                             total_scores[dataset][clf][configuration]['max_depth'],
                             np.mean(total_scores[dataset][clf][configuration]['time']),
                             np.mean(total_scores[dataset][clf][configuration]['accuracy']),
                             np.mean(total_scores[dataset][clf][configuration]['precision']),
                             np.mean(total_scores[dataset][clf][configuration]['recall']),
                             np.mean(total_scores[dataset][clf][configuration]['f1']) ]
            writer = csv.writer(open("results.csv", "a"), lineterminator='\n', dialect='excel')
            writer.writerow(log_list_test)

def write_headline():
    log_list_header = ['dataset', 'classifier', 'configuration',  'min_samples_leaf', 'max_depth', 'time', 'accuracy', 'precision', 'recall', 'f1']
    writer = csv.writer(open("results.csv", "a"), lineterminator='\n', dialect='excel')
    writer.writerow(log_list_header)

main()




