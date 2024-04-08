import numpy as np
np.set_printoptions(threshold=10000,suppress=True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import time
warnings.filterwarnings('ignore')


def import_data():
    return pd.read_csv("credit_scoring.csv", sep=";")

def get_data_variables(data):

    # Transformation du DataFrame en un tableau numpy
    data_array = data.values

    # Séparation des caractéristiques (X) de la variable à prédire (y)
    X = data_array[:, :-1]
    y = data_array[:, -1]

    return X, y

def display_data_head():
    print("test")

def analyse_data_properties(data, y):
    # Analyser les propriétés des données
    taille_echantillon = data.shape
    pourcentage_positifs = (sum(y == 1) / len(y)) * 100
    pourcentage_negatifs = (sum(y == 0) / len(y)) * 100

    print("L'échantillon contient", taille_echantillon[0], " individus décrits par", taille_echantillon[1], "variables")
    print("Pourcentage de positifs : ", pourcentage_positifs)
    print("Pourcentage de négatifs : ", pourcentage_negatifs)
    plt.hist(y)
    print("Ainsi pour être bon, nos modèles doivent avoir une accuracy supérieur à", pourcentage_positifs)
    print("Les modèles seront donc plus performants que le hasard.")

def launch_clfs(clfs, X_train, y_train, X_test, y_test):
    clfs_measures = {
        'DT': {'acc': 0, 'precision': 0},
        'KNN': {'acc': 0, 'precision': 0},
        'MLP': {'acc': 0, 'precision': 0}
    }

    for i in clfs:
        clf=clfs[i]
        clf.fit(X_train, y_train)
        prediction=clf.predict(X_test)
        print(confusion_matrix(y_test, prediction))
        acc=accuracy_score(y_test, prediction)
        precision=precision_score(y_test, prediction)
        clfs_measures[i]['acc'] = acc
        clfs_measures[i]['precision'] = precision
        print('Pour {0} : Accuracy = {1:.2f} %, Precision = {2:.2f} %'.format(i, acc*100, precision*100))
        print()
    
    return clfs_measures

def normalize_data(X_train, X_test):
    SS=StandardScaler()
    SS.fit(X_train)
    X_train_norm=SS.transform(X_train)
    X_test_norm=SS.transform(X_test)

    return X_train_norm, X_test_norm

def normalize_data_one(X_train):
    SS=StandardScaler()
    SS.fit(X_train)
    X_train_norm=SS.transform(X_train)

    return X_train_norm

def apply_pca(X_train, X_test, n):
    
    pca=PCA(n_components=n)
    X_train_pca=pca.fit_transform(X_train)
    X_test_pca=pca.transform(X_test)

    return X_train_pca, X_test_pca

def get_concatenated_data(X_train_pca, X_train_norm, X_test_pca, X_test_norm):
    ## On concatène les nouvelles variables X_train_norm et X_test_norm
    X_train_concat = np.concatenate((X_train_pca, X_train_norm), axis=1)
    X_test_concat = np.concatenate((X_test_pca, X_test_norm), axis=1)

    return X_train_concat, X_test_concat

def plot_measures_methods_difference(clfs_measures, clfs_measures_norm, clfs_measures_norm_acp):
    labels_legend = ["data", "normalized_data", "normalized_data_pca"]
    
    # Création des listes pour chaque métrique
    accuracy = [[d[model]['acc'] for model in d] for d in [clfs_measures, clfs_measures_norm, clfs_measures_norm_acp]]
    precision = [[d[model]['precision'] for model in d] for d in [clfs_measures, clfs_measures_norm, clfs_measures_norm_acp]]

    labels = list(clfs_measures.keys())
    x = range(len(labels))

    # Création des graphiques
    fig, ax = plt.subplots(2, figsize=(10, 10))  # Augmentation de la taille des graphiques

    # Graphique pour l'accuracy
    for i in range(len(accuracy)):
        bars = ax[0].bar([p + 0.2*i for p in x], accuracy[i], width=0.2, label=labels_legend[i])
        for bar in bars:
            yval = bar.get_height()
            ax[0].text(bar.get_x() + bar.get_width()/2.0, yval, "{:.2f}%".format(yval * 100), va='bottom') # va: vertical alignment
    ax[0].set_xticks([p + 0.2 for p in x])
    ax[0].set_xticklabels(labels)
    ax[0].set_title('Accuracy')
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Graphique pour la precision
    for i in range(len(precision)):
        bars = ax[1].bar([p + 0.2*i for p in x], precision[i], width=0.2, label=labels_legend[i])
        for bar in bars:
            yval = bar.get_height()
            ax[1].text(bar.get_x() + bar.get_width()/2.0, yval, "{:.2f}%".format(yval * 100), va='bottom') # va: vertical alignment
    ax[1].set_xticks([p + 0.2 for p in x])
    ax[1].set_xticklabels(labels)
    ax[1].set_title('Precision')
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_variables_importance(X_train_norm, y_train, data):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_norm, y_train)
    importances=clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
    sorted_idx = np.argsort(importances)[::-1]
    features = data.columns
    print(features[sorted_idx])
    padding = np.arange(X_train_norm.size/len(X_train_norm)) + 0.5
    plt.barh(padding, importances[sorted_idx],xerr=std[sorted_idx], align='center')
    plt.yticks(padding, features[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    plt.show()
    return sorted_idx

def number_variables_to_keep(Xtrain, Xtest, Ytest, Ytrain, sorted_idx, clf):
    scores=np.zeros(Xtrain.shape[1]+1)
    for f in np.arange(0, Xtrain.shape[1]+1):
        X1_f = Xtrain[:,sorted_idx[:f+1]]
        X2_f = Xtest [:,sorted_idx[:f+1]]
        clf.fit(X1_f,Ytrain)
        Yclassifier=clf.predict(X2_f)
        scores[f]=np.round(accuracy_score(Ytest,Yclassifier),3)
    plt.plot(scores)
    plt.xlabel("Nombre de Variables")
    plt.ylabel("Accuracy")
    plt.title("Evolution de l'accuracy en fonction des variables")
    plt.show()

def find_best_variables(X_train, y_train, X_test, y_test, clf, params):
    # Définition de la fonction de score personnalisée
    def custom_score(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        return (accuracy + precision) / 2

    # Création d'un objet de score à partir de la fonction personnalisée
    custom_scorer = make_scorer(custom_score)

    # Création de l'objet GridSearchCV avec la fonction de score personnalisée
    grid_search = GridSearchCV(estimator=clf, param_grid=params, cv=3, scoring=custom_scorer, n_jobs=-1)

    # Exécution de la recherche sur la grille
    grid_search.fit(X_train, y_train)

    # Affichage des meilleurs paramètres trouvés
    print("Meilleurs paramètres trouvés:")
    print(grid_search.best_params_)

    # Évaluation du modèle avec les meilleurs paramètres sur l'ensemble de test
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    score = custom_score(y_test, y_pred)
    print("Accuracy du modèle sur l'ensemble de test:", accuracy)
    print("Precision du modèle sur l'ensemble de test:", precision)
    print("Score du modèle sur l'ensemble de test (accuracy + precision) / 2:", score)

    return grid_search.best_params_

def run_classifiers(clfs, X, y, sorted_idx={}, best_number_variables={}):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    print("{:<20} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}".format("Classifier", "AP", "Acc", "Precision", "Std Acc", "Std Precision", "AUC", "Std AUC", "Time"))
    print("-" * 130)

    results = {}
    for name, clf in clfs.items():
        start_time = time.time()

        X_best = X
        if (name in best_number_variables):
            X_best = X[:,sorted_idx[:best_number_variables[name]]]
        
        # Accuracy estimation
        cv_acc = cross_val_score(clf, X_best, y, cv=kf)
        mean_acc = np.mean(cv_acc)
        std_acc = np.std(cv_acc)
        
        # Precision estimation
        cv_precision = cross_val_score(clf, X_best, y, cv=kf, scoring='precision')
        mean_precision = np.mean(cv_precision)
        std_precision = np.std(cv_precision)
        
        # AUC estimation
        cv_auc = cross_val_score(clf, X_best, y, cv=kf, scoring='roc_auc')
        mean_auc = np.mean(cv_auc)
        std_auc = np.std(cv_auc)
        
        # Mean of (accuracy + precision) / 2
        mean_acc_prec = np.mean((cv_acc + cv_precision) / 2)
        
        # Execution time
        exec_time = time.time() - start_time

        results[name] = {
            'mean_acc_prec': mean_acc_prec,
            'mean_acc': mean_acc,
            'mean_precision': mean_precision,
            'std_acc': std_acc,
            'std_precision': std_precision,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'exec_time': exec_time
        }

    # classer les résultats par mean_acc_prec
    results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1]['mean_acc_prec'], reverse=True)}
    for name, result in results.items():
        mean_acc_prec = result['mean_acc_prec']
        mean_acc = result['mean_acc']
        mean_precision = result['mean_precision']
        std_acc = result['std_acc']
        std_precision = result['std_precision']
        mean_auc = result['mean_auc']
        std_auc = result['std_auc']
        exec_time = result['exec_time']
        print("{:<20} | {:<10.3f} | {:<10.3f} | {:<10.3f} | {:<10.3f} | {:<10.3f} | {:<10.3f} | {:<10.3f} | {:<10.3f}".format(name, mean_acc_prec, mean_acc, mean_precision, std_acc,std_precision, mean_auc, std_auc, exec_time))

def keep_only_numeric(X_):
    X_c = np.copy(X_)
    X_c[X_c == '?'] = np.nan
    indices_numeric = []
    for i in range(0, len(X_c[0])):
        try:
            float(X_c[0][i])
            indices_numeric.append(i)
        except:
            pass
    X_c = X_c[:, indices_numeric]
    return X_c.astype(float)

def analyse_data_properties_2(X, y):
    # Analyser les propriétés des données

    number_positifs = 0
    number_negatifs = 0

    for i in range(0, len(y)):
        if y[i] == '+':
            number_positifs += 1
        else:
            number_negatifs += 1
    
    taille_echantillon = len(X)
    number_variables = len(X[0])
    pourcentage_positifs = (number_positifs / len(y)) * 100
    pourcentage_negatifs = (number_negatifs / len(y)) * 100

    print("L'échantillon contient", taille_echantillon, " individus décrits par", number_variables, "variables")
    print("Pourcentage de positifs : ", pourcentage_positifs)
    print("Pourcentage de négatifs : ", pourcentage_negatifs)
    plt.hist(y)

def binarisation(y):
    for i in range(0, len(y)):
        if y[i] == '+':
            y[i] = 1
        else:
            y[i] = 0
    return y.astype(int)

def get_categories_numeric_columns_ids(X):
    categories_columns_ids = []
    numeric_columns_ids = []
    for value in range(len(X[0])):
        try :
            float(X[0][value])
            numeric_columns_ids.append(value)
            pass
        except:
            categories_columns_ids.append(value)
    return categories_columns_ids, numeric_columns_ids

def get_final_imputed_dataset(X):

    categories_columns_ids, numeric_columns_ids = get_categories_numeric_columns_ids(X)
    
    # Variables Catégorielles
    # On remplace les valeurs manquantes par la valeur la plus fréquente
    X_cat = np.copy(X[:, categories_columns_ids])
    for col_id in range(len(categories_columns_ids)):
        unique_val, val_idx = np.unique(X_cat[:, col_id], return_inverse=True)
        X_cat[:, col_id] = val_idx

    imp_cat = SimpleImputer(missing_values=0, strategy='most_frequent')
    X_cat[:, range(5)] = imp_cat.fit_transform(X_cat[:, range(5)])
    X_cat_bin = OneHotEncoder().fit_transform(X_cat).toarray()

    # Variables Numériques
    X_num = np.copy(X[:, numeric_columns_ids])
    X_num[X_num == '?'] = np.nan
    X_num = X_num.astype(float)
    imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_num = imp_num.fit_transform(X_num)

    return np.concatenate((X_cat_bin, X_num), axis=1)