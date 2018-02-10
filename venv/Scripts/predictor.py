import numpy as np
from sklearn import tree
import pydotplus
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.decomposition import PCA
from fuzzy_decision_tree import FuzzyDecisionTree
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from fylearn.fpt import FuzzyPatternTreeTopDownClassifier
from cart_tree import CARTTree

class Predictor:
    def __init__(self, features, scores):
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.features_names = [value['features_names'] for key, value in features.items()][0]
        self.X = [value['features_values'] for key, value in features.items()]

        self.min_max_scaler.fit(self.X)
        self.X_scaled01 = self.min_max_scaler.transform(self.X)
        self.Y = [scores[key] for key, value in features.items()]

        self.pca_and_scale_X()
        self.pca_and_scale_X_scaled()

        self.X_transformed = self.X_scaled01[:, self.features_above_threshold]
        self.transformed_features_names = [self.features_names[index] for index in self.features_above_threshold]

    def pca_and_scale_X_scaled(self):
        # this one is just like the svm
        # the third input
        # this is the right one for knn
        # PCA 2
        FEATURE_TRESHHOLD = 4.0e-03
        pca = PCA()
        pca.fit(self.X_scaled01)

        self.scaled_features_above_threshold = np.where(pca.explained_variance_ > FEATURE_TRESHHOLD)[0]

        pca.n_components = len(np.where(pca.explained_variance_ > FEATURE_TRESHHOLD)[0])
        # print(pca.n_components)

        self.X_pca_transformed_scaled = pca.fit_transform(self.X_scaled01)
        self.X_pca_scaled_filtered = [self.filter_pca_features(row) for row in self.X]
        self.pca = pca

        # print(self.scaled_features_above_threshold)
        # print(len(self.X_pca_scaled_filtered[0]))

    def pca_and_scale_X(self):
        # the second and the last of the input
        # PCA
        FEATURE_TRESHHOLD = 0.5
        pca = PCA()
        pca.fit(self.X)

        self.features_above_threshold = np.where(pca.explained_variance_ > FEATURE_TRESHHOLD)[0]
        pca.n_components = len( np.where(pca.explained_variance_ > FEATURE_TRESHHOLD)[0])

        self.X_pca_transformed = pca.fit_transform(self.X)

        self.X_pca_filtered = [self.filter_pca_features(row) for row in self.X]

        # print(self.features_above_threshold)
        # print(len(self.X_pca_filtered[0]))

        # transoform in [0 1]
        self.pca_min_max_scaler = preprocessing.MinMaxScaler()
        self.pca_min_max_scaler.fit(self.X_pca_transformed)
        self.X_pca_transformed = self.pca_min_max_scaler.transform(self.X_pca_transformed)

    def scale_features(self, features):
        features = np.asarray(features).reshape(1,-1)
        return self.min_max_scaler.transform(features)

    def filter_pca_features(self, features):
        return [features[i] for i in self.features_above_threshold]

    def filter_scaled_pca_features(self, features):
        return [features[0][i] for i in self.scaled_features_above_threshold]

    def form_data(self):
        data = []
        for i in range(0, len(self.Y)):
            data.append(np.append(self.X_pca_transformed_scaled[i],self.Y[i]).tolist())
        return data

    def fuzzy_regression_tree(self):
        data = self.form_data()
        tree = FuzzyDecisionTree(data)
        return tree

    def fuzzy_pattern_tree(self):
        classifier = FuzzyPatternTreeTopDownClassifier()
        classifier.fit(self.X_pca_transformed_scaled,self.Y)
        return classifier

    def cart_tree(self):
        data = self.form_data()
        classifier = CARTTree(data)
        return classifier

    def form_result(seld, score1, score2):
        return (score1 + score2) / 2