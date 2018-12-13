import os
import itertools
import numpy as np
from scipy.stats import zscore, ttest_1samp
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import feature_selection
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


def _standardize(train_x, test_x, scaling):
    """Scale training data and apply training scaling parameters to test set"""
    if scaling == 'voxel':
        # scale within voxels (each voxel gets standardized)
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
    elif scaling == 'pattern':
        # scale within pattern (each trial pattern gets standardized)
        train_x = zscore(train_x, axis=1)
        test_x = zscore(test_x, axis=1)

    return train_x, test_x


def _min_max(train_x, test_x, scaling, scale_min=-1, scale_max=1):

    if scaling == 'voxel':
        # scale within voxels (each voxel gets rescaled)
        scaler = MinMaxScaler(feature_range=(scale_min, scale_max))
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
    elif scaling == 'pattern':
        # scale within pattern (each trial pattern gets rescaled). Need to perform first
        # transpose so that we scale each pattern (not voxel), then transform it back to
        # original time x voxel array
        train_x = MinMaxScaler(feature_range=(scale_min, scale_max)).fit_transform(train_x.T).T
        test_x = MinMaxScaler(feature_range=(scale_min, scale_max)).fit_transform(test_x.T).T

    return train_x, test_x


def _scale(train_x, test_x, scaler, direction='voxels'):
    """Pass in a scikit-learn scaling object (e.g., StandardScaler) to scale
    training data and test data (separately, based on the training data)"""
    if direction == 'voxels':
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
    elif direction == 'pattern':
        # scale within pattern (each trial pattern gets standardized). Scaling is
        # different for each sample/pattern and therefore you cannot transform your
        # test set based on your training set
        train_x = zscore(train_x, axis=1)
        test_x = zscore(test_x, axis=1)
    return train_x, test_x


def _classify(classifier, train_x, train_y, test_x, test_y,
              scale_direction='voxels'):
    """Train and evaluate SVM classifier"""

    classifier.fit(train_x, train_y)
    yhat = classifier.predict(test_x)
    return accuracy_score(test_y, yhat), yhat, classifier


def _rfe(classifier, n_voxels, train_x, train_y, test_x):

    rfe_model = classifier
    selector = feature_selection.RFE(rfe_model, n_voxels)
    selector.fit(train_x, train_y)

    # narrow down training set
    train_x = train_x[:, selector.support_]
    # narrow down test set **based on features only determined in test set**
    test_x = test_x[:, selector.support_]

    return train_x, test_x, selector


def leave_one_run_out(df, classifier, run_column='run', data_column='voxels',
                      response_column='condition', scaler=None,
                      scale_direction='voxel', rfe_voxels=None, shuffle_data=False,
                      hyperparameters=None):
    """Perform leave one run out cross-validation"""

    if not np.issubdtype(df[response_column].dtype, np.number):
        # prevent weird bug in SVC function which does not always correctly
        # interpret string class labels
        raise TypeError('Data in response column must be of type int or float')

    grouped = df.groupby('run')

    accuracies = []
    y_list = []
    yhat_list = []
    model_list = []
    selector_list = []
    for name, g in grouped:

        # test data (data of current group, g)
        if shuffle_data:
            g = g.sample(frac=1)

        test_x = np.vstack(g[data_column]).astype(np.float)
        test_y = g[response_column].values

        # train data (data not in current group, g)
        train_subset = df[df[run_column] != name]
        if shuffle_data:
            train_subset = train_subset.sample(frac=1)
        train_x = np.vstack(train_subset[data_column]).astype(np.float)
        train_y = train_subset[response_column].values

        if scaler is not None:
            train_x, test_x = _scale(train_x, test_x, scaler, scale_direction)

        if hyperparameters is not None:
            nested_folds = len(grouped) - 1
            gs = GridSearchCV(classifier, hyperparameters, cv=nested_folds,
                              scoring='accuracy')
            gs.fit(train_x, train_y)
            # replace classifier with best model
            classifier = gs.best_estimator_


        # if RFE, identify and select top N features based on training data only
        if rfe_voxels is not None:
            train_x, test_x, selector = _rfe(classifier, rfe_voxels, train_x, train_y, test_x)
            selector_list.append(selector)

        accuracy, yhat, model = _classify(classifier, train_x, train_y, test_x, test_y)

        # store data require for metrics
        accuracies.append(accuracy)
        y_list.append(test_y)
        yhat_list.append(yhat)
        model_list.append(model)



    return accuracies, y_list, yhat_list, model_list, selector_list


def cross_decode(train_data, classifier, test_data, data_column='voxels',
                 response_column='condition', mean_centre=False,
                 shuffle_data=False, return_as_lists=False):
    if shuffle_data:
        train_data = train_data.sample(frac=1)
        test_data = test_data.sample(frac=1)

    train_voxels = np.vstack(train_data[data_column])
    test_voxels = np.vstack(test_data[data_column])

    if mean_centre:
        train_voxels = train_voxels - np.mean(train_voxels, axis=0)
        test_voxels = test_voxels - np.mean(test_voxels, axis=0)

    train_x = train_voxels
    train_y = train_data[response_column].values
    test_x = test_voxels
    test_y = test_data[response_column].values


    accuracy, yhat, model = _classify(classifier, train_x, train_y, test_x,
                                      test_y)

    if return_as_lists:
        # used for Decoder to keep API consistent
        return [accuracy], [test_y], [yhat], [model]
    else:
        return accuracy, test_y, yhat, model

class Decoder:

    def __init__(self, data, classifier=SVC(kernel='linear'), run_column='run',
                 data_column='voxels', response_column='condition', scaler=StandardScaler(),
                 scale_direction='voxel'):

        self.data = data
        self.classifier = classifier
        self.run_column = run_column
        self.data_column = data_column
        self.response_column = response_column
        self.scaler = scaler
        self.scale_direction = scale_direction

        self.n_classes = len(np.unique(self.data[self.response_column]))


    def cross_validate(self, shuffle_data=True, conditions=None, rfe_voxels=None,
                       hyperparameters=None):

        if conditions is not None:
            self.cv_data = self.data[self.data[self.response_column].isin(conditions)]
        else:
            self.cv_data = self.data

        self.accuracies, self.test_y, self.yhat, self.models, selectors = (
            leave_one_run_out(self.cv_data, self.classifier, run_column=self.run_column,
                              data_column=self.data_column,
                              response_column=self.response_column,
                              scaler=self.scaler,
                              scale_direction=self.scale_direction,
                              shuffle_data=shuffle_data,
                              rfe_voxels=rfe_voxels,
                              hyperparameters=hyperparameters)
        )

        self.mean_accuracy = np.mean(self.accuracies)

        if len(selectors) > 0:
            self.rfe_models = selectors


    def eval_test_set(self, test_data, mean_centre=False, shuffle_data=True,
                      conditions=None):

        if conditions is not None:
            self.train_data = self.data[self.data[self.response_column].isin(conditions)]
            self.test_data = test_data[test_data[self.response_column].isin(conditions)]
        else:
            self.train_data = self.data
            self.test_data = test_data

        self.accuracies, self.test_y, self.yhat, self.models = (
            cross_decode(self.train_data, self.classifier, self.test_data, self.data_column,
                         self.response_column, mean_centre,
                         shuffle_data, return_as_lists=True)
        )

        self.mean_accuracy = self.accuracies

    def confusion_matrix(self, mean=True, normalized=True):

        cm = confusion_matrix(np.concatenate(self.test_y), np.concatenate(self.yhat))

        if mean:
            n_runs = max(self.data[self.run_column])
            cm = cm / n_runs

        if normalized:
            cm = cm / np.sum(cm, axis=1)

        return cm