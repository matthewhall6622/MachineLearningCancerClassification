"""
Project name: Machine learning cancer classification
Author name: Matthew Hall
Created: 15/10/21
"""

import math
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import csv
import time
from sklearn.feature_selection import RFE
import os;

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class CancerDetectionNeuralNetwork:

    def __init__(self, batch_size, number_of_epochs, train_size, useClassWeight, plot,
                 features_to_consider):
        self.features_to_consider = features_to_consider
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.train_size = train_size
        self.sensitivity = None
        self.specificity = None
        self.accuracy = None
        self.useClassWeight = useClassWeight
        self.plot = plot
        self.history = None
        self.model = None
        x, y = self.prepare_dataset()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, train_size=self.train_size,
                                                                                random_state=42,
                                                                                stratify=y)
        self.model = self.design_model()
        # calculation of class weights to deal with class imbalance of WDBC
        self.class_weight = {0: ((1 / 357) * (569 / 2.0)), 1: ((1 / 212) * (569 / 2.0))}
        self.x_train, self.y_train = self.reshape_dataset(self.x_train, self.y_train)
        self.x_test, self.y_test = self.reshape_dataset(self.x_test, self.y_test)

    def design_model(self):
        model = tf.keras.models.Sequential()

        # Size of the input layer varies on size of input
        model.add(tf.keras.layers.Input(self.x_train.shape[1], ))

        # Use of 2 hidden layers each with variable number of neurons depending on the size of the input
        model.add(tf.keras.layers.Dense((math.ceil(self.x_train.shape[1] / 2) + 1), activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2)) # randomly drops 20% of data exiting nodes to prevent overfitting
        model.add(tf.keras.layers.Dense((math.ceil(self.x_train.shape[1] / 2) + 1), activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())

        # Dense layer of 1 as output is binary Sigmoid activation function used to predict the probability between
        # range of 1 and 0, suitable for binary output
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        # adam using adaptive learning rate through calculation of first and second moments of gradient of loss
        # loss function computes difference between predicted and expected output of the algorithm
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy',
            verbose=0,
            patience=3,  # waits for accuracy to remain unchanged for 3 epochs then ends training
            mode='max',
            restore_best_weights=True)

        # option to test with and without class weights, switch in object definition
        if self.useClassWeight:
            self.history = self.model.fit(self.x_train, self.y_train, verbose=1, epochs=self.number_of_epochs,
                                          batch_size=self.batch_size,
                                          validation_data=(self.x_test, self.y_test), class_weight=self.class_weight,
                                          callbacks=early_stopping)
        else:
            self.history = self.model.fit(self.x_train, self.y_train, verbose=0, epochs=self.number_of_epochs,
                                          batch_size=self.batch_size,
                                          validation_data=(self.x_test, self.y_test), callbacks=early_stopping)

    def test_nn(self):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        # option to plot graphs each run, on/off in object definition
        if self.plot:
            plt.plot(epochs, loss, 'y', label='Training loss')
            plt.plot(epochs, val_loss, 'r', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        epochs = range(1, len(loss) + 1)
        if self.plot:
            plt.plot(epochs, acc, 'y', label='Training Acc')
            plt.plot(epochs, val_acc, 'r', label='Validation Acc')
            plt.title('Training and validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

        test_predictions = self.model.predict(self.x_test, batch_size=self.batch_size)

        # generates confusion matrix needed to generate sens, spec and acc
        cm = confusion_matrix(self.y_test, test_predictions > 0.5)
        if self.plot:
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title('Confusion matrix'.format(0.5))
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            plt.show()

        true_neg = cm[0][0]
        false_pos = cm[0][1]
        false_neg = cm[1][0]
        true_pos = cm[1][1]

        sensitivity = true_pos / (false_neg + true_pos)  # true positive rate
        specificity = true_neg / (true_neg + false_pos)  # true negative rate
        accuracy = (true_pos + true_neg) / (false_neg + false_pos + true_pos + true_neg) # accuracy

        self.sensitivity = sensitivity
        self.specificity = specificity
        self.accuracy = accuracy

    def prepare_dataset(self):
        cancer_data = pd.read_csv("BreastCancerData.csv", encoding='latin-1', engine='python')

        # replace diagnosis labels with binary labels
        cancer_data["diagnosis"].replace({"M": 1, "B": 0}, inplace=True)

        X = cancer_data.drop(['id'], axis=1)
        X = X.drop(['diagnosis'], axis=1)
        features = self.features_to_consider
        X = X[features]
        y = cancer_data['diagnosis']

        return X, y

    @staticmethod
    def reshape_dataset(X, y):
        scaler = StandardScaler()
        # fit all features to the same scale to improve performance of gradient-descent algorithms used in loss funcs
        X = scaler.fit_transform(X)

        y = LabelEncoder().fit_transform(y)

        return X, y

    def predict_case(self, patientData):
        scaler = StandardScaler()

        input = scaler.fit_transform(patientData)

        prediction = self.model.predict(input, batch_size=1)

        return prediction


def select_features(numberOfFeaturesToSelect):
    cancer_data = pd.read_csv("BreastCancerData.csv", encoding='latin-1', engine='python')

    X = cancer_data.drop(['id'], axis=1)
    X_final = X.drop(['diagnosis'], axis=1)

    # replace diagnosis labels with binary labels
    cancer_data["diagnosis"].replace({"M": 1, "B": 0}, inplace=True)

    X = cancer_data.drop(['id'], axis=1)
    X = X.drop(['diagnosis'], axis=1)

    y = cancer_data['diagnosis']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = LabelEncoder().fit_transform(y)
    X = pd.DataFrame(X, columns=X_final.columns)

    # Iteratively build logistic regression models using the dataset, rank, and remove the least important feature
    # each time
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=numberOfFeaturesToSelect, step=1)
    rfe_selector.fit(X, y)
    return X.columns[rfe_selector.get_support()].values.tolist()


# tests different batch sizes, no particular correlation with accuracy or train times
def test_batch_sizes_with_acc():
    resultsFile = "batchSizeAgainstAcc"
    resultsArray = []
    resultsArrayAvg = []
    bs = 1
    while bs <= 128:
        avgSens = 0
        avgSpec = 0
        for i in range(25):
            netTest = CancerDetectionNeuralNetwork(bs, 100, 0.8, True, False)
            netTest.train()
            netTest.test_nn()
            avgSens += netTest.sensitivity
            avgSpec += netTest.specificity
            del netTest
        avgSpec = avgSpec / 25
        avgSens = avgSens / 25
        resultsAvg = [bs, avgSens, avgSpec]
        resultsArrayAvg.append(resultsAvg)
        bs = bs * 2
        fields = ['Batch size', 'Sensitivity', 'Specificity']
    with open(resultsFile, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(resultsArrayAvg)

def test_batch_sizes_with_time():
    resultsFile = "batchSizeAgainstTrainTime"
    resultsArray = []
    bs = 1
    while bs <= 128:
        time = []
        for i in range(25):
            netTest = CancerDetectionNeuralNetwork(bs, 40, 0.8, True, False)
            start = time.perf_counter()
            netTest.train()
            end = time.perf_counter()
            totalTime = start - end
            time.append(totalTime)
            del netTest
        avgTime = np.mean(time)
        stdTime = np.std(time)
        results = [bs, avgTime, stdTime]
        resultsArray.append(results)
        bs = bs * 2
        fields = ['Batch size', 'Mean time', 'Std. time']
    with open(resultsFile, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(resultsArray)


def compare_train_time_with_features_selected():
    results = []
    for i in range(1, 31):  # go up to 30
        times = []
        featuresToConsider = select_features(i)
        for j in range(1, 11):
            netTest = CancerDetectionNeuralNetwork(16, 500, 0.8, True, False, featuresToConsider)
            start_time = time.perf_counter()
            netTest.train()
            end_time = time.perf_counter()
            total_time = end_time - start_time
            times.append(total_time)
            del netTest
            print(i, j)

        meanTime = np.mean(times)
        stdDevTime = np.std(times)
        results.append([i, round(meanTime, 4), round(stdDevTime, 4)])

    resultsfile = "trainTimes"
    fields = ["Number of features", "Mean time", "Std. Dev. time"]
    with open(resultsfile, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(results)


# compare_train_time_with_features_selected()

def perform_kcross_validation(paritions):
    features = select_features(9)
    cancer_data = pd.read_csv("BreastCancerData.csv", encoding='latin-1', engine='python')
    X = cancer_data.drop(['id'], axis=1)
    X = X.drop(['diagnosis'], axis=1)
    X = X[features]
    y = cancer_data['diagnosis']
    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    # fit all features to the same scale to improve performance of gradient-descent algorithms used in loss funcs
    X = scaler.fit_transform(X)

    y = LabelEncoder().fit_transform(y)
    skf = StratifiedKFold(n_splits=paritions, shuffle=False)
    results = [["Experiment", "Sensitivity", "Specificity", "Accuracy"]]
    i = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        netTest = CancerDetectionNeuralNetwork(16, 1000, 0.8, False, False, features_to_consider=features)
        netTest.x_train, netTest.x_test = X_train, X_test
        netTest.y_train, netTest.y_test = y_train, y_test
        netTest.train()
        netTest.test_nn()
        results.append([i, round(netTest.sensitivity, 4), round(netTest.specificity, 4), round(netTest.accuracy, 4)])
        del netTest
        i += 1

    resultsfile = "KCrossValidation"
    with open(resultsfile, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)

perform_kcross_validation(5)

def get_baseline_results():
    accuracy = []
    sensitivity = []
    specificity = []
    for i in range(1, 5):
        features = select_features(9)
        netTest = CancerDetectionNeuralNetwork(16, 40, 0.8, True, True, features)
        netTest.train()
        netTest.test_nn()
        accuracy.append(round(netTest.accuracy, 4))
        sensitivity.append(round(netTest.sensitivity, 4))
        specificity.append(round(netTest.specificity, 4))
        print(i)
        del netTest

    print("Average accuracy:", np.mean(accuracy))
    print("S.D accuracy", np.std(accuracy))
    print("Average sensitivity:", np.mean(sensitivity))
    print("S.D sensitivity", np.std(sensitivity))
    print("Average specificity:", np.mean(specificity))
    print("S.D specificity", np.std(specificity))

# get_baseline_results()

def mean_sens_spec_for_increasing_feature_numbers():
    results = [["Num of features", "Avg Sens", "Std. Sens", "Avg spec", "Std. spec"]]
    for i in range(1,31):
        sens_for_feat_num = []
        spec_for_feat_num = []
        features = select_features(i)
        for j in range(1,11):
            netTest = CancerDetectionNeuralNetwork(16,40,0.8, True, False, features)
            netTest.train()
            netTest.test_nn()
            sens_for_feat_num.append(netTest.sensitivity)
            spec_for_feat_num.append(netTest.specificity)
        avgSens = np.mean(sens_for_feat_num)
        stdSens = np.std(sens_for_feat_num)

        avgSpec = np.mean(spec_for_feat_num)
        stdSpec = np.std(spec_for_feat_num)

        results.append([i, round(avgSens, 4), round(stdSens, 4), round(avgSpec, 4), round(stdSpec, 4)])

    resultsFile = "MeanAndStdOfAccForIncreasingFeatureNum"
    with open(resultsFile, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)

mean_sens_spec_for_increasing_feature_numbers()