import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import csv
import time
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
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
        self.useClassWeight = useClassWeight
        self.plot = plot
        self.history = None
        self.model = None
        x, y = self.prepare_dataset()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, train_size=self.train_size,
                                                                                random_state=42,
                                                                                stratify=y)
        self.model = self.design_model()
        self.class_weight = {0: (1 / 357) * (569 / 2.0), 1:(1 / 212) * (569 / 2.0)}
        self.x_train, self.y_train = self.reshape_dataset(self.x_train, self.y_train)
        self.x_test, self.y_test = self.reshape_dataset(self.x_test, self.y_test)

    def set_sensitivity(self, value):
        self.sensitivity = value

    def set_specificity(self, value):
        self.specificity = value

    def train(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy',
            verbose=0,
            patience=15,
            mode='max',
            restore_best_weights=True)

        if self.useClassWeight:
            self.history = self.model.fit(self.x_train, self.y_train, verbose=0, epochs=self.number_of_epochs,
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
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if self.plot:
            plt.show()

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, acc, 'y', label='Training Acc')
        plt.plot(epochs, val_acc, 'r', label='Validation Acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        if self.plot:
            plt.show()

        test_predictions_baseline = self.model.predict(self.x_test, batch_size=self.batch_size)

        baseline_results = self.model.evaluate(self.x_test, self.y_test,
                                               batch_size=self.batch_size, verbose=0)
        """
        for name, value in zip(self.model.metrics_names, baseline_results):
            print(name, ': ', value)
        print()
        """

        self.plot_cm(self.y_test, test_predictions_baseline)

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

    def design_model(self):
        model = tf.keras.models.Sequential()

        # ReLU activation used in
        # model.add(
        # tf.keras.layers.Conv1D(filters=9, kernel_size=3, activation='relu', padding='same', input_shape=(6, 1)))
        # model.add(tf.keras.layers.Conv1D(filters=3, kernel_size=3, activation='relu', padding='same'))
        # model.add(tf.keras.layers.MaxPooling1D())

        model.add(tf.keras.layers.Dense(5, activation='relu', input_shape=(self.x_train.shape[1],)))
        model.add(tf.keras.layers.Dropout(0.33))
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(5, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.33))
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(5, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.33))
        # model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Flatten())

        # Dense layer of 1 as output is binary
        # Sigmoid activation function used to predict the probability between range of 1 and 0, suitable for binary output
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        # adam using adaptive learning rate
        # loss function computes difference between current and expected output of the algorithm
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def plot_cm(self, labels, predictions, p=0.5):
        cm = confusion_matrix(labels, predictions > p)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix @{:.2f}'.format(p))
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        if self.plot:
            plt.show()

        true_neg = cm[0][0]
        false_pos = cm[0][1]
        false_neg = cm[1][0]
        true_pos = cm[1][1]

        print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
        print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
        print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
        print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
        print('Total Fraudulent Transactions: ', np.sum(cm[1]))

        sensitivity = true_pos / (false_neg + true_pos)  # true positive rate
        specificity = true_neg / (true_neg + false_pos)  # true negative rate

        self.set_sensitivity(sensitivity)
        self.set_specificity(specificity)

        print("Sensitivity (true positive rate): ", sensitivity)
        print("Specificity (true negative rate): ", specificity)


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
    # fit all features to the same scale to improve performance of gradient-descent algorithms used in loss funcs
    X = scaler.fit_transform(X)

    y = LabelEncoder().fit_transform(y)
    X = pd.DataFrame(X, columns=X_final.columns)
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=numberOfFeaturesToSelect, step=1)
    rfe_selector.fit(X, y)
    return X.columns[rfe_selector.get_support()].values.tolist()


def test_batch_sizes():
    resultsFile = "results"
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


"""
for i in range(100):
    netTest = CancerDetectionNeuralNetwork(32, 100, 0.2, True, False)
    netTest.main()
    resultsWClassWeights = [netTest.sensitivity, netTest.specificity]
    resultsArrayWClassWeight.append(resultsWClassWeights)
    del netTest

for i in range(100):
    netTest = CancerDetectionNeuralNetwork(32, 100, 0.2, False, False)
    netTest.main()
    resultsWOClassWeights = [netTest.sensitivity, netTest.specificity]
    resultsArrayWOClassWeight.append(resultsWOClassWeights)
    del netTest

resultsFileClassWeight = "SensSpecWithClassWeight"
resultsFileNoClassWeight = "SensSpecNoClassWeight"

fields = ["Sensitivity", "Specificity"]

with open(resultsFileClassWeight, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(resultsArray)

with open(resultsFileNoClassWeight, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(resultsArrayWOClassWeight)

"""


def test_train_time():
    featuresToConsider = select_features(6)
    netTest = CancerDetectionNeuralNetwork(1, 500, 0.33, True, True, featuresToConsider, K_cross_validate=True)
    start_time = time.perf_counter()
    netTest.train()
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Time to train {total_time:0.4f} seconds")

    start_time = time.perf_counter()
    netTest.test_nn()
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Time to test {total_time:0.4f} seconds")


# test_train_time()


def compare_train_time_with_features_selected():
    results = []
    for i in range(1, 10): # go up to 32
        times = 0
        sens = 0
        spec = 0
        for j in range(1, 26):
            featuresToConsider = select_features(i)
            netTest = CancerDetectionNeuralNetwork(32, 500, 0.2, True, False, featuresToConsider, K_cross_validate=True)
            #start_time = time.perf_counter()
            netTest.train()
            netTest.test_nn()
            #end_time = time.perf_counter()
            #total_time = end_time - start_time
            #times += total_time
            sens += netTest.sensitivity
            spec += netTest.specificity
            del netTest
            print(j, i)
        avgSens = round(sens/25, 4)
        avgSpec = round(spec/25, 4)

        results.append([i, avgSens, avgSpec])
    resultsfile = "AvgAccForFeautres"
    fields = ["Number of features", "Average sens", "Average spec"]
    with open(resultsfile, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(results)


def perform_kcross_validation():
    features = select_features(7)
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

    skf = StratifiedKFold(n_splits= 5)
    StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
    results = [["Experiment", "Sensitivity", "Specificity"]]
    i = 1
    for train_index, test_index in skf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        netTest = CancerDetectionNeuralNetwork(8, 500, 0.2, True, False, features_to_consider=features)
        netTest.x_train, netTest.x_test = X_train, X_test
        netTest.y_train, netTest.y_test = y_train, y_test
        netTest.train()
        netTest.test_nn()
        results.append([i, round(netTest.sensitivity, 4), round(netTest.specificity, 4)])
        del netTest
        i += 1

    resultsfile = "experiment10,4"
    with open(resultsfile, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)

perform_kcross_validation()