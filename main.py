import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer


def prepare_dataset():
    cancer_data = pd.read_csv("BreastCancerData.csv", encoding='latin-1', engine='python')

    # plt.figure(figsize=(25, 25))
    # sns.heatmap(cancer_data.corr(), vmin=-1, vmax=1, annot=True)
    # plt.show()

    # replace diagnosis labels with binary labels
    cancer_data["diagnosis"].replace({"M": 1, "B": 0}, inplace=True)

    X = cancer_data.drop(['id'], axis=1)
    X = X.drop(['diagnosis'], axis=1)
    X = X[['radius_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'concave points_worst']]

    y = cancer_data['diagnosis']

    return X, y


def reshape_dataset(X, y):
    scaler = StandardScaler()
    # fit all features to the same scale to improve performance of gradient-descent algorithms used in loss funcs
    X = scaler.fit_transform(X)
    X = X.reshape(len(X), 6, 1)

    y = LabelEncoder().fit_transform(y)

    return X, y


def design_model(outputBias=None):
    if outputBias is not None:
        outputBias = tf.constant_initializer(outputBias)
    model = tf.keras.models.Sequential()

    # ReLU activation used in
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(6, 1)))
    # drops random nodes
    # helps prevent overwriting as cannot be dependent on any one node as it could be dropped during training
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(64, activation='relu'))

    # Dense layer of 1 as output is binary
    # Sigmoid activation function used to predict the probability between range of 1 and 0, suitable for binary output
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=outputBias))
    # adam using adaptive learning rate
    # loss function computes difference between current and expected output of the algorithm
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


X, y = prepare_dataset()  # X is data, y are the data's labels

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.33, random_state=42, stratify=y)
model = design_model()

X_train, y_train = reshape_dataset(X_train, y_train)
X_test, y_test = reshape_dataset(X_test, y_test)
history = model.fit(X_train, y_train, verbose=1, epochs=75, batch_size=16,
                    validation_data=(X_test, y_test))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, acc, 'y', label='Training Acc')
plt.plot(epochs, val_acc, 'r', label='Validation Acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
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

    print("Sensitivity (true positive rate): ", sensitivity)
    print("Specificity (true negative rate): ", specificity)


test_predictions_baseline = model.predict(X_test, batch_size=16)
baseline_results = model.evaluate(X_test, y_test,
                                  batch_size=16, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
    print(name, ': ', value)
print()

plot_cm(y_test, test_predictions_baseline)


def do_grid_search():
    batch_size = [6, 64]
    epochs = [10, 50]
    model = KerasRegressor(build_fn=history)
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        scoring=make_scorer(mean_squared_error, greater_is_better=False), return_train_score=True)
    grid_result = grid.fit(X_train, y_train, verbose=0)
    print(grid_result)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    print("Traininig")
    means = grid_result.cv_results_['mean_train_score']
    stds = grid_result.cv_results_['std_train_score']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# ------------- RANDOMIZED SEARCH --------------
def do_randomized_search():
    param_grid = {'batch_size': sp_randint(2, 16), 'nb_epoch': sp_randint(10, 100)}
    model = KerasRegressor(build_fn=history)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                              scoring=make_scorer(mean_squared_error, greater_is_better=False), n_iter=12)
    grid_result = grid.fit(X_train, y_train, verbose=0)
    print(grid_result)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


print("-------------- GRID SEARCH --------------------")
do_grid_search()
print("-------------- RANDOMIZED SEARCH --------------------")
do_randomized_search()
