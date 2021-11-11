import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder

tf.test.is_gpu_available


def prepare_dataset():
    cancer_data = pd.read_csv("BreastCancerData.csv", encoding='latin-1', engine='python')
    cancer_data["diagnosis"].replace({"M": 1, "B": 0}, inplace=True)

    X = cancer_data.drop(['id'], axis=1)
    X = X.drop(['diagnosis'], axis=1)
    # X = X[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean']]

    y = cancer_data['diagnosis']

    return X, y


def reshape_dataset(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(len(X), 30, 1)

    y = LabelEncoder().fit_transform(y)

    return X, y


def design_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(30, 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


X, y = prepare_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.33, random_state=42, stratify=y)
model = design_model()

X_train, y_train = reshape_dataset(X_train, y_train)
X_test, y_test = reshape_dataset(X_test, y_test)
history = model.fit(X_train, y_train, verbose=1, epochs=500, batch_size=32,
                    validation_data=(X_test, y_test))
"""
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
"""
