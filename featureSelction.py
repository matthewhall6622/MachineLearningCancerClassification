from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.options.mode.chained_assignment = None  # default='warn'


def prepare_dataset():
    cancer_data = pd.read_csv("BreastCancerData.csv", encoding='latin-1', engine='python')

    # plt.figure(figsize=(25, 25))
    # sns.heatmap(cancer_data.corr(), vmin=-1, vmax=1, annot=True)
    # plt.show()

    # replace diagnosis labels with binary labels
    cancer_data["diagnosis"].replace({"M": 1, "B": 0}, inplace=True)

    X = cancer_data.drop(['id'], axis=1)
    X = X.drop(['diagnosis'], axis=1)

    y = cancer_data['diagnosis']

    return X, y


def reshape_dataset(X, y):
    scaler = StandardScaler()
    # fit all features to the same scale to improve performance of gradient-descent algorithms used in loss funcs
    X = scaler.fit_transform(X)

    y = LabelEncoder().fit_transform(y)
    return X, y


cancer_data = pd.read_csv("BreastCancerData.csv", encoding='latin-1', engine='python')

X = cancer_data.drop(['id'], axis=1)
X_final = X.drop(['diagnosis'], axis=1)
y = cancer_data['diagnosis']
X,y = reshape_dataset(X_final,y)

X = pd.DataFrame(X, columns=X_final.columns)

rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=6, step=1)
rfe_selector.fit(X, y)
print(X.columns[rfe_selector.get_support()])
