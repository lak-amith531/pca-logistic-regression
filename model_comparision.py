# %%

import pandas as pd
import pickle

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)

# %%

x_columns = [
    'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12',
    'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22',
    'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30'
]


def get_x_y(cancer_df):

    X = cancer_df[x_columns]
    y = cancer_df['g']

    # cancer_df.describe()
    return X, y


def train(path_to_csv):
    print("Loading Training data...")
    x, y = get_x_y(pd.read_csv(path_to_csv))
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x,
    #     y,
    #     test_size=0.2,
    #     random_state=0,
    # )

    # Using Gaussian Kernel
    svc = SVC(kernel='rbf', degree=3, probability=True)

    standard_scaler = StandardScaler()

    # Creating pipeline
    svc_pipe = Pipeline(steps=[
        ('standard_scaler', standard_scaler),
        ("svc", svc),
    ])

    # StandardScaler().fit_transform(X)[0]
    # param_grid = {
    #     'svc__kernel': ['linear'],
    #     'svc__degree': [3, 4, 5, 6],
    #     # 'svc__scale': ['scale', 'auto'],
    # }

    # search = GridSearchCV(
    #     pipe,
    #     param_grid,
    #     n_jobs=-1,
    #     return_train_score=True,
    # )

    svc_pipe.fit(x, y)

    # print("Best parameter (CV score=%0.3f):" % search.best_score_)
    # print(search.best_params_)

    with open("svm_classifier.pkl", 'wb') as fp:
        pickle.dump(svc_pipe, fp)


def predict(path_to_csv):
    x, y = get_x_y(pd.read_csv(path_to_csv))
    with open("svm_classifier.pkl", 'rb') as fp:
        svm_classifier = pickle.load(fp)

    y_pred = svm_classifier.predict(x)

    print(confusion_matrix(y, y_pred))
    print(accuracy_score(y, y_pred))


# %%


def main():

    # Comment this once trained
    train(path_to_csv="cancer_100k.csv")

    predict(path_to_csv="cancer_1M.csv")


# %%

if __name__ == "__main__":
    main()

# %%
