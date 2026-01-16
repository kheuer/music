from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
)
from sklearn.ensemble import RandomForestClassifier
from hyperparameter_tuning import *
from features import *
from data import *
from models import *


def svm_tuning(X_train, y_train):

    svm_pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])

    svm_param_grid = {
        "svc__kernel": ["rbf", "linear"],
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", 0.01, 0.1],  # ignored for linear
        "svc__class_weight": [None, "balanced"],
    }

    grid_search = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=svm_param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation accuracy: ", grid_search.best_score_)

    return grid_search.best_estimator_


def random_forest_tuning(X_train, y_train):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_param_dist = {
        "n_estimators": [200, 500, 800],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"],
    }
    rf_search = RandomizedSearchCV(
        rf,
        rf_param_dist,
        n_iter=50,
        cv=5,
        scoring="f1_macro",
        random_state=42,
        n_jobs=-1,
        verbose=2,
    )

    rf_search.fit(X_train, y_train)

    print("Best RF params:", rf_search.best_params_)
    print("Best CV score:", rf_search.best_score_)


if __name__ == "__main__":
    # load X, y, ids
    (
        (X_train, y_train, ids_train),
        (X_test, y_test, ids_test),
        (X_val, y_val, ids_val),
    ) = get_train_test_val_features_and_targets(
        df=df,
        feature_extractor=compute_spectral_features,
        splits=10,
        train_size=0.6,
        test_size=0.2,
    )
    # normalize data
    X_train, X_test, X_val = norm(
        X_train, X_test, X_val, feature_extractor=compute_spectral_features
    )
    # reshape data to fit svm and RandomForest
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2])
    # use only a subset for faster tuning
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=10000, random_state=42  # subset size
    )

    subset_idx, _ = next(sss.split(X_train, y_train))
    # perform hyperparameter tuning
    best_svm_config = svm_tuning(X_train[subset_idx], y_train[subset_idx])
    best_rf_config = random_forest_tuning(X_train[subset_idx], y_train[subset_idx])
