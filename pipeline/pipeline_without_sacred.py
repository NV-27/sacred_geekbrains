import time
import yaml
from tqdm import tqdm
import pandas as pd

from src.validation import CrossValidationSplitter
from src.feature_engineering import FrequencyCounterTransformer
from src.models import CatBoostClassifierCV

def get_model_params(model_name):
    if model_name == "lightgbm":
        config_name = "configs/lgb_config.yaml"
    elif model_name == "xgboost":
        config_name = "configs/xgb_config.yaml"
    elif model_name == "catboost":
        config_name = "configs/catboost_config.yaml"
    else:
        raise ValueError("Pipeline only for GBDT-models.")

    with open(config_name) as cfg:
        model_params = yaml.load(cfg)

    return model_params


def get_cv(cv_strategy, n_folds, cv_seed):
    cv = CrossValidationSplitter(
        cv_strategy, n_folds, cv_seed=cv_seed
    )
    return cv


def create_features(X):
    for feature in tqdm(X.columns):
        encoder = FrequencyCounterTransformer()
        X[f"{feature}_freq"] = encoder.fit_transform(X[feature])

    return X

def main():
    print(f"{time.ctime()}, pipeline start")
    with open("configs/config.yaml") as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
        config["cv_seed"] = 27

    X = pd.read_csv(config["filename"], nrows=config["n_rows"])
    y = X[config["target_name"]]

    X = X.drop(config["drop_features"], axis=1)
    X = X.drop(config["target_name"], axis=1)

    X = create_features(X)
    cv = get_cv(**{key:config[key] for key in ["cv_strategy", "n_folds", "cv_seed"]})
    model_params = get_model_params(config["model_name"])

    model = CatBoostClassifierCV(cv, params=model_params, used_features=None)
    model.fit(X, y)


if __name__ == "__main__":
    main()