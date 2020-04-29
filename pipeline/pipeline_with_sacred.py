import time
import yaml
from tqdm import tqdm

from sacred import Experiment
from sacred.observers import FileStorageObserver
from ingredients.data import data_ingredient, get_input
from src.validation import CrossValidationSplitter
from src.feature_engineering import FrequencyCounterTransformer
from src.models import CatBoostClassifierCV

ex = Experiment(name="kaggle-santander-pipeline", ingredients=[data_ingredient])
ex.observers.append(FileStorageObserver("runs"))
ex.add_config("configs/config.yaml")
ex.add_config(cv_seed=27)


@ex.config
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

@ex.capture
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

@ex.automain
def main(_run, _config):
    print(_config)
    print(f"{time.ctime()}, pipeline start")
    X, y = get_input()
    X = create_features(X)

    cv = get_cv()
    model = CatBoostClassifierCV(cv, params=_config["model_params"], used_features=None)
    model.fit(X, y)

    _run.log_scalar("CV-Score (ROC-AUC): ", model.cv_score)
    _run.log_scalar("n_iterations: ", model.best_iteration)
    _run.log_scalar("Score on each fold: ", model.evals_result_)
    _run.log_scalar("n_iterations on each fold: ", model.best_iterations_)
