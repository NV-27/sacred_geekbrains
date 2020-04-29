from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold


class CrossValidationSplitter(BaseEstimator, TransformerMixin):
    """
    Выбор стратегии кросс-валидации.

    Parameters
    ----------
    cv_strategy: str
        Стратегия кросс-валидация. Может принимать значение kfold, stratified_kfold, group_kfold.
        Иначе выбрасывается исключение типа ValueError.

    n_folds: int
        Число фолдов для кросс-валидации. Игнорируетеся, если cv_strategy = group_kfold.

    shuffle: boolean, optional, default = False
        Флаг перемешивания данных перед разбиением, опциональный параметр, по умолчанию = False.

    split_column: str, optional, default = None
        Название столбца для проведения group_kfold. Игнорируется, если cv_strategy != group_kfold.

    cv_seed: int, optional, default = None
        seed для проведения разбиения.

    """
    def __init__(self,
                 cv_strategy: str,
                 n_folds: int,
                 shuffle: bool = False,
                 split_column: Optional[str] = None,
                 cv_seed: Optional[int] = None):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.split_column = split_column
        self.cv_seed = cv_seed

        if cv_strategy not in ["kfold", "stratified_kfold", "group_kfold"]:
            msg = (
                f"Unknown Cross-Validation Strategy ({cv_strategy}). "
                "Please use: kfold, stratified_kfold or group_kfold strategy."
            )
            raise ValueError(msg)
        else:
            self.cv_strategy = cv_strategy

        if cv_strategy == "group_kfold" and split_column is None:
            msg = "The split_column parameter should not be None."
            raise ValueError(msg)

    def transform(self, X: pd.DataFrame, y: pd.Series):
        """

        """
        splitter = self._get_splitter
        return splitter(X, y)

    def _get_splitter(self, X: pd.DataFrame, y: pd.Series):
        """
        Выбор генератора кросс-валидации на основе выбранной стратегии.
        """
        if self.cv_strategy == "kfold":
            splitter = KFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.cv_seed
            )
            return splitter.split(X, y)
        elif self.cv_strategy == "stratified_kfold":
            splitter = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.cv_seed
            )
            return splitter.split(X, y)
        else:
            splitter = GroupKFold(
                n_splits=self.n_folds
            )
            return splitter.split(X, y, groups=X[self.split_column])
