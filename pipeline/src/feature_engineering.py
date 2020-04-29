import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FrequencyCounterTransformer(BaseEstimator, TransformerMixin):
    """
    Преобразование значений признака в частоту использования
    данного признака.

    Parameters
    ----------
    copy: bool, optional, default = True
        Флаг создания копии вектора признаков.
        Опциональный параметр, по умолчанию, равен True.
        Если принимает значени True, то создается глубокая
        копия y и используется для дальнейших вычислений;
        если принимает значение False, то для вычислений
        используется исходный y.

    Attributes
    ----------
    classes_: dict
        Словарь, где ключ - значение признака,
        значение - частота признака.

    """
    def __init__(self, copy: bool = True) -> None:
        self.copy = copy

    @staticmethod
    def check_array(array) -> pd.DataFrame:
        """
        Проверка входящего набора данных и преобразование
        входящего набора данных в pd.Series.

        """
        if isinstance(array, pd.Series):
            return array

        if len(array.shape) == 1:
            return pd.Series(array)

        else:
            msg = f"1D vector expected, y.shape = {array.shape}"
            raise ValueError(msg)

    def _copy(self, y):
        return y.copy() if self.copy else y

    def fit(self, y):
        """
        Обучение частотного энкодера

        Parameters
        ----------
        y: array-like, shape = [n_samples, ]
            Вектор со значениями признака.

        Returns
        -------
        self

        """
        y = self._copy(y)
        y = self.check_array(y)
        self.classes_ = y.value_counts().to_dict()

        return self

    def transform(self, y):
        """
        Применение частотного энкодера к новому набору данных.

        Parameters
        ----------
        y: array-like, shape = [n_samples, ]
            Вектор со значениями признака.

        Returns
        -------
        y_encoded: np.array, shape = [n_samples, ]
            Преобразованный вектор с частотами признака.

        """
        y = self._copy(y)
        check_is_fitted(self, "classes_")
        y_transformed = self.check_array(y)

        return y_transformed.map(self.classes_).values
