import numpy as np
import pandas as pd
from typing import Optional, List
from sacred import Ingredient

data_ingredient = Ingredient("data")
data_ingredient.add_config("configs/config.yaml")


@data_ingredient.capture
def get_input(filename: str,
              target_name: Optional[str] = None,
              drop_features: Optional[List[str]] = None,
              n_rows: Optional[int] = None,
              ):
    """
    Загрузка данных с диска.

    Parameters
    ----------
    filename: str
        Путь до загружаемого файла.

    target_name: str, optional, default = None
        Название признака с целевой переменной. Опциональный параметр.
        По умолчанию не используется. Если значение задано, то функция
        возвращает матрицу признаков и вектор целевой переменной, иначе -
        - только матрицу признаков.

    drop_features: List[str], optional, default = None
        Список признаков, которые должны быть удалены из набора данных.

    n_rows: int, optional, default = None
        Количество загружаемых строк. Опциональный параметр, по умолчанию не используется.

    Returns
    -------
    data or data, target:

    """
    data = pd.read_csv(filename, nrows=n_rows)

    if drop_features:
        data = data.drop(drop_features, axis=1)

    if target_name:
        target = data[target_name]
        data = data.drop(target_name, axis=1)
        n_features, obs, events, eventrate = data.shape[1], target.shape[0], target.sum(), target.mean()
        print(f"obs = {obs}, features = {n_features}, events = {events}, eventrate = {100 * np.round(eventrate, 4)}%")

        return data, target

    return data