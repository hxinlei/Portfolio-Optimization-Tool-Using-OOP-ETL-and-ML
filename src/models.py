from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

@dataclass
class MLResults:
    predictions: pd.DataFrame
    mae: float
    rmse: float

class Backtester:
    def __init__(self, model: RandomForestRegressor, train_size: int = 1000, step: int = 250):
        self.model = model
        self.train_size = train_size
        self.step = step

    def run(self, data: pd.DataFrame, predictors: list[str]) -> MLResults:
        all_preds = []
        n = data.shape[0]

        # Bail out early if not enough rows to do one train/test split
        if n < max(10, self.train_size + 1):
            return MLResults(predictions=pd.DataFrame(columns=["Return", "Predictions"]),
                             mae=float("nan"), rmse=float("nan"))

        for i in range(self.train_size, n, max(1, self.step)):
            train = data.iloc[i - self.train_size:i]
            test  = data.iloc[i:i + self.step]
            if len(test) == 0:
                break
            self.model.fit(train[predictors], train["Return"])
            preds = self.model.predict(test[predictors])
            frame = pd.concat(
                [test["Return"], pd.Series(preds, index=test.index, name="Predictions")],
                axis=1
            )
            all_preds.append(frame)

        if not all_preds:
            return MLResults(predictions=pd.DataFrame(columns=["Return", "Predictions"]),
                             mae=float("nan"), rmse=float("nan"))

        predictions = pd.concat(all_preds)
        try:
            mae = mean_absolute_error(predictions["Return"], predictions["Predictions"])
            rmse = np.sqrt(mean_squared_error(predictions["Return"], predictions["Predictions"]))
        except ValueError:
            mae, rmse = float("nan"), float("nan")

        return MLResults(predictions=predictions, mae=float(mae), rmse=float(rmse))