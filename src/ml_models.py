"""
Модуль машинного обучения для прогнозирования реального времени полёта БПЛА.
Реализует обучение и сравнение моделей: Random Forest, XGBoost.
Включает подбор гиперпараметров (GridSearchCV), оценку качества (RMSE, MAE, R²),
сохранение/загрузку моделей и расчёт важности признаков.
Соответствует разделу 2.3 диссертации.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from typing import Optional, Dict, Any, Tuple, List

logger = logging.getLogger(__name__)


class UAVFlightTimePredictor:
    """
    Класс для обучения и применения регрессионных моделей предсказания Actual_Flight_Time_min.
    """

    def __init__(
        self,
        model_type: str = 'random_forest',
        random_state: int = 42,
        verbose: int = 0
    ):
        """
        Параметры:
        - model_type: 'random_forest' или 'xgboost'
        - random_state: фиксирует случайность
        - verbose: уровень вывода (0 - тихо, 1 - подробно)
        """
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.feature_names_ = None
        self.is_fitted = False
        self.metrics_ = {}
        self.cv_results_ = None

        self._init_model()

    def _init_model(self):
        """Создаёт базовую модель с параметрами по умолчанию."""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                objective='reg:squarederror',
                verbosity=self.verbose
            )
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.model_type}")
        logger.info(f"Модель {self.model_type} инициализирована.")

    def set_hyperparameters(self, params: Dict[str, Any]):
        """Устанавливает гиперпараметры модели перед обучением."""
        if self.model is None:
            self._init_model()
        self.model.set_params(**params)
        logger.info(f"Гиперпараметры обновлены: {params}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Обучает модель на тренировочных данных и, если переданы валидационные,
        вычисляет метрики.
        """
        self.feature_names_ = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("Модель обучена.")

        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
            self.metrics_ = metrics
            return metrics
        return {}

    def train_with_grid_search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict[str, List],
        cv: int = 5,
        scoring: str = 'neg_root_mean_squared_error',
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Выполняет GridSearchCV для подбора гиперпараметров.
        Возвращает лучшие параметры и результаты кросс-валидации.
        """
        if self.model is None:
            self._init_model()

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=self.verbose,
            return_train_score=True
        )
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        self.is_fitted = True
        self.feature_names_ = X_train.columns.tolist()
        self.cv_results_ = grid_search.cv_results_

        logger.info(f"Лучшие параметры: {grid_search.best_params_}")
        logger.info(f"Лучший CV score ({scoring}): {grid_search.best_score_:.4f}")

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Предсказание на новых данных."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Модель не обучена.")
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        """Рассчитывает основные метрики регрессии."""
        y_pred = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan

        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        logger.info(f"Оценка: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}, MAPE={mape:.1f}%")
        return metrics

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'neg_root_mean_squared_error'
    ) -> Dict[str, np.ndarray]:
        """Кросс-валидация модели."""
        if self.model is None:
            self._init_model()
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        logger.info(f"CV scores ({scoring}): mean={scores.mean():.4f}, std={scores.std():.4f}")
        return {'scores': scores, 'mean': scores.mean(), 'std': scores.std()}

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Возвращает важность признаков из обученной модели.
        Для Random Forest и XGBoost доступно.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Модель не обучена.")
        if self.feature_names_ is None:
            raise ValueError("Имена признаков не сохранены.")

        if self.model_type == 'random_forest':
            importances = self.model.feature_importances_
        elif self.model_type == 'xgboost':
            importances = self.model.feature_importances_
        else:
            raise ValueError("Важность признаков доступна только для RF и XGBoost.")

        df_imp = pd.DataFrame({
            'Feature': self.feature_names_,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        return df_imp

    def plot_feature_importance(self, top_n: int = 10, figsize=(10, 6)):
        """Строит столбчатую диаграмму важности признаков."""
        df_imp = self.get_feature_importance()
        top_features = df_imp.head(top_n)
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis', ax=ax)
        ax.set_title(f'Важность признаков ({self.model_type})')
        ax.set_xlabel('Важность')
        ax.set_ylabel('Признак')
        plt.tight_layout()
        return fig

    def plot_predictions_vs_actual(self, y_true: pd.Series, y_pred: np.ndarray):
        """Диаграмма рассеяния предсказанных и фактических значений."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеал')
        ax.set_xlabel('Фактическое время (мин)')
        ax.set_ylabel('Предсказанное время (мин)')
        ax.set_title('Предсказание vs Факт')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def save_model(self, filepath: str):
        """Сериализует модель в файл."""
        if not self.is_fitted:
            logger.warning("Сохранение необученной модели.")
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names_,
            'metrics': self.metrics_,
            'random_state': self.random_state
        }, filepath)
        logger.info(f"Модель сохранена в {filepath}")

    def load_model(self, filepath: str):
        """Загружает модель из файла."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.model_type = data['model_type']
        self.feature_names_ = data['feature_names']
        self.metrics_ = data.get('metrics', {})
        self.random_state = data.get('random_state', 42)
        self.is_fitted = True
        logger.info(f"Модель {self.model_type} загружена из {filepath}")


class ModelComparator:
    """
    Сравнение нескольких моделей (RF, XGBoost) на одном наборе данных.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = {}

    def compare(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        models_to_compare: List[str] = ['random_forest', 'xgboost'],
        custom_params: Optional[Dict[str, Dict]] = None
    ) -> pd.DataFrame:
        """
        Обучает и оценивает указанные модели, возвращает сводную таблицу метрик.
        """
        summary = []
        for model_type in models_to_compare:
            logger.info(f"Обучение модели: {model_type}")
            predictor = UAVFlightTimePredictor(model_type=model_type, random_state=self.random_state)

            if custom_params and model_type in custom_params:
                predictor.set_hyperparameters(custom_params[model_type])

            predictor.train(X_train, y_train)
            metrics = predictor.evaluate(X_test, y_test)
            metrics['Model'] = model_type
            summary.append(metrics)
            self.results[model_type] = predictor

        df_summary = pd.DataFrame(summary).set_index('Model')
        return df_summary

    def get_best_model(self, metric: str = 'R2', higher_is_better: bool = True) -> Tuple[str, UAVFlightTimePredictor]:
        """Возвращает лучшую модель по заданной метрике."""
        if not self.results:
            raise RuntimeError("Сравнение не проводилось.")
        best_model = None
        best_value = -np.inf if higher_is_better else np.inf
        for name, pred in self.results.items():
            value = pred.metrics_.get(metric, np.nan)
            if higher_is_better:
                if value > best_value:
                    best_value = value
                    best_model = (name, pred)
            else:
                if value < best_value:
                    best_value = value
                    best_model = (name, pred)
        return best_model


if __name__ == "__main__":
    from data_generator import AdvancedUAVDataGenerator
    from preprocessing import DataPreprocessor

    gen = AdvancedUAVDataGenerator(random_seed=42)
    df = gen.generate_full_dataset(n_samples=500, include_seed_models=True)

    prep = DataPreprocessor(scaling_method='standard')
    df_clean, _ = prep.fit_transform(df)

    feature_cols = ['Weight_g', 'Max_Speed_ms', 'Battery_Capacity_mAh',
                    'Propeller_Size_inch', 'Flight_Time_min', 'Range_km',
                    'Camera_MP', 'Price_USD']
    target = 'Actual_Flight_Time_min'

    X = df_clean[feature_cols]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    comparator = ModelComparator(random_state=42)
    summary = comparator.compare(X_train, y_train, X_test, y_test)
    print("Сводка метрик:")
    print(summary)

    best_name, best_predictor = comparator.get_best_model(metric='R2')
    print(f"\nЛучшая модель по R2: {best_name}")

    fig_imp = best_predictor.plot_feature_importance()
    plt.show()

    y_pred = best_predictor.predict(X_test)
    fig_pred = best_predictor.plot_predictions_vs_actual(y_test, y_pred)
    plt.show()

    best_predictor.save_model("best_uav_model.joblib")