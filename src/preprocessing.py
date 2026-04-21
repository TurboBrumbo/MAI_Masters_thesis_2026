"""
Модуль предварительной обработки и нормализации данных БПЛА.
Реализует очистку пропусков, Z-score стандартизацию, выделение числовых признаков
и подготовку данных для кластеризации, машинного обучения и MCDM.
Соответствует разделам 3.2 и 3.3 диссертации.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Класс для полного цикла предобработки датасета БПЛА.
    """

    def __init__(
        self,
        scaling_method: str = 'standard',  # 'standard', 'minmax', 'robust'
        impute_strategy: str = 'median',    # 'mean', 'median', 'constant'
        outlier_threshold: float = 3.5       # порог для обнаружения выбросов (Z-score)
    ):
        """
        Параметры:
        - scaling_method: метод масштабирования ('standard' – Z-score, 'minmax' – [0,1], 'robust' – устойчивый к выбросам)
        - impute_strategy: стратегия заполнения пропусков
        - outlier_threshold: количество стандартных отклонений для пометки выброса
        """
        self.scaling_method = scaling_method
        self.impute_strategy = impute_strategy
        self.outlier_threshold = outlier_threshold
        self.scaler = None
        self.feature_columns = None
        self.is_fitted = False
        self.outlier_mask = None
        logger.info(f"Препроцессор инициализирован: scaling={scaling_method}, impute={impute_strategy}")

    def _detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Обнаружение выбросов методом Z-score.
        Возвращает булеву маску, где True – выброс хотя бы по одному признаку.
        """
        z_scores = np.abs((df[columns] - df[columns].mean()) / df[columns].std())
        outlier_mask = (z_scores > self.outlier_threshold).any(axis=1)
        self.outlier_mask = outlier_mask
        logger.info(f"Обнаружено выбросов: {outlier_mask.sum()} из {len(df)}")
        return outlier_mask

    def identify_numerical_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Автоматически определяет числовые колонки, исключая идентификаторы и целевые переменные
        (если они не нужны для нормализации).
        """
        exclude_patterns = ['Model', 'Cluster', 'Actual', 'Efficiency_Score', 'Power_to_Weight']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        filtered = [col for col in numeric_cols if not any(pat in col for pat in exclude_patterns)]
        essential = ['Weight_g', 'Max_Speed_ms', 'Battery_Capacity_mAh',
                     'Propeller_Size_inch', 'Flight_Time_min', 'Range_km',
                     'Camera_MP', 'Price_USD']
        for col in essential:
            if col in numeric_cols and col not in filtered:
                filtered.append(col)
        logger.info(f"Числовые признаки для нормализации: {filtered}")
        return filtered

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Обучает scaler на данных и возвращает очищенный и нормализованный датафреймы.

        Возвращает:
        - df_clean: исходные данные с заполненными пропусками и, возможно, без выбросов
        - df_scaled: нормализованные числовые признаки + немасштабированные колонки (Model и др.)
        """
        if df.empty:
            raise ValueError("Входной DataFrame пуст.")

        df_clean = df.copy()

        self.feature_columns = self.identify_numerical_columns(df_clean)
        if not self.feature_columns:
            raise ValueError("Не найдено числовых колонок для обработки.")

        imputer = SimpleImputer(strategy=self.impute_strategy)
        df_clean[self.feature_columns] = imputer.fit_transform(df_clean[self.feature_columns])
        logger.info(f"Пропуски заполнены стратегией '{self.impute_strategy}'.")

        outlier_mask = self._detect_outliers(df_clean, self.feature_columns)
        df_clean['Is_Outlier'] = outlier_mask

        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Неизвестный метод масштабирования: {self.scaling_method}")

        scaled_array = self.scaler.fit_transform(df_clean[self.feature_columns])
        df_scaled = pd.DataFrame(scaled_array, columns=self.feature_columns)

        non_numeric_cols = [col for col in df_clean.columns if col not in self.feature_columns]
        for col in non_numeric_cols:
            if col in df_clean.columns:
                df_scaled[col] = df_clean[col].values

        self.is_fitted = True
        logger.info("Масштабирование завершено, скейлер обучен.")
        return df_clean, df_scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет уже обученный скейлер к новым данным.
        """
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError("Скейлер не обучен. Сначала вызовите fit_transform().")
        df_new = df.copy()
        imputer = SimpleImputer(strategy=self.impute_strategy)
        df_new[self.feature_columns] = imputer.fit_transform(df_new[self.feature_columns])
        scaled_array = self.scaler.transform(df_new[self.feature_columns])
        df_scaled = pd.DataFrame(scaled_array, columns=self.feature_columns)
        non_numeric = [col for col in df_new.columns if col not in self.feature_columns]
        for col in non_numeric:
            if col in df_new.columns:
                df_scaled[col] = df_new[col].values
        return df_scaled

    def get_feature_importance_hint(self) -> Dict[str, float]:
        """
        Возвращает дисперсию каждого признака после масштабирования
        (полезно для отладки кластеризации).
        """
        if not self.is_fitted:
            return {}
        var = self.scaler.var_ if hasattr(self.scaler, 'var_') else np.ones(len(self.feature_columns))
        return dict(zip(self.feature_columns, var))


class DataExplorer:
    """
    Вспомогательный класс для быстрого разведочного анализа данных (EDA).
    """
    @staticmethod
    def generate_report(df: pd.DataFrame) -> Dict[str, Any]:
        """Создаёт словарь с базовой статистикой."""
        report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': df.describe().to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict()
        }
        return report

    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, method: str = 'pearson'):
        """Возвращает матрицу корреляций (для последующей визуализации)."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None
        corr = numeric_df.corr(method=method)
        return corr


if __name__ == "__main__":
    from data_generator import AdvancedUAVDataGenerator
    gen = AdvancedUAVDataGenerator(random_seed=42)
    test_df = gen.generate_full_dataset(n_samples=50, include_seed_models=True)

    preprocessor = DataPreprocessor(scaling_method='standard', outlier_threshold=3.0)
    df_clean, df_scaled = preprocessor.fit_transform(test_df)

    print("Очищенные данные (первые 5 строк):")
    print(df_clean.head())
    print("\nНормализованные данные (первые 5 строк):")
    print(df_scaled.head())
    print("\nДисперсии признаков после масштабирования:")
    print(preprocessor.get_feature_importance_hint())

    explorer = DataExplorer()
    report = explorer.generate_report(test_df)
    print(f"\nРазмер датасета: {report['shape']}")
    print("Пропуски:", report['missing_values'])