"""
Модуль многокритериального принятия решений (MCDM) для ранжирования БПЛА.
Реализует методы: AHP (анализ иерархий), TOPSIS, VIKOR.
Позволяет вычислять веса критериев на основе матриц парных сравнений,
проверять согласованность, выполнять ранжирование альтернатив.
Интегрируется с предсказаниями ML для корректировки времени полёта.
Соответствует разделу 2.2 диссертации.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AHP:
    """
    Метод анализа иерархий (Analytic Hierarchy Process).
    Вычисляет веса критериев из матрицы парных сравнений с проверкой согласованности.
    """

    RI_TABLE = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
        11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59
    }

    def __init__(self, pairwise_matrix: Optional[np.ndarray] = None):
        """
        Параметры:
        - pairwise_matrix: квадратная матрица парных сравнений (n x n).
        """
        self.pairwise_matrix = pairwise_matrix
        self.n = pairwise_matrix.shape[0] if pairwise_matrix is not None else 0
        self.weights: Optional[np.ndarray] = None
        self.lambda_max: Optional[float] = None
        self.consistency_index: Optional[float] = None
        self.consistency_ratio: Optional[float] = None
        self.is_consistent: bool = False

    def set_pairwise_matrix(self, matrix: np.ndarray):
        """Устанавливает матрицу парных сравнений."""
        self.pairwise_matrix = matrix
        self.n = matrix.shape[0]
        self._validate_matrix()

    def _validate_matrix(self):
        """Проверяет, что матрица квадратная и положительная."""
        if self.pairwise_matrix is None:
            raise ValueError("Матрица не задана.")
        if self.pairwise_matrix.shape[0] != self.pairwise_matrix.shape[1]:
            raise ValueError("Матрица должна быть квадратной.")
        if not np.all(self.pairwise_matrix > 0):
            raise ValueError("Все элементы матрицы должны быть положительными.")
        for i in range(self.n):
            for j in range(self.n):
                if not np.isclose(self.pairwise_matrix[i, j], 1.0 / self.pairwise_matrix[j, i], rtol=1e-3):
                    logger.warning(f"Элемент ({i},{j}) нарушает обратную симметричность.")

    def compute_weights(self, method: str = 'eigenvector') -> np.ndarray:
        """
        Вычисляет вектор весов критериев.
        method: 'eigenvector' (собственный вектор) или 'geometric' (среднее геометрическое).
        """
        if self.pairwise_matrix is None:
            raise ValueError("Матрица парных сравнений не задана.")

        if method == 'eigenvector':
            eigvals, eigvecs = np.linalg.eig(self.pairwise_matrix)
            max_idx = np.argmax(eigvals.real)
            self.lambda_max = eigvals[max_idx].real
            weights = np.abs(eigvecs[:, max_idx].real)
            weights = weights / weights.sum()
        elif method == 'geometric':
            geom_means = np.exp(np.mean(np.log(self.pairwise_matrix), axis=1))
            weights = geom_means / geom_means.sum()
            weighted_sum = self.pairwise_matrix @ weights
            self.lambda_max = np.mean(weighted_sum / weights)
        else:
            raise ValueError(f"Неизвестный метод: {method}")

        self.weights = weights
        self._calculate_consistency()
        return self.weights

    def _calculate_consistency(self):
        """Вычисляет индекс согласованности (CI) и отношение согласованности (CR)."""
        if self.lambda_max is None or self.n == 0:
            return
        self.consistency_index = (self.lambda_max - self.n) / (self.n - 1) if self.n > 1 else 0.0
        ri = self.RI_TABLE.get(self.n, 1.49)
        self.consistency_ratio = self.consistency_index / ri if ri != 0 else 0.0
        self.is_consistent = self.consistency_ratio <= 0.10
        logger.info(f"λ_max = {self.lambda_max:.4f}, CI = {self.consistency_index:.4f}, CR = {self.consistency_ratio:.4f}")
        if self.is_consistent:
            logger.info("Матрица согласована (CR ≤ 0.10).")
        else:
            logger.warning("Матрица несогласована! Рекомендуется пересмотреть оценки.")

    @staticmethod
    def create_pairwise_matrix_from_vector(importance_vector: np.ndarray) -> np.ndarray:
        """
        Создаёт матрицу парных сравнений из вектора относительной важности.
        a_ij = w_i / w_j.
        """
        n = len(importance_vector)
        matrix = np.outer(importance_vector, 1.0 / importance_vector)
        return matrix

    @staticmethod
    def random_index(n: int) -> float:
        """Возвращает случайный индекс для заданной размерности."""
        return AHP.RI_TABLE.get(n, 1.49)

    def plot_weights(self, criteria_names: Optional[List[str]] = None, figsize=(8, 5)):
        """Строит столбчатую диаграмму весов критериев."""
        if self.weights is None:
            raise RuntimeError("Сначала вычислите веса.")
        if criteria_names is None:
            criteria_names = [f"C{i+1}" for i in range(self.n)]
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x=self.weights, y=criteria_names, palette='viridis', ax=ax)
        ax.set_xlabel('Вес')
        ax.set_title('Веса критериев (AHP)')
        ax.set_xlim(0, 1)
        for i, w in enumerate(self.weights):
            ax.text(w + 0.01, i, f"{w:.3f}", va='center')
        plt.tight_layout()
        return fig


class TOPSIS:
    """
    Метод TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).
    Ранжирует альтернативы на основе расстояний до идеального и антиидеального решений.
    """

    def __init__(self):
        self.decision_matrix: Optional[np.ndarray] = None
        self.criteria_types: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.scores_: Optional[np.ndarray] = None

    def fit(self, matrix: np.ndarray, weights: np.ndarray, criteria_types: np.ndarray):
        """
        Задаёт матрицу решений, веса критериев и типы критериев.
        matrix: альтернативы (строки) x критерии (столбцы)
        weights: веса критериев (сумма = 1)
        criteria_types: 1 для максимизации (benefit), -1 для минимизации (cost)
        """
        self.decision_matrix = matrix.astype(float)
        self.weights = weights / weights.sum()
        self.criteria_types = criteria_types

    def compute_scores(self) -> np.ndarray:
        """Вычисляет коэффициенты близости (scores) для каждой альтернативы."""
        if self.decision_matrix is None or self.weights is None or self.criteria_types is None:
            raise RuntimeError("Сначала вызовите fit().")

        norm_matrix = self.decision_matrix / np.sqrt((self.decision_matrix ** 2).sum(axis=0))

        weighted_matrix = norm_matrix * self.weights

        ideal_best = np.zeros(self.decision_matrix.shape[1])
        ideal_worst = np.zeros(self.decision_matrix.shape[1])

        for j, ctype in enumerate(self.criteria_types):
            if ctype == 1:
                ideal_best[j] = weighted_matrix[:, j].max()
                ideal_worst[j] = weighted_matrix[:, j].min()
            else:
                ideal_best[j] = weighted_matrix[:, j].min()
                ideal_worst[j] = weighted_matrix[:, j].max()

        dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

        with np.errstate(divide='ignore', invalid='ignore'):
            scores = dist_worst / (dist_best + dist_worst)
        scores[np.isnan(scores)] = 0.0

        self.scores_ = scores
        return scores

    def rank(self) -> np.ndarray:
        """Возвращает ранги альтернатив (1 - лучший)."""
        if self.scores_ is None:
            self.compute_scores()
        ranks = self.scores_.argsort()[::-1].argsort() + 1
        return ranks


class VIKOR:
    """
    Метод VIKOR (VIseKriterijumska Optimizacija I Kompromisno Resenje).
    Предназначен для ранжирования и выбора компромиссного решения.
    """

    def __init__(self, v: float = 0.5):
        """
        v: вес стратегии "большинства" (0.5 - консенсус, >0.5 - большинство, <0.5 - вето).
        """
        self.v = v
        self.decision_matrix: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.criteria_types: Optional[np.ndarray] = None
        self.S_: Optional[np.ndarray] = None
        self.R_: Optional[np.ndarray] = None
        self.Q_: Optional[np.ndarray] = None

    def fit(self, matrix: np.ndarray, weights: np.ndarray, criteria_types: np.ndarray):
        self.decision_matrix = matrix.astype(float)
        self.weights = weights / weights.sum()
        self.criteria_types = criteria_types

    def compute(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Возвращает S, R, Q значения для альтернатив.
        """
        if self.decision_matrix is None:
            raise RuntimeError("Сначала вызовите fit().")

        norm_matrix = np.zeros_like(self.decision_matrix)
        for j in range(self.decision_matrix.shape[1]):
            col = self.decision_matrix[:, j]
            f_best = col.max() if self.criteria_types[j] == 1 else col.min()
            f_worst = col.min() if self.criteria_types[j] == 1 else col.max()
            denom = f_best - f_worst
            if denom == 0:
                norm_matrix[:, j] = 0
            else:
                norm_matrix[:, j] = (f_best - col) / denom if self.criteria_types[j] == 1 else (col - f_best) / denom

        weighted_norm = norm_matrix * self.weights
        self.S_ = weighted_norm.sum(axis=1)
        self.R_ = weighted_norm.max(axis=1)

        S_star, S_minus = self.S_.min(), self.S_.max()
        R_star, R_minus = self.R_.min(), self.R_.max()
        denom_S = S_minus - S_star
        denom_R = R_minus - R_star

        if denom_S == 0:
            q1 = np.zeros_like(self.S_)
        else:
            q1 = self.v * (self.S_ - S_star) / denom_S

        if denom_R == 0:
            q2 = np.zeros_like(self.R_)
        else:
            q2 = (1 - self.v) * (self.R_ - R_star) / denom_R

        self.Q_ = q1 + q2
        return self.S_, self.R_, self.Q_

    def rank(self) -> np.ndarray:
        """Возвращает ранги (1 - лучший) на основе Q."""
        if self.Q_ is None:
            self.compute()
        ranks = self.Q_.argsort().argsort() + 1
        return ranks


class MCDMEngine:
    """
    Единый интерфейс для многокритериального ранжирования БПЛА.
    Интегрирует AHP, TOPSIS, VIKOR и использует предсказания ML для корректировки времени полёта.
    """

    def __init__(self):
        self.ahp = AHP()
        self.topsis = TOPSIS()
        self.vikor = VIKOR()
        self.criteria_names: List[str] = []
        self.alternative_names: List[str] = []

    def set_criteria_names(self, names: List[str]):
        self.criteria_names = names

    def set_alternative_names(self, names: List[str]):
        self.alternative_names = names

    def run_ahp(self, pairwise_matrix: np.ndarray) -> np.ndarray:
        """Запускает AHP и возвращает веса."""
        self.ahp.set_pairwise_matrix(pairwise_matrix)
        weights = self.ahp.compute_weights()
        return weights

    def run_topsis(
        self,
        decision_matrix: np.ndarray,
        weights: np.ndarray,
        criteria_types: np.ndarray,
        return_scores: bool = True
    ) -> pd.DataFrame:
        """
        Выполняет TOPSIS и возвращает DataFrame с результатами.
        """
        self.topsis.fit(decision_matrix, weights, criteria_types)
        scores = self.topsis.compute_scores()
        ranks = self.topsis.rank()

        result_df = pd.DataFrame({
            'Alternative': self.alternative_names if self.alternative_names else [f"A{i+1}" for i in range(len(scores))],
            'TOPSIS_Score': scores,
            'TOPSIS_Rank': ranks
        }).sort_values('TOPSIS_Rank')
        return result_df

    def run_vikor(
        self,
        decision_matrix: np.ndarray,
        weights: np.ndarray,
        criteria_types: np.ndarray,
        v: float = 0.5
    ) -> pd.DataFrame:
        """
        Выполняет VIKOR и возвращает DataFrame.
        """
        self.vikor.v = v
        self.vikor.fit(decision_matrix, weights, criteria_types)
        S, R, Q = self.vikor.compute()
        ranks = self.vikor.rank()

        result_df = pd.DataFrame({
            'Alternative': self.alternative_names if self.alternative_names else [f"A{i+1}" for i in range(len(Q))],
            'S': S,
            'R': R,
            'Q': Q,
            'VIKOR_Rank': ranks
        }).sort_values('VIKOR_Rank')
        return result_df

    def prepare_decision_matrix_from_df(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        predicted_time_col: Optional[str] = None,
        original_time_col: str = 'Flight_Time_min'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Из DataFrame формирует матрицу решений.
        Если predicted_time_col задан, заменяет им оригинальное время.
        criteria_types предполагается стандартным: масса (-1), скорость (1), время (1),
        дальность (1), камера (1), цена (-1). Можно настраивать.
        """
        df_copy = df.copy()
        if predicted_time_col and predicted_time_col in df_copy.columns:
            df_copy[original_time_col] = df_copy[predicted_time_col]

        matrix = df_copy[feature_columns].values

        default_types = {
            'Weight_g': -1,
            'Max_Speed_ms': 1,
            'Flight_Time_min': 1,
            'Range_km': 1,
            'Camera_MP': 1,
            'Price_USD': -1
        }
        types = np.array([default_types.get(col, 1) for col in feature_columns])
        return matrix, types

    def full_pipeline(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        pairwise_matrix: np.ndarray,
        criteria_types: Optional[np.ndarray] = None,
        predicted_time_col: Optional[str] = None,
        method: str = 'topsis'
    ) -> pd.DataFrame:
        """
        Полный конвейер: AHP -> формирование матрицы -> TOPSIS/VIKOR.
        Возвращает исходный DataFrame с добавленными колонками рейтинга.
        """
        weights = self.run_ahp(pairwise_matrix)

        if criteria_types is None:
            matrix, crit_types = self.prepare_decision_matrix_from_df(
                df, feature_columns, predicted_time_col
            )
        else:
            matrix = df[feature_columns].values
            crit_types = criteria_types

        if method.lower() == 'topsis':
            result = self.run_topsis(matrix, weights, crit_types)
            df_out = df.copy()
            df_out['TOPSIS_Score'] = result['TOPSIS_Score'].values
            df_out['TOPSIS_Rank'] = result['TOPSIS_Rank'].values
        elif method.lower() == 'vikor':
            result = self.run_vikor(matrix, weights, crit_types)
            df_out = df.copy()
            df_out['VIKOR_Q'] = result['Q'].values
            df_out['VIKOR_Rank'] = result['VIKOR_Rank'].values
        else:
            raise ValueError(f"Неизвестный метод: {method}")

        return df_out.sort_values(by=df_out.columns[-1])


if __name__ == "__main__":
    from data_generator import AdvancedUAVDataGenerator
    from preprocessing import DataPreprocessor
    from ml_models import UAVFlightTimePredictor
    from sklearn.model_selection import train_test_split

    gen = AdvancedUAVDataGenerator(random_seed=42)
    df = gen.generate_full_dataset(n_samples=100, include_seed_models=True)

    prep = DataPreprocessor(scaling_method='standard')
    df_clean, _ = prep.fit_transform(df)

    feature_cols = ['Weight_g', 'Max_Speed_ms', 'Battery_Capacity_mAh',
                    'Propeller_Size_inch', 'Flight_Time_min', 'Range_km',
                    'Camera_MP', 'Price_USD']
    target = 'Actual_Flight_Time_min'

    X = df_clean[feature_cols]
    y = df_clean[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    predictor = UAVFlightTimePredictor(model_type='random_forest')
    predictor.train(X_train, y_train)
    df_clean['Predicted_Flight_Time'] = predictor.predict(X)

    n_crit = len(feature_cols)
    ahp_matrix = np.ones((n_crit, n_crit))
    ahp_matrix[4, :] = [1/5, 1/3, 1/2, 1/2, 1, 1/3, 1/4, 1/2]
    for i in range(n_crit):
        for j in range(n_crit):
            if i != j:
                ahp_matrix[i, j] = 1.0 / ahp_matrix[j, i] if ahp_matrix[j, i] != 0 else 1

    engine = MCDMEngine()
    engine.set_alternative_names(df_clean['Model'].tolist())
    engine.set_criteria_names(feature_cols)

    result_df = engine.full_pipeline(
        df_clean,
        feature_columns=feature_cols,
        pairwise_matrix=ahp_matrix,
        predicted_time_col='Predicted_Flight_Time',
        method='topsis'
    )

    print("\nТоп-5 БПЛА по TOPSIS (с учётом предсказанного времени):")
    print(result_df[['Model', 'TOPSIS_Score', 'Predicted_Flight_Time', 'Price_USD']].head())

    print(f"\nAHP CR = {engine.ahp.consistency_ratio:.4f} (согласована: {engine.ahp.is_consistent})")

    fig_weights = engine.ahp.plot_weights(feature_cols)
    plt.show()