"""
Модуль иерархической плотностной кластеризации (HDBSCAN) для сегментации БПЛА.
Реализует подбор оптимальных параметров, оценку качества (силуэт, индекс Дэвиса-Болдина),
визуализацию дендрограммы сжатия и 3D-проекцию кластеров.
Соответствует разделу 2.1 и 3.3 диссертации.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hdbscan
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class UAVClusterAnalyzer:
    """
    Класс для проведения кластеризации HDBSCAN с расширенной диагностикой.
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        metric: str = 'euclidean',
        cluster_selection_epsilon: float = 0.0,
        alpha: float = 1.0
    ):
        """
        Параметры HDBSCAN:
        - min_cluster_size: минимальный размер кластера (основной параметр)
        - min_samples: число соседей для оценки плотности (по умолчанию = min_cluster_size)
        - metric: метрика расстояния
        - cluster_selection_epsilon: порог для выделения кластеров
        - alpha: коэффициент сжатия дерева
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.alpha = alpha
        self.clusterer: Optional[hdbscan.HDBSCAN] = None
        self.labels_: Optional[np.ndarray] = None
        self.probabilities_: Optional[np.ndarray] = None
        self._is_fitted = False
        logger.info(f"Инициализирован HDBSCAN с min_cluster_size={min_cluster_size}")

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Обучает кластеризатор на матрице признаков X.
        Возвращает метки кластеров.
        """
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            alpha=self.alpha,
            prediction_data=True,
            gen_min_span_tree=True
        )
        self.labels_ = self.clusterer.fit_predict(X)
        self.probabilities_ = self.clusterer.probabilities_
        self._is_fitted = True
        logger.info(f"Кластеризация выполнена. Найдено кластеров: {len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)}")
        logger.info(f"Точек шума: {np.sum(self.labels_ == -1)}")
        return self.labels_

    def fit_dataframe(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Принимает DataFrame и список признаков, возвращает DataFrame с добавленной колонкой 'Cluster'.
        """
        if not feature_columns:
            raise ValueError("Список признаков пуст.")
        X = df[feature_columns].values
        self.fit(X)
        df_clustered = df.copy()
        df_clustered['Cluster'] = self.labels_.astype(str)
        df_clustered['Cluster_Probability'] = self.probabilities_
        return df_clustered

    def find_optimal_parameters(
        self,
        X: np.ndarray,
        min_cluster_size_range: Tuple[int, int] = (3, 20),
        min_samples_range: Optional[Tuple[int, int]] = None,
        metric: str = 'euclidean',
        scoring: str = 'silhouette'
    ) -> Dict[str, Any]:
        """
        Простой поиск по сетке для подбора min_cluster_size и min_samples.
        Возвращает лучшие параметры и соответствующую модель.
        scoring: 'silhouette', 'davies_bouldin', 'calinski_harabasz'
        """
        if min_samples_range is None:
            min_samples_range = (min_cluster_size_range[0], min_cluster_size_range[1])

        best_score = -np.inf if scoring != 'davies_bouldin' else np.inf
        best_params = {}
        best_labels = None
        best_model = None

        logger.info(f"Запуск поиска параметров по сетке: min_cluster_size {min_cluster_size_range}, min_samples {min_samples_range}")

        for mcs in range(min_cluster_size_range[0], min_cluster_size_range[1] + 1):
            for ms in range(min_samples_range[0], min_samples_range[1] + 1):
                try:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=mcs,
                        min_samples=ms,
                        metric=metric,
                        cluster_selection_epsilon=self.cluster_selection_epsilon,
                        alpha=self.alpha
                    )
                    labels = clusterer.fit_predict(X)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters < 2:
                        continue

                    mask = labels != -1
                    if np.sum(mask) < 10:
                        continue
                    X_core = X[mask]
                    labels_core = labels[mask]

                    if scoring == 'silhouette':
                        score = silhouette_score(X_core, labels_core)
                        better = score > best_score
                    elif scoring == 'davies_bouldin':
                        score = davies_bouldin_score(X_core, labels_core)
                        better = score < best_score
                    elif scoring == 'calinski_harabasz':
                        score = calinski_harabasz_score(X_core, labels_core)
                        better = score > best_score
                    else:
                        raise ValueError(f"Неизвестная метрика: {scoring}")

                    if better:
                        best_score = score
                        best_params = {'min_cluster_size': mcs, 'min_samples': ms}
                        best_labels = labels
                        best_model = clusterer

                    logger.debug(f"mcs={mcs}, ms={ms}, clusters={n_clusters}, {scoring}={score:.4f}")
                except Exception as e:
                    logger.warning(f"Ошибка при mcs={mcs}, ms={ms}: {e}")
                    continue

        if best_model is not None:
            self.clusterer = best_model
            self.labels_ = best_labels
            self.probabilities_ = best_model.probabilities_
            self.min_cluster_size = best_params['min_cluster_size']
            self.min_samples = best_params['min_samples']
            self._is_fitted = True
            logger.info(f"Лучшие параметры: {best_params}, {scoring}={best_score:.4f}")
        else:
            logger.warning("Не удалось найти подходящие параметры, используются значения по умолчанию.")
            self.fit(X)
            best_params = {'min_cluster_size': self.min_cluster_size, 'min_samples': self.min_samples}

        return best_params

    def get_cluster_statistics(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Возвращает сводную статистику по кластерам: средние значения признаков, размер.
        """
        if not self._is_fitted:
            raise RuntimeError("Кластеризатор не обучен.")
        df_copy = df.copy()
        df_copy['Cluster'] = self.labels_.astype(str)
        stats = df_copy.groupby('Cluster')[feature_columns].mean().round(2)
        stats['Count'] = df_copy['Cluster'].value_counts()
        stats['Percentage'] = (stats['Count'] / len(df_copy) * 100).round(1)
        return stats

    def plot_condensed_tree(self, figsize=(12, 6)):
        """Визуализация сжатого дерева кластеров."""
        if not self._is_fitted or self.clusterer is None:
            raise RuntimeError("Модель не обучена.")
        fig, ax = plt.subplots(figsize=figsize)
        self.clusterer.condensed_tree_.plot(ax=ax, select_clusters=True,
                                            label_clusters=True,
                                            axis='xy')
        ax.set_title("Сжатое дерево кластеров HDBSCAN")
        return fig

    def plot_single_linkage_tree(self, figsize=(12, 6)):
        """Визуализация полного дерева одиночной связи."""
        if not self._is_fitted or self.clusterer is None:
            raise RuntimeError("Модель не обучена.")
        fig, ax = plt.subplots(figsize=figsize)
        self.clusterer.single_linkage_tree_.plot(ax=ax, cmap='viridis')
        ax.set_title("Дерево одиночной связи (single linkage)")
        return fig

    def plot_k_distance(self, X: np.ndarray, k: int = 5, figsize=(8, 5)):
        """
        График k-расстояний для выбора min_samples (эвристика).
        """
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(X)
        distances, _ = neigh.kneighbors(X)
        k_dist = np.sort(distances[:, -1])
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(k_dist)
        ax.set_xlabel("Точки (отсортированы)")
        ax.set_ylabel(f"{k}-е расстояние")
        ax.set_title(f"График {k}-расстояний для выбора min_samples")
        ax.grid(True, alpha=0.3)
        return fig

    def plot_3d_clusters(
        self,
        df: pd.DataFrame,
        x_col: str = 'Weight_g',
        y_col: str = 'Max_Speed_ms',
        z_col: str = 'Price_USD',
        color_col: str = 'Cluster',
        hover_name: str = 'Model',
        title: str = "3D визуализация кластеров БПЛА"
    ):
        """Интерактивный 3D scatter plot с Plotly."""
        fig = px.scatter_3d(
            df, x=x_col, y=y_col, z=z_col,
            color=color_col,
            hover_name=hover_name,
            title=title,
            opacity=0.8,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_traces(marker=dict(size=5))
        return fig

    def plot_cluster_profiles(self, df: pd.DataFrame, feature_columns: List[str]):
        """
        Радарная диаграмма (spider chart) для сравнения средних профилей кластеров.
        """
        if not self._is_fitted:
            raise RuntimeError("Модель не обучена.")
        df_copy = df.copy()
        df_copy['Cluster'] = self.labels_.astype(str)
        df_clean = df_copy[df_copy['Cluster'] != '-1']
        if df_clean.empty:
            return None

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df_clean[feature_columns])
        scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)
        scaled_df['Cluster'] = df_clean['Cluster'].values

        cluster_means = scaled_df.groupby('Cluster')[feature_columns].mean()

        fig = go.Figure()
        for cluster in cluster_means.index:
            values = cluster_means.loc[cluster].tolist()
            values.append(values[0])
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=feature_columns + [feature_columns[0]],
                fill='toself',
                name=f'Cluster {cluster}'
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Профили кластеров (средние нормализованные значения)"
        )
        return fig


if __name__ == "__main__":
    from data_generator import AdvancedUAVDataGenerator
    from preprocessing import DataPreprocessor

    gen = AdvancedUAVDataGenerator(random_seed=42)
    df = gen.generate_full_dataset(n_samples=300, include_seed_models=True)

    prep = DataPreprocessor(scaling_method='standard')
    df_clean, df_scaled = prep.fit_transform(df)

    feature_cols = ['Weight_g', 'Max_Speed_ms', 'Battery_Capacity_mAh',
                    'Propeller_Size_inch', 'Flight_Time_min', 'Price_USD']

    analyzer = UAVClusterAnalyzer()
    X = df_scaled[feature_cols].values

    best_params = analyzer.find_optimal_parameters(
        X, min_cluster_size_range=(3, 15),
        scoring='silhouette'
    )
    print("Лучшие параметры:", best_params)

    df_clustered = analyzer.fit_dataframe(df_scaled, feature_cols)

    stats = analyzer.get_cluster_statistics(df_clustered, feature_cols)
    print("\nСтатистика кластеров:")
    print(stats)

    fig_tree = analyzer.plot_condensed_tree()
    plt.show()

    fig_3d = analyzer.plot_3d_clusters(df_clustered)
    fig_3d.show()