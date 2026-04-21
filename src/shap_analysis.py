"""
Модуль объяснимого искусственного интеллекта (XAI) на основе SHAP.
Реализует вычисление SHAP-значений для ансамблевых моделей (Random Forest, XGBoost),
построение графиков: waterfall, summary bar, beeswarm, dependence, force plot.
Обеспечивает прозрачность предсказаний ML-моделей и повышает доверие к системе.
Соответствует разделу 2.4 диссертации.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import warnings
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    Класс для генерации и визуализации SHAP-объяснений для моделей прогноза времени полёта.
    Поддерживает модели, совместимые с shap.TreeExplainer (RandomForest, XGBoost и др.)
    """

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[shap.TreeExplainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.expected_value: Optional[float] = None
        self.X_background: Optional[np.ndarray] = None
        self.is_initialized = False

    def initialize_explainer(self, X_background: Union[pd.DataFrame, np.ndarray]):
        """Инициализирует TreeExplainer с фоновыми данными."""
        if isinstance(X_background, pd.DataFrame):
            self.feature_names = X_background.columns.tolist()
            self.X_background = X_background.values
        else:
            self.X_background = X_background

        self.explainer = shap.TreeExplainer(
            self.model,
            self.X_background,
            feature_perturbation="tree_path_dependent"
        )
        if hasattr(self.explainer, 'check_additivity'):
            self.explainer.check_additivity = False
        self.expected_value = self.explainer.expected_value
        if isinstance(self.expected_value, np.ndarray):
            self.expected_value = self.expected_value[0]
        self.is_initialized = True
        logger.info(f"SHAP TreeExplainer инициализирован. Expected value = {self.expected_value:.4f}")

    def compute_shap_values(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Вычисляет SHAP-значения, игнорируя ошибку аддитивности."""
        if not self.is_initialized:
            raise RuntimeError("Сначала вызовите initialize_explainer().")
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.shap_values = self.explainer.shap_values(X_array, check_additivity=False)
            except TypeError:
                self.shap_values = self.explainer.shap_values(X_array)
            except shap.utils._exceptions.ExplainerError as e:
                if "Additivity check failed" in str(e):
                    logger.warning("Ошибка аддитивности SHAP проигнорирована. Значения могут быть приблизительными.")
                    self.explainer.check_additivity = False
                    self.shap_values = self.explainer.shap_values(X_array)
                else:
                    raise e
        logger.info(f"SHAP-значения вычислены для {len(X_array)} экземпляров.")
        return self.shap_values

    def get_shap_values_for_instance(self, X_instance: Union[pd.DataFrame, np.ndarray, pd.Series]) -> np.ndarray:
        """
        Возвращает SHAP-значения для одного экземпляра.
        """
        if isinstance(X_instance, (pd.Series, pd.DataFrame)):
            X_array = X_instance.values.reshape(1, -1)
        elif isinstance(X_instance, np.ndarray):
            X_array = X_instance.reshape(1, -1) if X_instance.ndim == 1 else X_instance
        else:
            raise ValueError("Неподдерживаемый тип для X_instance.")
        return self.explainer.shap_values(X_array)[0]

    def plot_waterfall(
        self,
        shap_values_single: np.ndarray,
        features_single: Union[pd.Series, np.ndarray],
        max_display: int = 10,
        show: bool = True
    ):
        """
        Строит waterfall plot для одного экземпляра.
        """
        if self.feature_names is None:
            raise ValueError("Не заданы имена признаков.")
        if not self.is_initialized:
            raise RuntimeError("Explainer не инициализирован.")

        if isinstance(features_single, pd.Series):
            feature_values = features_single.values
        else:
            feature_values = np.array(features_single)

        exp = shap.Explanation(
            values=shap_values_single,
            base_values=self.expected_value,
            data=feature_values,
            feature_names=self.feature_names
        )

        fig = plt.figure(figsize=(10, 6))
        shap.waterfall_plot(exp, max_display=max_display, show=False)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_summary_bar(self, X: Union[pd.DataFrame, np.ndarray], max_display: int = 15, show: bool = True):
        """
        Строит столбчатую диаграмму средней абсолютной важности признаков.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, X, feature_names=self.feature_names,
            plot_type="bar", max_display=max_display, show=False
        )
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_summary_beeswarm(self, X: Union[pd.DataFrame, np.ndarray], max_display: int = 15, show: bool = True):
        """
        Строит beeswarm plot (точки) распределения SHAP-значений.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, X, feature_names=self.feature_names,
            max_display=max_display, show=False
        )
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_dependence(
        self,
        feature_name: str,
        X: Union[pd.DataFrame, np.ndarray],
        interaction_feature: Optional[str] = None,
        show: bool = True
    ):
        """
        Строит график зависимости SHAP-значений от значения признака.
        Возвращает matplotlib figure.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        if self.feature_names is None or feature_name not in self.feature_names:
            raise ValueError(f"Признак '{feature_name}' не найден в списке признаков.")
        feature_idx = self.feature_names.index(feature_name)

        if interaction_feature is not None and interaction_feature in self.feature_names:
            interaction_idx = self.feature_names.index(interaction_feature)
        else:
            interaction_idx = None

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_idx,
            ax=ax,
            show=False
        )
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_force(
        self,
        shap_values_single: np.ndarray,
        features_single: Union[pd.Series, np.ndarray],
        matplotlib: bool = True,
        show: bool = True
    ):
        """
        Строит force plot для одного экземпляра (локальное объяснение).
        """
        if self.feature_names is None:
            raise ValueError("Не заданы имена признаков.")
        if not self.is_initialized:
            raise RuntimeError("Explainer не инициализирован.")

        if isinstance(features_single, pd.Series):
            feature_values = features_single.values
        else:
            feature_values = np.array(features_single)

        if matplotlib:
            fig = plt.figure(figsize=(20, 3))
            shap.force_plot(
                self.expected_value, shap_values_single, feature_values,
                feature_names=self.feature_names, matplotlib=True, show=False
            )
            plt.tight_layout()
            if show:
                plt.show()
            return fig
        else:
            return shap.force_plot(
                self.expected_value, shap_values_single, feature_values,
                feature_names=self.feature_names, matplotlib=False
            )

    def plot_force_multiple(
        self,
        shap_values_multiple: np.ndarray,
        X_multiple: Union[pd.DataFrame, np.ndarray],
        matplotlib: bool = True,
        show: bool = True
    ):
        """
        Строит stacked force plot для нескольких экземпляров.
        """
        if not self.is_initialized:
            raise RuntimeError("Explainer не инициализирован.")

        if matplotlib:
            fig = plt.figure(figsize=(20, len(shap_values_multiple) * 0.5))
            shap.force_plot(
                self.expected_value, shap_values_multiple, X_multiple,
                feature_names=self.feature_names, matplotlib=True, show=False
            )
            plt.tight_layout()
            if show:
                plt.show()
            return fig
        else:
            return shap.force_plot(
                self.expected_value, shap_values_multiple, X_multiple,
                feature_names=self.feature_names, matplotlib=False
            )

    def get_feature_importance_df(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Возвращает DataFrame со средней абсолютной важностью признаков по SHAP.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        df_imp = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_ABS_SHAP': mean_abs_shap
        }).sort_values('Mean_ABS_SHAP', ascending=False)
        return df_imp

    def explain_global(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Возвращает словарь с глобальными объяснениями (важность признаков, ожидаемое значение).
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        return {
            'expected_value': self.expected_value,
            'feature_importance': self.get_feature_importance_df(X).to_dict('records'),
            'mean_abs_shap': np.abs(self.shap_values).mean(axis=0).tolist(),
            'feature_names': self.feature_names
        }

    def explain_local(self, X_single: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Возвращает словарь с локальным объяснением для одного экземпляра.
        """
        shap_vals = self.get_shap_values_for_instance(X_single)
        if isinstance(X_single, pd.Series):
            feature_values = X_single.values
        else:
            feature_values = np.array(X_single)

        explanation = {
            'base_value': self.expected_value,
            'predicted': self.expected_value + np.sum(shap_vals),
            'shap_values': shap_vals.tolist(),
            'feature_values': feature_values.tolist(),
            'feature_names': self.feature_names,
            'contributions': {
                name: val for name, val in zip(self.feature_names, shap_vals)
            }
        }
        return explanation


class InteractiveSHAPDashboard:
    """
    Вспомогательный класс для подготовки данных к визуализации в Streamlit/Plotly.
    Генерирует графики Plotly на основе SHAP-значений.
    """

    @staticmethod
    def waterfall_plotly(
        base_value: float,
        shap_values: np.ndarray,
        feature_names: List[str],
        feature_values: np.ndarray,
        title: str = "SHAP Waterfall"
    ):
        """
        Создаёт интерактивный waterfall plot с помощью Plotly.
        """
        import plotly.graph_objects as go

        indices = np.argsort(np.abs(shap_values))[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_values = [shap_values[i] for i in indices]
        sorted_feat_vals = [feature_values[i] for i in indices]

        text_labels = [
            f"{name} = {val:.2f}" for name, val in zip(sorted_names, sorted_feat_vals)
        ]

        cumulative = base_value
        measures = []
        y_vals = []
        texts = []
        for i, v in enumerate(sorted_values):
            measures.append("relative")
            y_vals.append(v)
            texts.append(f"{v:+.2f}")
        measures.append("total")
        y_vals.append(base_value + sum(sorted_values))
        texts.append(f"{base_value + sum(sorted_values):.2f}")

        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=measures,
            x=text_labels + ["Итог"],
            textposition="outside",
            text=texts,
            y=y_vals,
            base=base_value,
            decreasing={"marker": {"color": "crimson"}},
            increasing={"marker": {"color": "forestgreen"}},
            totals={"marker": {"color": "navy"}},
            connector={"line": {"color": "gray", "dash": "dot"}}
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Признаки",
            yaxis_title="Вклад в предсказание (SHAP)",
            height=600
        )
        return fig

    @staticmethod
    def summary_bar_plotly(importance_df: pd.DataFrame, title: str = "Важность признаков (SHAP)"):
        """
        Строит столбчатую диаграмму важности с Plotly.
        """
        import plotly.express as px

        fig = px.bar(
            importance_df.head(15),
            x='Mean_ABS_SHAP',
            y='Feature',
            orientation='h',
            title=title,
            color='Mean_ABS_SHAP',
            color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        return fig


if __name__ == "__main__":
    from data_generator import AdvancedUAVDataGenerator
    from preprocessing import DataPreprocessor
    from ml_models import UAVFlightTimePredictor
    from sklearn.model_selection import train_test_split

    gen = AdvancedUAVDataGenerator(random_seed=42)
    df = gen.generate_full_dataset(n_samples=300, include_seed_models=True)

    prep = DataPreprocessor(scaling_method='standard')
    df_clean, df_scaled = prep.fit_transform(df)

    feature_cols = ['Weight_g', 'Max_Speed_ms', 'Battery_Capacity_mAh',
                    'Propeller_Size_inch', 'Flight_Time_min', 'Range_km',
                    'Camera_MP', 'Price_USD']
    target = 'Actual_Flight_Time_min'

    X = df_clean[feature_cols]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    predictor = UAVFlightTimePredictor(model_type='random_forest', random_state=42)
    predictor.train(X_train, y_train)

    analyzer = SHAPAnalyzer(predictor.model, feature_names=feature_cols)
    background = shap.sample(X_train, 100)
    analyzer.initialize_explainer(background)
    analyzer.compute_shap_values(X_test)

    imp_df = analyzer.get_feature_importance_df(X_test)
    print("Важность признаков (SHAP):")
    print(imp_df)

    idx = 0
    shap_vals = analyzer.get_shap_values_for_instance(X_test.iloc[idx])
    analyzer.plot_waterfall(shap_vals, X_test.iloc[idx], max_display=10)

    analyzer.plot_summary_bar(X_test, max_display=10)

    analyzer.plot_dependence('Weight_g', X_test)

    local_exp = analyzer.explain_local(X_test.iloc[idx])
    print("\nЛокальное объяснение:")
    print(f"Базовое значение: {local_exp['base_value']:.2f}")
    print(f"Предсказание: {local_exp['predicted']:.2f}")
    print("Вклады признаков:")
    for name, contrib in local_exp['contributions'].items():
        print(f"  {name}: {contrib:+.2f}")

    plt.show()