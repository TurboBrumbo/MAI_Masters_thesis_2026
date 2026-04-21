"""
Интеллектуальная аналитическая система тактико-технических характеристик гражданских БПЛА.
Веб-интерфейс на Streamlit, объединяющий генерацию данных, предобработку, кластеризацию HDBSCAN,
обучение моделей Random Forest/XGBoost, SHAP-анализ и многокритериальное ранжирование (AHP+TOPSIS/VIKOR).
Соответствует главе 3 диссертации.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import joblib
import logging
from io import BytesIO
import base64

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_generator import AdvancedUAVDataGenerator
from preprocessing import DataPreprocessor
from clustering import UAVClusterAnalyzer
from ml_models import UAVFlightTimePredictor
from shap_analysis import SHAPAnalyzer, InteractiveSHAPDashboard
from mcdm import MCDMEngine

st.set_page_config(
    page_title="UAV Analytics System",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------------------------
# Инициализация состояния сессии (кэширование моделей и данных)
# ----------------------------------------------------------------------
SESSION_KEYS = [
    'data_raw', 'data_clean', 'data_scaled', 'feature_cols',
    'clusterer', 'df_clustered', 'predictor', 'best_model_name',
    'ml_metrics', 'shap_analyzer', 'shap_values', 'X_test',
    'mcdm_engine', 'ahp_weights', 'df_ranked'
]

for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None

# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------
def get_table_download_link(df: pd.DataFrame, filename: str = "data.csv") -> str:
    """Генерирует ссылку для скачивания DataFrame в CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Скачать CSV</a>'
    return href

def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list):
    """Строит тепловую карту корреляций с помощью Plotly."""
    corr = df[feature_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Матрица корреляций признаков"
    )
    return fig

# ----------------------------------------------------------------------
# Боковая панель: управление данными и настройки
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🚁 Управление системой")
    st.markdown("---")

    data_mode = st.radio(
        "Источник данных",
        ["📂 Загрузить CSV", "🔄 Сгенерировать синтетику"],
        index=0
    )

    if data_mode == "🔄 Сгенерировать синтетику":
        n_samples = st.slider("Количество моделей", 100, 1000, 300, step=50)
        gen_seed = st.number_input("Random seed", value=42)
        if st.button("✨ Сгенерировать датасет"):
            with st.spinner("Генерация данных..."):
                gen = AdvancedUAVDataGenerator(random_seed=gen_seed)
                df = gen.generate_full_dataset(n_samples=n_samples, include_seed_models=True)
                st.session_state.data_raw = df
                st.success(f"Сгенерировано {len(df)} записей")
    else:
        uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.data_raw = df
            st.success(f"Загружено {len(df)} записей")

    if st.session_state.data_raw is not None:
        st.markdown("---")
        st.subheader("Предобработка")
        scaling_method = st.selectbox("Масштабирование", ["standard", "minmax", "robust"])
        if st.button("🔄 Применить предобработку"):
            with st.spinner("Обработка данных..."):
                prep = DataPreprocessor(scaling_method=scaling_method)
                df_clean, df_scaled = prep.fit_transform(st.session_state.data_raw)
                st.session_state.data_clean = df_clean
                st.session_state.data_scaled = df_scaled
                st.session_state.feature_cols = prep.feature_columns
                st.success("Предобработка завершена")

    st.markdown("---")
    st.markdown("### ℹ️ Статус системы")
    if st.session_state.data_raw is not None:
        st.write(f"📊 Записей: {len(st.session_state.data_raw)}")
    else:
        st.write("⏳ Данные не загружены")
    if st.session_state.predictor is not None:
        st.write(f"🤖 ML модель: {st.session_state.best_model_name} (обучена)")
    else:
        st.write("🤖 ML модель: не обучена")

# ----------------------------------------------------------------------
# Основная область интерфейса
# ----------------------------------------------------------------------
st.title("🚁 Интеллектуальная аналитическая система БПЛА")
st.markdown("""
Гибридная платформа для анализа, кластеризации, прогнозирования и многокритериального выбора
беспилотных летательных аппаратов с применением методов машинного обучения и объяснимого ИИ.
""")

if st.session_state.data_raw is None:
    st.info("👈 Загрузите CSV файл или сгенерируйте синтетический датасет в боковой панели.")
    st.stop()

# ----------------------------------------------------------------------
# Вкладки интерфейса
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Данные и кластеризация",
    "🤖 ML моделирование",
    "🔍 SHAP объяснения",
    "⚖️ MCDM ранжирование",
    "📈 Отчёты и экспорт"
])

# ----------------------------- Вкладка 1: Данные и кластеризация -----------------------------
with tab1:
    st.header("Обзор данных и сегментация рынка")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Исходные данные")
        st.dataframe(st.session_state.data_raw.head(10))
        st.markdown(get_table_download_link(st.session_state.data_raw, "uav_raw.csv"), unsafe_allow_html=True)

    with col2:
        if st.session_state.data_clean is not None:
            st.subheader("Статистика после очистки")
            st.dataframe(st.session_state.data_clean.describe())

    st.markdown("---")
    st.subheader("Пространственная кластеризация (HDBSCAN)")

    if st.session_state.data_scaled is None:
        st.warning("Сначала выполните предобработку в боковой панели.")
    else:
        feature_options = st.session_state.feature_cols
        selected_features = st.multiselect(
            "Признаки для кластеризации",
            options=feature_options,
            default=feature_options[:5] if len(feature_options) >= 5 else feature_options
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            min_cluster_size = st.slider("min_cluster_size", 2, 30, 5)
            auto_tune = st.checkbox("Автоподбор параметров", value=True)
        with col2:
            if st.button("🚀 Запустить кластеризацию"):
                with st.spinner("Выполняется HDBSCAN..."):
                    clusterer = UAVClusterAnalyzer(min_cluster_size=min_cluster_size)
                    X = st.session_state.data_scaled[selected_features].values
                    if auto_tune:
                        best_params = clusterer.find_optimal_parameters(
                            X, min_cluster_size_range=(2, 15), scoring='silhouette'
                        )
                        st.success(f"Подобраны параметры: {best_params}")
                    else:
                        clusterer.fit(X)
                    df_clustered = clusterer.fit_dataframe(st.session_state.data_clean, selected_features)
                    st.session_state.clusterer = clusterer
                    st.session_state.df_clustered = df_clustered

        if st.session_state.df_clustered is not None:
            df_clust = st.session_state.df_clustered
            cluster_counts = df_clust['Cluster'].value_counts()
            st.write(f"Найдено кластеров: {len(cluster_counts[cluster_counts.index != '-1'])}, шум: {cluster_counts.get('-1', 0)}")

            fig_3d = st.session_state.clusterer.plot_3d_clusters(
                df_clust,
                x_col='Weight_g',
                y_col='Max_Speed_ms',
                z_col='Price_USD',
                hover_name='Model'
            )
            st.plotly_chart(fig_3d, use_container_width=True)

            with st.expander("📊 Статистика кластеров"):
                stats = st.session_state.clusterer.get_cluster_statistics(df_clust, selected_features)
                st.dataframe(stats)

# ----------------------------- Вкладка 2: ML моделирование -----------------------------
with tab2:
    st.header("Прогнозирование реального времени полёта")

    if st.session_state.data_clean is None:
        st.warning("Сначала выполните предобработку данных.")
    else:
        df_clean = st.session_state.data_clean
        if 'Actual_Flight_Time_min' not in df_clean.columns:
            st.error("В данных отсутствует целевая переменная 'Actual_Flight_Time_min'.")
        else:
            available_features = st.session_state.feature_cols
            default_features = [f for f in available_features if f != 'Actual_Flight_Time_min']
            selected_ml_features = st.multiselect(
                "Признаки для обучения",
                options=available_features,
                default=default_features[:6] if len(default_features) >= 6 else default_features
            )
            target = 'Actual_Flight_Time_min'

            col1, col2 = st.columns(2)
            with col1:
                model_type = st.selectbox("Тип модели", ["random_forest", "xgboost"])
                test_size = st.slider("Доля тестовой выборки", 0.1, 0.4, 0.2)
                n_estimators = st.slider("Количество деревьев", 50, 300, 150)

            with col2:
                if st.button("🎯 Обучить модель"):
                    with st.spinner("Обучение..."):
                        predictor = UAVFlightTimePredictor(model_type=model_type)
                        predictor.set_hyperparameters({'n_estimators': n_estimators})
                        X = df_clean[selected_ml_features]
                        y = df_clean[target]
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                        predictor.train(X_train, y_train, X_test, y_test)
                        st.session_state.predictor = predictor
                        st.session_state.best_model_name = model_type
                        st.session_state.ml_metrics = predictor.metrics_
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        st.success("Модель обучена!")

            if st.session_state.predictor is not None:
                st.markdown("---")
                st.subheader("Метрики модели")
                metrics = st.session_state.ml_metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("RMSE", f"{metrics.get('RMSE', 0):.3f}")
                col_m2.metric("MAE", f"{metrics.get('MAE', 0):.3f}")
                col_m3.metric("R²", f"{metrics.get('R2', 0):.3f}")
                col_m4.metric("MAPE", f"{metrics.get('MAPE', 0):.1f}%")

                y_pred = st.session_state.predictor.predict(st.session_state.X_test)
                fig_pred = st.session_state.predictor.plot_predictions_vs_actual(st.session_state.y_test, y_pred)
                st.pyplot(fig_pred)

                st.subheader("Важность признаков")
                fig_imp = st.session_state.predictor.plot_feature_importance(top_n=10)
                st.pyplot(fig_imp)

                if st.button("💾 Сохранить модель"):
                    model_bytes = BytesIO()
                    joblib.dump(st.session_state.predictor.model, model_bytes)
                    st.download_button(
                        label="📥 Скачать модель",
                        data=model_bytes.getvalue(),
                        file_name=f"uav_{model_type}_model.joblib",
                        mime="application/octet-stream"
                    )

# ----------------------------- Вкладка 3: SHAP объяснения -----------------------------
with tab3:
    st.header("Объяснимый ИИ (SHAP)")

    if st.session_state.predictor is None:
        st.warning("Сначала обучите ML модель на вкладке 'ML моделирование'.")
    else:
        predictor = st.session_state.predictor
        df_clean = st.session_state.data_clean
        feature_cols = [c for c in predictor.feature_names_ if c in df_clean.columns]

        # Инициализация SHAP анализатора (если ещё не)
        if st.session_state.shap_analyzer is None or st.button("🔄 Пересчитать SHAP"):
            with st.spinner("Вычисление SHAP-значений..."):
                feature_cols = predictor.feature_names_
                X = df_clean[feature_cols].copy()

                if X.isnull().any().any():
                    X = X.fillna(X.median())
                
                background = shap.sample(X, min(100, len(X)), random_state=42)
                analyzer = SHAPAnalyzer(predictor.model, feature_names=feature_cols)
                analyzer.initialize_explainer(background)
                analyzer.compute_shap_values(X)
                st.session_state.shap_analyzer = analyzer
                st.session_state.shap_values = analyzer.shap_values
                st.success("SHAP-значения вычислены")

        if st.session_state.shap_analyzer is not None:
            analyzer = st.session_state.shap_analyzer
            X_full = df_clean[feature_cols]

            st.subheader("Глобальная важность признаков (SHAP)")
            imp_df = analyzer.get_feature_importance_df(X_full)
            fig_bar = InteractiveSHAPDashboard.summary_bar_plotly(imp_df)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("Локальное объяснение (Waterfall)")
            model_names = df_clean['Model'].tolist()
            selected_model = st.selectbox("Выберите БПЛА для объяснения", model_names)
            idx = model_names.index(selected_model)
            instance = X_full.iloc[idx]
            shap_vals = analyzer.get_shap_values_for_instance(instance)

            fig_wf = InteractiveSHAPDashboard.waterfall_plotly(
                base_value=analyzer.expected_value,
                shap_values=shap_vals,
                feature_names=feature_cols,
                feature_values=instance.values,
                title=f"SHAP Waterfall: {selected_model}"
            )
            st.plotly_chart(fig_wf, use_container_width=True)

            with st.expander("🔽 Дополнительные графики SHAP"):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🐝 Показать Beeswarm"):
                        with st.spinner("Построение beeswarm..."):
                            fig_beeswarm = analyzer.plot_summary_beeswarm(X_full, max_display=15, show=False)
                            st.pyplot(fig_beeswarm)
                            plt.close(fig_beeswarm)
                with col2:
                    dep_feature = st.selectbox("Признак для dependence plot", feature_cols, key="dep_feature")
                    if st.button("📈 Построить dependence plot"):
                        with st.spinner(f"Построение зависимости для {dep_feature}..."):
                            fig_dep = analyzer.plot_dependence(dep_feature, X_full, show=False)
                            if fig_dep is not None:
                                st.pyplot(fig_dep)
                                plt.close(fig_dep)
                            else:
                                st.warning("Не удалось построить график.")

# ----------------------------- Вкладка 4: MCDM ранжирование -----------------------------
with tab4:
    st.header("Многокритериальное ранжирование (AHP + TOPSIS/VIKOR)")

    if st.session_state.data_clean is None:
        st.warning("Сначала выполните предобработку данных.")
    else:
        df = st.session_state.data_clean.copy()

        use_predicted_time = False
        predicted_col = None
        if st.session_state.predictor is not None:
            use_predicted_time = st.checkbox("Использовать предсказанное время полёта (ML) вместо заводского", value=True)
            if use_predicted_time:
                X_pred = df[st.session_state.predictor.feature_names_]
                df['Predicted_Flight_Time'] = st.session_state.predictor.predict(X_pred)
                predicted_col = 'Predicted_Flight_Time'

        available_criteria = ['Weight_g', 'Max_Speed_ms', 'Flight_Time_min',
                              'Range_km', 'Camera_MP', 'Price_USD']
        selected_criteria = st.multiselect(
            "Критерии для MCDM",
            options=available_criteria,
            default=available_criteria
        )

        st.markdown("**Типы критериев** (✓ = максимизировать, ✗ = минимизировать)")
        types_dict = {}
        cols = st.columns(len(selected_criteria))
        for i, crit in enumerate(selected_criteria):
            with cols[i]:
                if crit in ['Weight_g', 'Price_USD']:
                    types_dict[crit] = -1
                    st.markdown(f"**{crit}** ✗")
                else:
                    types_dict[crit] = 1
                    st.markdown(f"**{crit}** ✓")
        criteria_types = np.array([types_dict[c] for c in selected_criteria])

        st.subheader("Веса критериев (AHP)")
        st.markdown("Укажите относительную важность (сумма будет нормализована)")
        weights_raw = []
        cols_w = st.columns(len(selected_criteria))
        for i, crit in enumerate(selected_criteria):
            with cols_w[i]:
                w = st.slider(crit, 0.0, 1.0, 0.5, 0.05, key=f"w_{crit}")
                weights_raw.append(w)
        weights = np.array(weights_raw) / np.sum(weights_raw)

        method_mcdm = st.radio("Метод ранжирования", ["TOPSIS", "VIKOR"])

        if st.button("⚖️ Выполнить ранжирование"):
            with st.spinner("Расчёт MCDM..."):
                engine = MCDMEngine()
                engine.set_alternative_names(df['Model'].tolist())
                engine.set_criteria_names(selected_criteria)

                df_mcdm = df.copy()
                time_col = 'Flight_Time_min'
                if predicted_col:
                    df_mcdm[time_col] = df_mcdm[predicted_col]

                matrix = df_mcdm[selected_criteria].values

                if method_mcdm == "TOPSIS":
                    result_df = engine.run_topsis(matrix, weights, criteria_types)
                    df_ranked = df_mcdm.copy()
                    df_ranked['TOPSIS_Score'] = result_df['TOPSIS_Score']
                    df_ranked['TOPSIS_Rank'] = result_df['TOPSIS_Rank']
                    score_col = 'TOPSIS_Score'
                    rank_col = 'TOPSIS_Rank'
                else:
                    result_df = engine.run_vikor(matrix, weights, criteria_types)
                    df_ranked = df_mcdm.copy()
                    df_ranked['VIKOR_Q'] = result_df['Q']
                    df_ranked['VIKOR_Rank'] = result_df['VIKOR_Rank']
                    score_col = 'VIKOR_Q'
                    rank_col = 'VIKOR_Rank'

                st.session_state.df_ranked = df_ranked.sort_values(score_col, ascending=(method_mcdm == "VIKOR"))
                st.session_state.mcdm_method = method_mcdm
                st.session_state.mcdm_score_col = score_col
                st.session_state.mcdm_rank_col = rank_col

        if st.session_state.df_ranked is not None:
            df_ranked = st.session_state.df_ranked
            method_mcdm = st.session_state.mcdm_method
            score_col = st.session_state.mcdm_score_col
            rank_col = st.session_state.mcdm_rank_col

            st.subheader("Результаты ранжирования")
            display_cols = ['Model'] + selected_criteria + [score_col, rank_col]
            if predicted_col:
                display_cols.insert(2, predicted_col)
            st.dataframe(df_ranked[display_cols].head(20).style.highlight_max(subset=[score_col], color='lightgreen'))

            fig_rank = px.bar(
                df_ranked.head(15),
                x='Model',
                y=score_col,
                color=score_col,
                title=f"Топ-15 по {method_mcdm}",
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_rank, use_container_width=True)

            st.markdown(get_table_download_link(df_ranked, "uav_ranking.csv"), unsafe_allow_html=True)

# ----------------------------- Вкладка 5: Отчёты и экспорт -----------------------------
with tab5:
    st.header("Сводный отчёт и экспорт")

    if st.session_state.data_clean is not None:
        df = st.session_state.data_clean

        st.subheader("Экспорт обработанных данных")
        st.markdown(get_table_download_link(df, "uav_processed.csv"), unsafe_allow_html=True)

        st.subheader("Корреляционный анализ")
        if st.button("Построить матрицу корреляций"):
            fig_corr = plot_correlation_heatmap(df, st.session_state.feature_cols)
            st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Генерация отчёта")
        if st.button("📄 Сформировать отчёт"):
            report_lines = []
            report_lines.append("# Отчёт по анализу БПЛА\n")
            report_lines.append(f"Датасет: {len(df)} моделей\n")
            report_lines.append("## Статистика признаков\n")
            report_lines.append(df[st.session_state.feature_cols].describe().to_string())
            if st.session_state.ml_metrics:
                report_lines.append("\n## ML модель\n")
                report_lines.append(f"Тип: {st.session_state.best_model_name}\n")
                report_lines.append(f"Метрики: {st.session_state.ml_metrics}\n")
            if st.session_state.df_ranked is not None:
                report_lines.append("\n## Топ-5 по MCDM\n")
                top5 = st.session_state.df_ranked.head(5)[['Model', st.session_state.df_ranked.columns[-1]]]
                report_lines.append(top5.to_string(index=False))

            report_text = "\n".join(report_lines)
            st.download_button(
                label="📥 Скачать отчёт (TXT)",
                data=report_text,
                file_name="uav_report.txt",
                mime="text/plain"
            )

        st.markdown("---")
        st.markdown("### О системе")
        st.markdown("""
        **Интеллектуальная аналитическая система тактико-технических характеристик БПЛА**
        разработана в рамках магистерской диссертации (МАИ, 2026).

        **Ключевые возможности:**
        - Генерация синтетических данных на основе физических моделей
        - Кластеризация HDBSCAN для сегментации рынка
        - Прогнозирование реального времени полёта (Random Forest / XGBoost)
        - Объяснение предсказаний с помощью SHAP
        - Многокритериальное ранжирование AHP + TOPSIS / VIKOR
        """)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    pass