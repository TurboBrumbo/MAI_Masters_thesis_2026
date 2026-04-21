"""
Расширенный генератор синтетического набора данных гражданских БПЛА.
Реализует физически обоснованную модель расчёта реального времени полёта,
учитывающую аэродинамические штрафы, характеристики батареи и погодные условия.
Соответствует главе 2.3 диссертации.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UAVPhysicsModel:
    """
    Физическая модель для расчёта реального времени полёта.
    Учитывает:
    - Массу аппарата
    - Ёмкость батареи
    - Эффективность винтов (размер)
    - Скорость ветра
    - Температуру воздуха
    """

    def __init__(
        self,
        base_efficiency: float = 0.85,       # базовая эффективность силовой установки
        battery_voltage: float = 22.2,        # номинальное напряжение (6S LiPo)
        air_density: float = 1.225            # плотность воздуха кг/м³
    ):
        self.base_efficiency = base_efficiency
        self.battery_voltage = battery_voltage
        self.air_density = air_density

    def theoretical_hover_power(self, mass_kg: float, prop_diameter_inch: float) -> float:
        """
        Теоретическая мощность висения (Вт) на основе теории импульса.
        P = (m * g)^1.5 / sqrt(2 * rho * A)
        где A - площадь, ометаемая винтами.
        Для простоты предполагаем квадрокоптер с 4 винтами.
        """
        g = 9.81
        thrust_per_motor = mass_kg * g / 4.0
        # Площадь одного винта (м²)
        radius_m = (prop_diameter_inch * 0.0254) / 2.0
        area = np.pi * radius_m ** 2
        # Мощность на один винт (идеальная)
        p_ideal = thrust_per_motor * np.sqrt(thrust_per_motor / (2 * self.air_density * area))
        # Общая мощность с учётом КПД
        total_power = 4 * p_ideal / self.base_efficiency
        return total_power

    def flight_time_from_battery(
        self,
        mass_g: float,
        battery_capacity_mah: float,
        prop_size_inch: float,
        wind_speed_ms: float = 0.0,
        temperature_c: float = 20.0
    ) -> float:
        """
        Рассчитывает ожидаемое время полёта (мин) с учётом всех факторов.
        """
        mass_kg = mass_g / 1000.0
        # Базовая мощность висения в штиль
        power_hover = self.theoretical_hover_power(mass_kg, prop_size_inch)

        # Штраф за ветер: дополнительная мощность для удержания позиции
        # (эмпирическая модель: ~ кубическая зависимость от скорости ветра)
        wind_factor = 1.0 + 0.05 * wind_speed_ms + 0.002 * (wind_speed_ms ** 3)

        # Температурный коэффициент (LiPo батареи теряют ёмкость при низких температурах)
        temp_factor = 1.0
        if temperature_c < 15:
            temp_factor = 1.0 + 0.02 * (15 - temperature_c)  # падение ёмкости
        elif temperature_c > 35:
            temp_factor = 1.0 + 0.01 * (temperature_c - 35)   # повышенное сопротивление

        # Скорректированная потребляемая мощность
        power_actual = power_hover * wind_factor * temp_factor

        # Энергия батареи (Вт*ч)
        energy_wh = (battery_capacity_mah / 1000.0) * self.battery_voltage

        # Время полёта в часах
        time_hours = energy_wh / power_actual
        time_minutes = time_hours * 60.0

        # Ограничение: не менее 5 минут, не более 120
        return np.clip(time_minutes, 5.0, 120.0)


class AdvancedUAVDataGenerator:
    """
    Генератор синтетического датасета, интегрирующий физическую модель.
    Позволяет создавать как полностью синтетические данные, так и расширять
    seed-датасет коммерческих моделей.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)
        self.physics = UAVPhysicsModel()
        logger.info("Генератор данных инициализирован.")

    def _generate_base_parameters(self, n_samples: int) -> pd.DataFrame:
        """
        Генерация базовых конструктивных параметров дронов.
        Распределения подобраны на основе анализа реального рынка.
        """
        # Масса (г) - смесь логнормального и экспоненциального распределений
        # Основная масса лёгких дронов + тяжёлый хвост промышленных
        weights = self.rng.lognormal(mean=7.2, sigma=0.9, size=n_samples)
        weights = np.clip(weights, 200, 40000).astype(int)

        # Максимальная скорость (м/с)
        speeds = self.rng.normal(18.0, 4.5, n_samples)
        speeds = np.clip(speeds, 8.0, 40.0).round(1)

        # Ёмкость батареи (мАч) - коррелирует с массой
        battery_base = weights * 0.9
        battery_noise = self.rng.normal(0, 400, n_samples)
        battery = (battery_base + battery_noise).astype(int)
        battery = np.clip(battery, 800, 30000)

        # Размер пропеллеров (дюймы) - зависит от массы
        prop_mean = 5.0 + (weights / 1000.0) * 0.8
        prop = self.rng.normal(prop_mean, 1.5, n_samples)
        prop = np.clip(prop, 4.0, 24.0).round(1)

        # Заявленное производителем время полёта (мин) - оптимистичное
        flight_time_mfg = (battery / 180.0) + self.rng.normal(0, 5, n_samples)
        flight_time_mfg = np.clip(flight_time_mfg, 10, 90).astype(int)

        # Дальность связи (км) - зависит от цены и класса
        range_km = (battery / 800.0) + self.rng.normal(0, 3, n_samples)
        range_km = np.clip(range_km, 2, 35).astype(int)

        # Разрешение камеры (МП)
        camera_options = [8, 12, 20, 48, 64, 108]
        camera = self.rng.choice(camera_options, n_samples, p=[0.05, 0.20, 0.30, 0.25, 0.15, 0.05])

        # Цена (USD) - нелинейная зависимость от характеристик
        price = (weights * 2.2 + battery * 0.6 + camera * 30 + self.rng.normal(0, 800, n_samples))
        price = np.clip(price, 300, 45000).astype(int)

        # Погодные условия (для расчёта реального времени)
        wind = self.rng.exponential(scale=3.5, size=n_samples)  # м/с
        wind = np.clip(wind, 0, 18).round(1)
        temperature = self.rng.normal(20.0, 12.0, n_samples)   # °C
        temperature = np.clip(temperature, -10, 45).round(1)

        df = pd.DataFrame({
            'Weight_g': weights,
            'Max_Speed_ms': speeds,
            'Battery_Capacity_mAh': battery,
            'Propeller_Size_inch': prop,
            'Flight_Time_min': flight_time_mfg,
            'Range_km': range_km,
            'Camera_MP': camera,
            'Price_USD': price,
            'Wind_Speed_ms': wind,
            'Temperature_C': temperature
        })
        return df

    def _compute_actual_flight_time(self, df: pd.DataFrame) -> pd.Series:
        """
        Применяет физическую модель для расчёта Actual_Flight_Time_min.
        """
        actual_times = []
        for _, row in df.iterrows():
            time_real = self.physics.flight_time_from_battery(
                mass_g=row['Weight_g'],
                battery_capacity_mah=row['Battery_Capacity_mAh'],
                prop_size_inch=row['Propeller_Size_inch'],
                wind_speed_ms=row['Wind_Speed_ms'],
                temperature_c=row['Temperature_C']
            )
            actual_times.append(time_real)
        return pd.Series(actual_times, index=df.index).round(1)

    def generate_full_dataset(
        self,
        n_samples: int = 500,
        include_seed_models: bool = True,
        seed_models: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Основной метод генерации датасета.

        Параметры:
        - n_samples: целевое количество моделей (если не заданы seed_models)
        - include_seed_models: добавлять ли эталонные коммерческие модели
        - seed_models: пользовательский DataFrame с эталонными моделями

        Возвращает:
        - pd.DataFrame с полным набором признаков, включая 'Actual_Flight_Time_min'
        """
        dfs = []

        if include_seed_models:
            if seed_models is None:
                seed_data = {
                    'Model': [
                        'DJI Matrice 350 RTK',
                        'DJI Mavic 3 Enterprise',
                        'Autel EVO Max 4T',
                        'Skydio X10',
                        'Parrot Anafi Ai',
                        'DJI Agras T40',
                        'Yuneec H850-RTK',
                        'Freefly Astro'
                    ],
                    'Weight_g': [6300, 920, 1990, 1300, 898, 42000, 8500, 2500],
                    'Max_Speed_ms': [23, 21, 23, 20, 16, 10, 18, 22],
                    'Battery_Capacity_mAh': [5800, 5000, 7100, 5400, 3400, 30000, 12000, 8000],
                    'Propeller_Size_inch': [21.0, 8.0, 12.0, 10.0, 7.0, 38.0, 22.0, 15.0],
                    'Flight_Time_min': [55, 45, 42, 35, 32, 30, 45, 35],
                    'Range_km': [20, 15, 25, 10, 22, 5, 15, 12],
                    'Camera_MP': [48, 20, 48, 48, 48, 12, 64, 48],
                    'Price_USD': [15000, 3500, 9000, 7000, 5000, 25000, 12000, 18000],
                    'Wind_Speed_ms': [3.0, 2.5, 4.0, 3.5, 2.0, 5.0, 3.0, 4.0],
                    'Temperature_C': [22.0, 20.0, 25.0, 18.0, 23.0, 20.0, 21.0, 19.0]
                }
                seed_df = pd.DataFrame(seed_data)
            else:
                seed_df = seed_models.copy()
            # Вычисляем реальное время для seed-моделей
            seed_df['Actual_Flight_Time_min'] = self._compute_actual_flight_time(seed_df)
            dfs.append(seed_df)
            logger.info(f"Добавлено {len(seed_df)} эталонных моделей.")

        synthetic_needed = n_samples - len(dfs[0]) if include_seed_models else n_samples
        if synthetic_needed > 0:
            synth_df = self._generate_base_parameters(synthetic_needed)
            synth_df.insert(0, 'Model', [f'UAV_Synth_{i:04d}' for i in range(synthetic_needed)])
            synth_df['Actual_Flight_Time_min'] = self._compute_actual_flight_time(synth_df)
            dfs.append(synth_df)
            logger.info(f"Сгенерировано {synthetic_needed} синтетических моделей.")

        full_df = pd.concat(dfs, ignore_index=True)

        # Добавление производных признаков
        full_df['Power_to_Weight'] = (full_df['Battery_Capacity_mAh'] / full_df['Weight_g']).round(3)
        full_df['Efficiency_Score'] = (
            full_df['Actual_Flight_Time_min'] / full_df['Flight_Time_min']
        ).round(3)  # коэффициент реалистичности заявленного времени

        column_order = [
            'Model', 'Weight_g', 'Max_Speed_ms', 'Battery_Capacity_mAh',
            'Propeller_Size_inch', 'Flight_Time_min', 'Actual_Flight_Time_min',
            'Efficiency_Score', 'Power_to_Weight', 'Range_km', 'Camera_MP',
            'Price_USD', 'Wind_Speed_ms', 'Temperature_C'
        ]
        full_df = full_df[column_order]

        logger.info(f"Итоговый датасет: {len(full_df)} записей.")
        return full_df


if __name__ == "__main__":
    generator = AdvancedUAVDataGenerator(random_seed=123)
    df = generator.generate_full_dataset(n_samples=300, include_seed_models=True)
    print(df.head())
    print(f"\nСгенерировано {len(df)} записей.")
    print(df.describe())
    df.to_csv("uav_dataset_extended.csv", index=False)
    logger.info("Датасет сохранён в 'uav_dataset_extended.csv'")