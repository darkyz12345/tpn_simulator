
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import matplotlib.pyplot as plt

from models.beryllium import (
    generate_six_beam_chords,
    simulate_absorption,
    analytical_capture_probability,
    analytical_capture_rate,
    calculate_flux_field,
)

from analysis.plotting import (
    plot_flux_slice,
    plot_capture_slice,
    plot_relative_flux_slice,
)


st.set_page_config(
    page_title="Нейтронный захват в бериллии",
    page_icon="☢️",
    layout="wide",
)

st.title("Модель радиационного захвата нейтронов в бериллиевой сфере")

st.markdown("""
Моделирование взаимодействия шести параллельных нейтронных пучков
со сферической бериллиевой мишенью.

В модели учитывается радиационный захват нейтронов (MT=102).
Рассеяние и изменение энергии нейтронов пока не рассматриваются.

Каждый пучок направлен вдоль одной из осей:
\\(+X, -X, +Y, -Y, +Z, -Z\\).
""")

st.info(
    "Используется макроскопическое сечение радиационного захвата "
    "бериллия при энергии 0.0253 эВ."
)

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

st.sidebar.header("Параметры моделирования")

with st.sidebar.form("beryllium_parameters"):

    st.subheader("Геометрия")

    radius = st.number_input(
        "Радиус сферы R (см)",
        min_value=0.1,
        max_value=100.0,
        value=2.0,
        step=0.5,
    )

    st.subheader("Нейтронные пучки")

    N_per_beam = st.number_input(
        "Количество нейтронов на один пучок",
        min_value=1000,
        max_value=1_000_000,
        value=100_000,
        step=10_000,
    )

    intensity = st.number_input(
        "Интенсивность одного пучка (нейтрон/(см²·с))",
        min_value=1.0,
        max_value=1e12,
        value=3e8,
        step=1e7,
        format="%.0f",
    )

    st.subheader("Ядерные параметры")

    sigma = st.number_input(
        "Макроскопическое сечение Σ (см⁻¹)",
        min_value=0.000001,
        max_value=1.0,
        value=0.001240,
        step=0.00001,
        format="%.6f",
    )

    st.subheader("Вычисления")

    resolution = st.select_slider(
        "Разрешение пространственной сетки",
        options=[31, 51, 71, 101, 151, 201],
        value=101,
    )

    seed = st.number_input(
        "Seed генератора",
        min_value=0,
        max_value=1_000_000,
        value=42,
    )

    submitted = st.form_submit_button(
        "Запустить моделирование",
        type="primary",
    )


# --------------------------------------------------
# SIMULATION
# --------------------------------------------------

if submitted:

    with st.spinner("Выполняется моделирование..."):

        # Generate neutron trajectories through the sphere
        chord_results = generate_six_beam_chords(
            radius=float(radius),
            N_per_beam=int(N_per_beam),
            seed=int(seed),
        )

        # Monte Carlo radiative capture simulation
        result = simulate_absorption(
            chord_results=chord_results,
            sigma=float(sigma),
            radius=float(radius),
            intensity=float(intensity),
            seed=int(seed) + 1,
        )

        # Analytical results
        analytical_probability = analytical_capture_probability(
            radius=float(radius),
            sigma=float(sigma),
        )

        analytical_rate = analytical_capture_rate(
            radius=float(radius),
            sigma=float(sigma),
            intensity=float(intensity),
        )

        # Spatial neutron flux and capture field
        field = calculate_flux_field(
            radius=float(radius),
            sigma=float(sigma),
            intensity=float(intensity),
            resolution=int(resolution),
        )

    # --------------------------------------------------
    # GENERAL RESULTS
    # --------------------------------------------------

    st.header("1. Общие результаты")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Всего нейтронов",
        f"{result['total_N']:,}",
    )

    col2.metric(
        "Радиационный захват",
        f"{result['total_captured']:,}",
    )

    col3.metric(
        "Прошло без захвата",
        f"{result['total_transmitted']:,}",
    )

    st.subheader("Вероятность радиационного захвата")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Монте-Карло",
        f"{result['probability']:.8f}",
    )

    col2.metric(
        "Аналитическая",
        f"{analytical_probability:.8f}",
    )

    col3.metric(
        "Стандартная ошибка",
        f"{result['standard_error']:.8f}",
    )

    difference = abs(
        result["probability"] - analytical_probability
    )

    st.write(f"Абсолютное отклонение: `{difference:.8e}`")

    if result["standard_error"] > 0:
        z_score = (
            result["probability"] - analytical_probability
        ) / result["standard_error"]

        st.write(f"Отклонение в единицах стандартной ошибки: `{z_score:.3f}`")

    # --------------------------------------------------
    # BEAM RESULTS
    # --------------------------------------------------

    st.header("2. Результаты по каждому пучку")

    beam_rows = []

    for beam_name, beam in result["beams"].items():
        beam_rows.append({
            "Пучок": beam_name,
            "Всего нейтронов": beam["N"],
            "Захвачено": beam["captured"],
            "Прошло": beam["transmitted"],
            "P (Монте-Карло)": beam["probability"],
            "P (аналитическая)": beam["analytical_probability"],
            "Стандартная ошибка": beam["standard_error"],
        })

    st.dataframe(
        beam_rows,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------
    # CAPTURE RATE
    # --------------------------------------------------

    st.header("3. Физическая скорость радиационного захвата")

    col1, col2 = st.columns(2)

    col1.metric(
        "Скорость захвата (Монте-Карло)",
        f"{result['capture_rate']:.5e} нейтрон/с",
    )

    col2.metric(
        "Скорость захвата (аналитическая)",
        f"{analytical_rate:.5e} нейтрон/с",
    )

    st.write(
        f"Скорость падающих нейтронов: "
        f"`{result['incident_rate']:.5e} нейтрон/с`"
    )

    rate_error = abs(
        result["capture_rate"] - analytical_rate
    )

    st.write(
        f"Абсолютное отклонение скорости захвата: "
        f"`{rate_error:.5e} нейтрон/с`"
    )

    # --------------------------------------------------
    # SPATIAL DISTRIBUTION
    # --------------------------------------------------

    st.header("4. Пространственное распределение нейтронов")

    st.markdown("""
    Рассматривается центральное сечение сферы в плоскости XY.

    Используемые величины:

    - Нейтронный поток \\(\\Phi(x,y,0)\\).
    - Объёмная плотность скорости радиационного захвата
      \\(q_\\gamma = \\Sigma_\\gamma \\Phi\\).
    - Относительный поток, нормированный на значение в центре сферы.
    """)

    tab1, tab2, tab3 = st.tabs([
        "Нейтронный поток",
        "Плотность захватов",
        "Относительный поток",
    ])

    with tab1:
        fig = plot_flux_slice(field, radius)
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        fig = plot_capture_slice(field, radius)
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        fig = plot_relative_flux_slice(field)
        st.pyplot(fig)
        plt.close(fig)

    # --------------------------------------------------
    # GRID INTEGRATION
    # --------------------------------------------------

    st.header("5. Проверка пространственного интегрирования")

    integrated_rate = field["integrated_capture_rate"]

    col1, col2 = st.columns(2)

    col1.metric(
        "Интеграл по пространственной сетке",
        f"{integrated_rate:.5e} нейтрон/с",
    )

    col2.metric(
        "Аналитическая скорость захвата",
        f"{analytical_rate:.5e} нейтрон/с",
    )

    integration_error = abs(
        integrated_rate - analytical_rate
    ) / analytical_rate * 100

    st.write(
        f"Относительное отклонение пространственного интеграла: "
        f"`{integration_error:.5f}%`"
    )

    st.caption(
        "Отклонение связано в том числе с дискретизацией "
        "сферической области декартовой сеткой."
    )

    # --------------------------------------------------
    # PARAMETERS SUMMARY
    # --------------------------------------------------

    with st.expander("Параметры выполненного моделирования"):

        st.write(f"Радиус сферы: {radius} см")
        st.write(f"Нейтронов на пучок: {N_per_beam:,}")
        st.write(f"Всего пучков: 6")
        st.write(f"Интенсивность одного пучка: {intensity:.5e}")
        st.write(f"Макроскопическое сечение: {sigma:.8f} см⁻¹")
        st.write(f"Разрешение сетки: {resolution}")
        st.write(f"Seed: {seed}")

else:

    st.info(
        "Задай параметры моделирования в боковой панели "
        "и нажми «Запустить моделирование»."
    )
