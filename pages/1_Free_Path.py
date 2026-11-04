import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from models.free_path import simulate_free_path


st.set_page_config(
    page_title="Свободный пробег нейтронов",
    page_icon="☢️",
    layout="wide",
)

st.title("Модель свободного пробега нейтронов")

st.markdown("""
Моделирование длины свободного пробега нейтронов
в однородной среде с использованием метода Монте-Карло.

Теоретическое распределение:

\\[
f(l) = \\Sigma e^{-\\Sigma l}
\\]

Вероятность прохождения расстояния \\(x\\) без взаимодействия:

\\[
P(l > x) = e^{-\\Sigma x}
\\]
""")

st.sidebar.header("Параметры моделирования")

N = st.sidebar.number_input(
    "Количество нейтронов",
    min_value=100,
    max_value=1_000_000,
    value=100_000,
    step=10_000,
)

sigma = st.sidebar.number_input(
    "Макроскопическое сечение Σ (см⁻¹)",
    min_value=0.001,
    max_value=10.0,
    value=0.216,
    step=0.001,
    format="%.4f",
)

distance = st.sidebar.number_input(
    "Расстояние x (см)",
    min_value=0.0,
    max_value=1000.0,
    value=4.5,
    step=0.1,
)

seed = st.sidebar.number_input(
    "Seed генератора",
    min_value=0,
    max_value=1_000_000,
    value=42,
)

if st.button("Запустить моделирование", type="primary"):
    result = simulate_free_path(
        N=int(N),
        sigma=float(sigma),
        distance=float(distance),
        seed=int(seed),
    )

    st.subheader("Результаты моделирования")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Монте-Карло",
        f"{result['probability']:.6f}",
    )

    col2.metric(
        "Аналитическая вероятность",
        f"{result['analytical_probability']:.6f}",
    )

    col3.metric(
        "Стандартная ошибка",
        f"{result['standard_error']:.6f}",
    )

    st.write(
        f"Количество нейтронов без взаимодействия: "
        f"{result['successes']:,} из {result['N']:,}"
    )

    st.write(
        f"Абсолютная погрешность: "
        f"{result['absolute_error']:.6f}"
    )

    st.subheader("Распределение длины свободного пробега")

    fig, ax = plt.subplots(figsize=(10, 5))

    paths = result["free_paths"]

    ax.hist(
        paths,
        bins=100,
        density=True,
        alpha=0.65,
        label="Монте-Карло",
    )

    x = np.linspace(0, np.percentile(paths, 99.5), 500)
    theoretical_pdf = sigma * np.exp(-sigma * x)

    ax.plot(
        x,
        theoretical_pdf,
        linewidth=2,
        label="Теоретическая плотность",
    )

    ax.set_xlabel("Длина свободного пробега l, см")
    ax.set_ylabel("Плотность вероятности")
    ax.set_title("Экспоненциальное распределение")
    ax.grid(alpha=0.3)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Сравнение вероятностей")

    fig, ax = plt.subplots(figsize=(8, 4))

    labels = ["Монте-Карло", "Аналитическая"]
    values = [
        result["probability"],
        result["analytical_probability"],
    ]

    ax.bar(labels, values)
    ax.set_ylabel("Вероятность")
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("Задай параметры слева и нажми «Запустить моделирование».")
